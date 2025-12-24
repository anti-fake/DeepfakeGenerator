import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from diffusers import AutoPipelineForInpainting

def process_worker(gpu_id, image_list, args):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    pbar = tqdm(total=len(image_list), desc=f"GPU {gpu_id}", position=gpu_id)

    for img_path in image_list:
        base_name = os.path.basename(img_path)
        stem = os.path.splitext(base_name)[0]
        mask_path = os.path.join(args.mask_dir, f"{stem}_mask.png")
        save_path = os.path.join(args.output_dir, f"{stem}_inpainted.png")

        if not os.path.exists(mask_path): continue
        
        init_img = Image.open(img_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        orig_w, orig_h = init_img.size

        # Inpaint Logic
        init_512 = init_img.resize((512, 512), Image.BICUBIC)
        mask_512 = mask_img.resize((512, 512), Image.NEAREST)
        mask_512_blur = pipeline.mask_processor.blur(mask_512, blur_factor=33)

        result_512 = pipeline(prompt="", image=init_512, mask_image=mask_512_blur).images[0]

        # Overlay & Save
        res_np = np.array(result_512).astype(np.float32)
        orig_np = np.array(init_512).astype(np.float32)
        m_np = np.array(mask_512_blur).astype(np.float32)[..., None] / 255.0
        
        final_512 = Image.fromarray((res_np * m_np + orig_np * (1-m_np)).astype(np.uint8))
        final_512.resize((orig_w, orig_h), Image.BICUBIC).save(save_path)
        pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    imgs = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    mp.set_start_method("spawn", force=True)
    chunks = np.array_split(imgs, args.num_gpus)
    procs = [mp.Process(target=process_worker, args=(i, chunks[i].tolist(), args)) for i in range(args.num_gpus)]
    for p in procs: p.start()
    for p in procs: p.join()

if __name__ == "__main__":
    main()