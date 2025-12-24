import os
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

def generate_random_rect_mask(width, height, min_area_ratio=0.02, max_area_ratio=0.2):
    img_area = width * height
    target_area = random.uniform(min_area_ratio, max_area_ratio) * img_area
    aspect_ratio = random.uniform(0.5, 2.0)

    h = int(round((target_area / aspect_ratio) ** 0.5))
    w = int(round((target_area * aspect_ratio) ** 0.5))
    h, w = min(h, height), min(w, width)

    top = random.randint(0, height - h)
    left = random.randint(0, width - w)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:top + h, left:left + w] = 255
    return Image.fromarray(mask, mode="L")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.mask_dir, exist_ok=True)
    imgs = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for f in tqdm(imgs, desc="Generating Masks"):
        img_path = os.path.join(args.input_dir, f)
        with Image.open(img_path) as img:
            w, h = img.size
            mask = generate_random_rect_mask(w, h)
            # 원본 파일명과 매칭되도록 저장 (예: image1.png -> image1_mask.png)
            mask.save(os.path.join(args.mask_dir, f"{os.path.splitext(f)[0]}_mask.png"))

if __name__ == "__main__":
    main()