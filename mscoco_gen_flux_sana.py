#!/usr/bin/env python3
import os, yaml, time, random, argparse, re
from typing import Dict, List, Tuple
import torch
import numpy as np
import pickle as pkl
from PIL import Image
from tqdm import tqdm
from diffusers import AutoPipelineForText2Image
import multiprocessing as mp
from functools import partial
import warnings

def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p:str): os.makedirs(p, exist_ok=True)

def validate_and_fix_image(image: Image.Image) -> Tuple[Image.Image, bool]:
    """
    Validate image for NaN/Inf values and fix if necessary.
    
    Returns:
        Tuple of (fixed_image, is_valid)
        is_valid is False if the image had NaN/Inf values
    """
    img_array = np.array(image, dtype=np.float32)
    
    # Check for NaN or Inf values
    has_nan = np.isnan(img_array).any()
    has_inf = np.isinf(img_array).any()
    
    if has_nan or has_inf:
        # Replace NaN with 0 and Inf with max/min values
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)
        # Clamp to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array), False
    
    return image, True

def batch_prompts(prompts: List[Tuple[int, str]], batch_size: int) -> List[List[Tuple[int, str]]]:
    """Split prompts into batches of specified size."""
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    return batches

def batch_mixed_prompts(prompts: List[Tuple[str, str, int, str]], batch_size: int) -> List[List[Tuple[str, str, int, str]]]:
    """
    Split mixed prompts (from multiple categories) into batches.
    
    Args:
        prompts: List of (cat, outdir, idx, caption) tuples
        batch_size: Maximum batch size
    
    Returns:
        List of batches, each batch is a list of (cat, outdir, idx, caption) tuples
    """
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    return batches

def load_pipe(hub_id:str, dtype_str:str, model_cfg:Dict, gpu_id:int=None):
    """
    Load pipeline on a specific GPU.
    
    Args:
        hub_id: HuggingFace model hub ID
        dtype_str: Data type string ("bfloat16" or "float16")
        model_cfg: Model configuration dictionary
        gpu_id: Specific GPU ID to use (if None, uses default CUDA device)
    """
    dtype = torch.bfloat16 if dtype_str=="bfloat16" and torch.cuda.is_available() else torch.float16
    
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = dict(torch_dtype=dtype, safety_checker=None, add_watermarker=False)
    pipe = AutoPipelineForText2Image.from_pretrained(hub_id, **kwargs).to(device)

    # Memory optimizations (best-effort)
    try:
        if model_cfg.get("attention_slicing", True): pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        if model_cfg.get("vae_slicing", True): pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        if model_cfg.get("vae_tiling", True): pipe.enable_vae_tiling()
    except Exception:
        pass

    return pipe, device

def existing_indices_for_variant(out_dir: str, cat: str, mkey: str, variant: str) -> set:
    existing: set = set()
    if not os.path.isdir(out_dir):
        return existing
    # Robust: rely only on the numeric suffix pattern anywhere in the filename
    suffix_pattern = re.compile(r"__([0-9]{5})\\.png$")
    try:
        for fname in os.listdir(out_dir):
            m = suffix_pattern.search(fname)
            if m:
                try:
                    existing.add(int(m.group(1)))
                except Exception:
                    pass
    except Exception:
        pass
    return existing

def count_png_in_dir(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    try:
        return sum(1 for f in os.listdir(out_dir) if f.lower().endswith('.png'))
    except Exception:
        return 0


def get_gpu_ids() -> List[int]:
    """
    Get list of available GPU IDs.
    
    When CUDA_VISIBLE_DEVICES is set, CUDA remaps the physical GPUs to logical 
    indices starting from 0. For example, if CUDA_VISIBLE_DEVICES=3,4, then:
      - Physical GPU 3 -> Logical GPU 0
      - Physical GPU 4 -> Logical GPU 1
    
    This function returns the logical indices that should be used with torch.cuda.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None and cuda_visible.strip():
        # CUDA_VISIBLE_DEVICES가 설정되면 GPU들이 0, 1, 2, ...로 재매핑됨
        # 따라서 논리적 인덱스를 반환해야 함
        num_visible = len([x.strip() for x in cuda_visible.split(",") if x.strip().isdigit()])
        return list(range(num_visible)) if num_visible > 0 else [0]
    else:
        # Return all available GPUs
        num_gpus = torch.cuda.device_count()
        return list(range(num_gpus)) if num_gpus > 0 else [0]


def worker_process(
    gpu_id: int,
    assigned_tasks: List[Tuple[str, str, List[Tuple[int, str]]]],
    mkey: str,
    model_cfg: Dict,
    hub_id: str,
    common_batch_size: int,
    interleaved: bool = True
):
    """
    Worker function that runs on a specific GPU and processes assigned categories.
    
    Args:
        gpu_id: GPU ID to use for this worker
        assigned_tasks: List of (cat, outdir, missing_prompts) tuples
        mkey: Model key name
        model_cfg: Model configuration
        hub_id: HuggingFace model hub ID
        common_batch_size: Default batch size
        interleaved: If True, mix prompts from all categories for efficient batching
    """
    if not assigned_tasks:
        return
    
    device_id = gpu_id
    
    print(f"[worker GPU:{gpu_id}] Starting with {len(assigned_tasks)} categories (interleaved={interleaved})")
    
    try:
        pipe, device = load_pipe(hub_id, model_cfg.get("dtype", "float16"), model_cfg, gpu_id=device_id)
    except Exception as e:
        print(f"[worker GPU:{gpu_id}] Failed to load model: {e}")
        return
    
    batch_size = model_cfg.get("batch_size", common_batch_size)
    
    if interleaved:
        # Interleaved mode: Mix prompts from all categories into single batches
        # This maximizes GPU utilization by always having full batches
        
        # Flatten all tasks into a single list: (cat, outdir, idx, caption)
        all_prompts = []
        for cat, outdir, missing_prompts in assigned_tasks:
            ensure_dir(outdir)
            for idx, caption in missing_prompts:
                all_prompts.append((cat, outdir, idx, caption))
        
        total_images = len(all_prompts)
        batches = batch_mixed_prompts(all_prompts, batch_size)
        total_batches = len(batches)
        
        # Count per category for logging
        cat_counts = {}
        for cat, _, _, _ in all_prompts:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        print(f"[worker GPU:{gpu_id}] {mkey}: {total_images} images in {total_batches} batches (mixed from {len(assigned_tasks)} categories)")
        for cat, count in cat_counts.items():
            print(f"  - {cat}: {count} images")
        
        invalid_count = 0
        for batch in tqdm(batches, desc=f"GPU:{gpu_id}|{mkey}:mixed", leave=False):
            captions = [caption for _, _, _, caption in batch]
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in cast")
                images = pipe(
                    prompt=captions,
                    height=model_cfg.get("height", 1024),
                    width=model_cfg.get("width", 1024),
                    num_inference_steps=model_cfg.get("steps", 30),
                    guidance_scale=model_cfg.get("guidance_scale", 5.0)
                ).images
            
            # Route each image to its correct output directory
            for (cat, outdir, idx, _), image in zip(batch, images):
                fixed_image, is_valid = validate_and_fix_image(image)
                if not is_valid:
                    invalid_count += 1
                fixed_image.save(os.path.join(outdir, f"{cat}__{mkey}__{idx:05d}.png"))
        
        if invalid_count > 0:
            print(f"[warning GPU:{gpu_id}] {mkey} - {invalid_count} images had NaN/Inf values (fixed)")
    
    else:
        # Original mode: Process each category separately
        for cat, outdir, missing_prompts in assigned_tasks:
            ensure_dir(outdir)
            batches = batch_prompts(missing_prompts, batch_size)
            total_batches = len(batches)
            total_images = len(missing_prompts)
            
            print(f"[worker GPU:{gpu_id}] {mkey}:{cat} - {total_images} images in {total_batches} batches")
            
            invalid_count = 0
            for batch_idx, batch in enumerate(tqdm(batches, desc=f"GPU:{gpu_id}|{mkey}:{cat}", leave=False), 1):
                indices = [idx for idx, _ in batch]
                captions = [caption for _, caption in batch]
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="invalid value encountered in cast")
                    images = pipe(
                        prompt=captions,
                        height=model_cfg.get("height", 1024),
                        width=model_cfg.get("width", 1024),
                        num_inference_steps=model_cfg.get("steps", 30),
                        guidance_scale=model_cfg.get("guidance_scale", 5.0)
                    ).images
                
                for idx, image in zip(indices, images):
                    fixed_image, is_valid = validate_and_fix_image(image)
                    if not is_valid:
                        invalid_count += 1
                    fixed_image.save(os.path.join(outdir, f"{cat}__{mkey}__{idx:05d}.png"))
            
            if invalid_count > 0:
                print(f"[warning GPU:{gpu_id}] {mkey}:{cat} - {invalid_count} images had NaN/Inf values (fixed)")
            print(f"[worker GPU:{gpu_id}] Completed {mkey}:{cat}")
    
    print(f"[worker GPU:{gpu_id}] All tasks completed")
    
    # Clean up
    del pipe
    torch.cuda.empty_cache()


def distribute_tasks(tasks: List, num_workers: int) -> List[List]:
    """
    Distribute tasks evenly among workers using round-robin.
    
    Args:
        tasks: List of tasks to distribute
        num_workers: Number of workers
    
    Returns:
        List of task lists for each worker
    """
    distributed = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        distributed[i % num_workers].append(task)
    return distributed


def main(models_yaml_path:str, prompts_category_path, one_per_base:bool=False, num_parallel:int=1, interleaved:bool=True):
    with open(models_yaml_path, "r", encoding="utf-8") as f: mcfg = yaml.safe_load(f)
    with open(prompts_category_path, "rb") as f: prompts_category = pkl.load(f)

    out_root = mcfg["common"]["out_root"]
    common_batch_size = mcfg["common"].get("batch_size", 1)
    if mcfg["common"].get("per_variant") is not None:
        print("[info] common.per_variant 설정이 감지되었지만 전체 프롬프트를 사용하기 위해 무시합니다.")
    set_seed(mcfg["common"].get("seed",1234))
    cat_to_prompts: Dict[str, List] = {}
    
    # Get available GPU IDs
    available_gpus = get_gpu_ids()
    num_gpus = len(available_gpus)
    
    # Allow multiple workers per GPU for better VRAM utilization
    actual_parallel = num_parallel
    
    print(f"[info] Available GPUs: {available_gpus}")
    print(f"[info] Using {actual_parallel} parallel workers ({actual_parallel // num_gpus} per GPU, {actual_parallel % num_gpus} extra)")
    
    seen_bases = set()
    if one_per_base:
        print("[info] one_per_base option enabled: Filtering duplicates based on base prompt.")

    for idx, cap_cat in enumerate(prompts_category):
        full_caption = cap_cat["caption"]
        
        if one_per_base:
            # Try to extract base prompt assuming "base, induce" format
            parts = full_caption.rsplit(", ", 1)
            base_prompt = parts[0] if len(parts) > 1 else full_caption
            
            if base_prompt in seen_bases:
                continue
            seen_bases.add(base_prompt)

        cat = cap_cat["category"]
        cat_to_prompts.setdefault(cat, []).append((idx, full_caption))

    for mkey, m in mcfg["models"].items():
        hub_id = (m.get("hub_id") or "").strip()
        if not hub_id and m.get("optional", False):
            print(f"[skip] {mkey}: hub_id 비어있음(optional)")
            continue

        planned_tasks = []
        plan_logs = []
        if m.get("per_variant") is not None:
            print(f"[info] {mkey}.per_variant 설정이 감지되었지만 전체 프롬프트를 사용하기 위해 무시합니다.")

        for cat, prompts in cat_to_prompts.items():
            if not prompts:
                continue
            target_prompts = prompts  # per_variant 제한을 제거하여 모든 프롬프트를 생성
            target_total = len(target_prompts)
            if target_total == 0:
                continue

            outdir = os.path.join(out_root, mkey, cat)
            existing_png_count = count_png_in_dir(outdir)
            if existing_png_count >= target_total:
                plan_logs.append(f"[skip] {mkey}:{cat} -> {outdir} 이미 {existing_png_count}/{target_total}개 존재")
                continue

            existing_indices = existing_indices_for_variant(outdir, cat, mkey, "mscoco")
            idx_to_caption = dict(target_prompts)
            missing_indices = [idx for idx in idx_to_caption.keys() if idx not in existing_indices]

            if not missing_indices:
                plan_logs.append(f"[skip] {mkey}:{cat} -> {outdir} 필요한 인덱스 없음 (기존 png {existing_png_count}/{target_total})")
                continue

            missing_prompts = [(idx, idx_to_caption[idx]) for idx in sorted(missing_indices)]
            planned_tasks.append((cat, outdir, missing_prompts))
            plan_logs.append(
                f"[plan] {mkey}:{cat} -> {outdir} 생성 {len(missing_prompts)}/{target_total} (기존 png {existing_png_count})"
            )

        if plan_logs:
            for log in plan_logs:
                print(log)

        if not planned_tasks:
            print(f"[skip] {mkey}: 모든 카테고리가 이미 생성됨 -> {os.path.join(out_root, mkey)}")
            continue

        # Decide whether to use parallel or sequential processing
        if actual_parallel > 1 and len(planned_tasks) > 1:
            # Parallel processing: distribute tasks among workers
            num_workers = min(actual_parallel, len(planned_tasks))
            
            # Assign GPUs to workers using round-robin (allows multiple workers per GPU)
            worker_gpus = [available_gpus[i % num_gpus] for i in range(num_workers)]
            distributed_tasks = distribute_tasks(planned_tasks, num_workers)
            
            # Group workers by GPU for logging
            gpu_worker_info = {}
            for i, gpu_id in enumerate(worker_gpus):
                if gpu_id not in gpu_worker_info:
                    gpu_worker_info[gpu_id] = []
                gpu_worker_info[gpu_id].append(i)
            
            print(f"[info] {mkey}: {num_workers} workers on {num_gpus} GPU(s)")
            for gpu_id, worker_ids in gpu_worker_info.items():
                print(f"  - GPU {gpu_id}: {len(worker_ids)} worker(s)")
            
            for i, (gpu_id, tasks) in enumerate(zip(worker_gpus, distributed_tasks)):
                cats = [t[0] for t in tasks]
                print(f"  - Worker {i} (GPU {gpu_id}): {len(tasks)} categories -> {cats}")
            
            # Create and start worker processes
            processes = []
            mp.set_start_method('spawn', force=True)
            
            for gpu_id, tasks in zip(worker_gpus, distributed_tasks):
                if not tasks:
                    continue
                p = mp.Process(
                    target=worker_process,
                    args=(gpu_id, tasks, mkey, m, hub_id, common_batch_size, interleaved)
                )
                processes.append(p)
                p.start()
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
            
            print(f"[done] {mkey} -> {out_root}/{mkey} (parallel processing completed)")
        
        else:
            # Sequential processing (single GPU or single task)
            gpu_id = available_gpus[0] if available_gpus else 0
            print(f"[info] {mkey}: Processing on GPU {gpu_id} (interleaved={interleaved})")
            
            try:
                pipe, device = load_pipe(hub_id, m.get("dtype","float16"), m, gpu_id=gpu_id)
            except Exception as e:
                if m.get("optional", False):
                    print(f"[skip] {mkey}: '{hub_id}' 로드 실패(optional) -> {e}")
                    continue
                raise

            batch_size = m.get("batch_size", common_batch_size)
            print(f"[info] {mkey}: batch_size={batch_size}")
            
            if interleaved and len(planned_tasks) > 1:
                # Interleaved mode: Mix prompts from all categories
                all_prompts = []
                for cat, outdir, missing_prompts in planned_tasks:
                    ensure_dir(outdir)
                    for idx, caption in missing_prompts:
                        all_prompts.append((cat, outdir, idx, caption))
                
                total_images = len(all_prompts)
                batches = batch_mixed_prompts(all_prompts, batch_size)
                total_batches = len(batches)
                
                # Count per category
                cat_counts = {}
                for cat, _, _, _ in all_prompts:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                
                print(f"[info] {mkey}: {total_images} images in {total_batches} batches (mixed from {len(planned_tasks)} categories)")
                for cat, count in cat_counts.items():
                    print(f"  - {cat}: {count} images")
                
                invalid_count = 0
                for batch in tqdm(batches, desc=f"{mkey}:mixed", leave=False):
                    captions = [caption for _, _, _, caption in batch]
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="invalid value encountered in cast")
                        images = pipe(
                            prompt=captions,
                            height=m.get("height",1024), width=m.get("width",1024),
                            num_inference_steps=m.get("steps",30),
                            guidance_scale=m.get("guidance_scale",5.0)
                        ).images
                    
                    for (cat, outdir, idx, _), image in zip(batch, images):
                        fixed_image, is_valid = validate_and_fix_image(image)
                        if not is_valid:
                            invalid_count += 1
                        fixed_image.save(os.path.join(outdir, f"{cat}__{mkey}__{idx:05d}.png"))
                
                if invalid_count > 0:
                    print(f"[warning] {mkey} - {invalid_count} images had NaN/Inf values (fixed)")
            
            else:
                # Original mode: Process each category separately
                for cat, outdir, missing_prompts in planned_tasks:
                    ensure_dir(outdir)
                    batches = batch_prompts(missing_prompts, batch_size)
                    total_batches = len(batches)
                    total_images = len(missing_prompts)
                    invalid_count = 0
                    
                    print(f"[info] {mkey}:{cat} - {total_images} images in {total_batches} batches")
                    for batch_idx, batch in enumerate(tqdm(batches, desc=f"{mkey}:{cat}", leave=False), 1):
                        indices = [idx for idx, _ in batch]
                        captions = [caption for _, caption in batch]
                        
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="invalid value encountered in cast")
                            images = pipe(
                                prompt=captions,
                                height=m.get("height",1024), width=m.get("width",1024),
                                num_inference_steps=m.get("steps",30),
                                guidance_scale=m.get("guidance_scale",5.0)
                            ).images
                        
                        for idx, image in zip(indices, images):
                            fixed_image, is_valid = validate_and_fix_image(image)
                            if not is_valid:
                                invalid_count += 1
                            fixed_image.save(os.path.join(outdir, f"{cat}__{mkey}__{idx:05d}.png"))
                    
                    if invalid_count > 0:
                        print(f"[warning] {mkey}:{cat} - {invalid_count} images had NaN/Inf values (fixed)")
            
            # Clean up
            del pipe
            torch.cuda.empty_cache()
            print(f"[done] {mkey} -> {out_root}/{mkey}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate images for MSCOCO categories with optional multi-GPU parallel processing."
    )
    ap.add_argument("--models_yaml_path", default="/home/nas5/chanheekang/etri_deepfake/kaist_year1/configs/model_t2i.yaml")
    ap.add_argument("--prompts_category_path", default="/home/nas5/chanheekang/etri_deepfake/kaist_year1/outputs/error_captions_2017_1perimage.pkl")
    ap.add_argument("--one_per_base", action="store_true", help="Enable to generate only one image per unique base prompt.")
    ap.add_argument(
        "--num_parallel", 
        type=int, 
        default=1, 
        help="Number of parallel worker processes. Can exceed GPU count for better VRAM utilization. "
             "Workers are distributed across GPUs using round-robin. "
             "Example: 4 workers on 2 GPUs = 2 workers per GPU. Default: 1"
    )
    ap.add_argument(
        "--interleaved",
        action="store_true",
        default=True,
        help="Mix prompts from multiple categories into single batches for efficient VRAM usage. Default: True"
    )
    ap.add_argument(
        "--no-interleaved",
        action="store_false",
        dest="interleaved",
        help="Process each category separately (original behavior)"
    )
    args = ap.parse_args()
    main(args.models_yaml_path, args.prompts_category_path, args.one_per_base, args.num_parallel, args.interleaved)




