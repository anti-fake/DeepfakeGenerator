import os, json, argparse
from typing import Dict, List, Optional, Sequence, Set, Tuple
from PIL import Image
from tqdm import tqdm
import torch
import clip
from transformers import BlipForQuestionAnswering, BlipProcessor

Q = {
 "LIGHT_SHADOW":"Are the cast shadows consistent with the main light direction? Answer yes or no.",
 "REFLECTION":"Is the mirror/floor reflection consistent with the subject? Answer yes or no.",
 "SUPPORT":"Are any objects floating or not touching surfaces? Answer yes or no.",
 "INTERPEN":"Do any objects appear to interpenetrate? Answer yes or no.",
 "BRDF":"Do material reflectance properties look physically correct? Answer yes or no.",
 "FACIAL_OPTICS":"Are occlusions around glasses/hair/eyes physically plausible? Answer yes or no.",
 "MOTION":"Is the motion blur consistent between subject and background? Answer yes or no.",
 "ENV":"Are sky, sun position and shadows mutually consistent? Answer yes or no."
}
TXT = {
 "LIGHT_SHADOW":["inconsistent shadows","wrong shadow direction"],
 "REFLECTION":["wrong mirror reflection","inconsistent reflection"],
 "SUPPORT":["objects floating","no contact shadow"],
 "INTERPEN":["interpenetration","object passes through another"],
 "BRDF":["wrong material reflectance","glossy rubber matte metal"],
 "FACIAL_OPTICS":["impossible occlusion with glasses or hair"],
 "MOTION":["inconsistent motion blur"],
 "ENV":["inconsistent time of day lighting"]
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)

def list_image_files(dir_path: str) -> List[str]:
    try:
        return sorted([f for f in os.listdir(dir_path) if is_image_file(f) and os.path.isfile(os.path.join(dir_path, f))])
    except FileNotFoundError:
        return []

def list_subdirs(dir_path: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
    except FileNotFoundError:
        return []

def run_vqa(img_path, q, device, proc, model):
    image = Image.open(img_path).convert("RGB")
    inputs = proc(images=image, text=q, return_tensors="pt").to(device)
    with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=5)
    ans = proc.decode(out[0], skip_special_tokens=True).strip().lower()
    return 1 if ans.startswith("yes") else 0, ans

def run_clip(img_path, texts, device, model, preprocess):
    with torch.no_grad():
        text = clip.tokenize(texts).to(device)
        t = model.encode_text(text); t = t/t.norm(dim=-1, keepdim=True)
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        v = model.encode_image(img); v = v/v.norm(dim=-1, keepdim=True)
        score = (v @ t.T).max().item()
    return float(score)

def _load_existing_autolabel(out_json: str) -> Tuple[List[Dict], Set[str]]:
    if not os.path.isfile(out_json):
        return [], set()
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            rows = json.load(f)
        done = set()
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict) and "filename" in r:
                    done.add(str(r["filename"]))
        return rows if isinstance(rows, list) else [], done
    except Exception:
        # Corrupted/partial file: treat as empty so we can overwrite with a clean one later
        return [], set()

def _process_dir(
    idir: str,
    model_name: str,
    cat: str,
    vdir: Optional[str],
    device: str,
    blip_proc,
    blip,
    clip_model,
    preprocess,
    *,
    overwrite: bool,
    max_images: int,
    out_name: str = "autolabel.json",
):
    image_files = list_image_files(idir)
    if not image_files:
        return

    out_json = os.path.join(idir, out_name)
    existing_rows: List[Dict] = []
    done_files: Set[str] = set()
    if not overwrite:
        existing_rows, done_files = _load_existing_autolabel(out_json)

    to_process = [f for f in image_files if f not in done_files]
    if max_images > 0:
        to_process = to_process[:max_images]

    if not to_process and existing_rows:
        # Nothing new to do
        return

    new_rows: List[Dict] = []
    desc = f"{model_name}/{cat}/{vdir or '.'}"
    for fname in tqdm(to_process, desc=desc):
        path = os.path.join(idir, fname)
        label, ans = run_vqa(path, Q[cat], device, blip_proc, blip)
        score = run_clip(path, TXT[cat], device, clip_model, preprocess)
        new_rows.append(
            {
                "filename": fname,
                "category": cat,
                "model": model_name,
                "variant": vdir,
                "vqa_label": int(label),
                "vqa_answer": ans,
                "clip_score": score,
            }
        )

    all_rows = existing_rows + new_rows
    # Stable order for diffs/debugging
    try:
        all_rows = sorted(all_rows, key=lambda r: str(r.get("filename", "")))
    except Exception:
        pass

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print("saved", out_json, f"(+{len(new_rows)})")

def main(
    root: str,
    *,
    categories: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    overwrite: bool = False,
    max_images: int = 0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cats = list(categories) if categories is not None and len(categories) > 0 else list(Q.keys())

    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device).eval()
    clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)

    if models is None or len(models) == 0:
        model_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    else:
        model_names = list(models)

    for model_name in model_names:
        mdir = os.path.join(root, model_name)
        if not os.path.isdir(mdir):
            continue

        for cat in cats:
            if cat not in Q or cat not in TXT:
                continue

            cdir = os.path.join(mdir, cat)
            if not os.path.isdir(cdir):
                continue

            # Layout A (flat): <root>/<model>/<cat>/*.png
            direct_images = list_image_files(cdir)
            if direct_images:
                _process_dir(
                    cdir,
                    model_name,
                    cat,
                    None,
                    device,
                    blip_proc,
                    blip,
                    clip_model,
                    preprocess,
                    overwrite=overwrite,
                    max_images=max_images,
                )

            # Layout B (variant): <root>/<model>/<cat>/<variant>/*.png
            for vdir in list_subdirs(cdir):
                idir = os.path.join(cdir, vdir)
                _process_dir(
                    idir,
                    model_name,
                    cat,
                    vdir,
                    device,
                    blip_proc,
                    blip,
                    clip_model,
                    preprocess,
                    overwrite=overwrite,
                    max_images=max_images,
                )

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images_root",
        default="outputs_t2i_mscoco2017",
        help="Generated images root. Expected layouts:\n"
             "  A) <root>/<model>/<category>/*.(png|jpg|jpeg|webp)\n"
             "  B) <root>/<model>/<category>/<variant>/*.(png|jpg|jpeg|webp)"
    )
    ap.add_argument("--device", default=None, help="cuda/cpu/cuda:{id} (default: auto)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing autolabel.json (default: resume)")
    ap.add_argument("--max_images", type=int, default=0, help="Process at most N new images per folder (0 = all)")
    ap.add_argument("--models", nargs="*", default=None, help="Optional model folder names to process (default: all)")
    ap.add_argument("--categories", nargs="*", default=None, help="Optional category names to process (default: all known)")
    args = ap.parse_args()
    main(
        args.images_root,
        categories=args.categories,
        models=args.models,
        device=args.device,
        overwrite=args.overwrite,
        max_images=args.max_images,
    )
