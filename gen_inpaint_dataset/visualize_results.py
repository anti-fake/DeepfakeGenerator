import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def create_comparison_grid(orig_path, mask_path, result_path, save_path):
    """
    세 장의 이미지를 로드하여 상단에 제목을 포함한 [Original | Masked Input | Inpainted] 그리드 생성
    """
    # 1. 이미지 로드
    orig_img = Image.open(orig_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")
    result_img = Image.open(result_path).convert("RGB")

    # 2. 크기 통일 (원본 기준)
    w, h = orig_img.size
    
    # 상단 텍스트가 들어갈 여백 설정 (예: 60픽셀)
    header_h = 60
    
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), Image.NEAREST)
    if result_img.size != (w, h):
        result_img = result_img.resize((w, h), Image.BICUBIC)

    # 3. Masked Input 생성 (원본 위에 마스크 영역을 검게 처리)
    orig_np = np.array(orig_img)
    mask_np = np.array(mask_img)
    mask_bool = mask_np > 127
    masked_input_np = orig_np.copy()
    masked_input_np[mask_bool] = 0 
    masked_input_img = Image.fromarray(masked_input_np)

    # 4. 캔버스 생성 (높이에 header_h 추가)
    grid_img = Image.new('RGB', (w * 3, h + header_h), (255, 255, 255)) # 배경 흰색

    # 5. 이미지 붙여넣기 (Y축 좌표를 header_h 만큼 내림)
    grid_img.paste(orig_img, (0, header_h))
    grid_img.paste(masked_input_img, (w, header_h))
    grid_img.paste(result_img, (w * 2, header_h))

    # 6. 텍스트 라벨 추가
    draw = ImageDraw.Draw(grid_img)
    
    # 폰트 설정 (서버 환경에 따라 경로가 다를 수 있음)
    font_size = 30
    try:
        # 일반적인 리눅스 서버의 폰트 경로 시도
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not os.path.exists(font_path):
            # 폰트가 없을 경우 다른 경로 시도 (Ubuntu 등)
            font_path = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
            
        font = ImageFont.truetype(font_path, font_size)
    except:
        # 폰트 로드 실패 시 기본 폰트 사용
        font = ImageFont.load_default()

    labels = ["Original", "Masked Input", "Inpainted Result"]
    
    for i, label in enumerate(labels):
        # 텍스트 중앙 정렬을 위한 좌표 계산
        # 텍스트 박스 크기 구하기
        if hasattr(draw, 'textbbox'): # 최신 Pillow 버전
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else: # 구버전 Pillow
            text_w, text_h = draw.textsize(label, font=font)
            
        x = (w * i) + (w // 2) - (text_w // 2)
        y = (header_h // 2) - (text_h // 2)
        
        draw.text((x, y), label, fill=(0, 0, 0), font=font)

    # 7. 저장
    grid_img.save(save_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Visualize Inpainting Results with Titles")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to original images")
    parser.add_argument('--mask_dir', type=str, required=True, help="Path to generated masks")
    parser.add_argument('--result_dir', type=str, required=True, help="Path to inpainted results")
    parser.add_argument('--vis_dir', type=str, default="./data/visualization", help="Path to save comparison images")
    args = parser.parse_args()

    os.makedirs(args.vis_dir, exist_ok=True)

    # 결과 폴더 스캔
    exts = ('.png', '.jpg', '.jpeg')
    result_files = [f for f in os.listdir(args.result_dir) if f.lower().endswith(exts)]
    
    print(f"Found {len(result_files)} result images. Starting visualization...")

    processed_count = 0
    for res_f in tqdm(result_files):
        # 파일명에서 stem(기본이름) 추출
        stem = res_f.replace("_inpainted.png", "").replace("_inpainted.jpg", "").replace("_inpaint.png", "")
        
        # 원본 이미지 찾기
        orig_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
             temp_path = os.path.join(args.input_dir, f"{stem}{ext}")
             if os.path.exists(temp_path):
                 orig_path = temp_path
                 break
        
        # 마스크 이미지 찾기 (이름 규칙에 따라 수정 가능)
        mask_path = os.path.join(args.mask_dir, f"{stem}_mask.png")
        result_path = os.path.join(args.result_dir, res_f)

        if orig_path and os.path.exists(mask_path) and os.path.exists(result_path):
            save_path = os.path.join(args.vis_dir, f"{stem}_compare.jpg")
            try:
                create_comparison_grid(orig_path, mask_path, result_path, save_path)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {stem}: {e}")
                
    print(f"\nDone! {processed_count} images saved to: {args.vis_dir}")

if __name__ == "__main__":
    main()