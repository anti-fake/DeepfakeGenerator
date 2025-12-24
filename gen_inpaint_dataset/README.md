# Image Inpainting Dataset Generator

이미지 인페인팅(inpainting) 데이터셋을 생성하는 파이프라인
* Stable Diffusion Inpainting 1.5 모델을 활용하여 원본 이미지에서 마스크 영역을 자동으로 복원한 데이터셋을 생성

## 주요 기능

- 🎯 **자동 마스크 생성**: 랜덤한 직사각형 마스크 자동 생성
- 🚀 **멀티 GPU 지원**: 대량 이미지 처리를 위한 멀티 GPU 병렬 처리
- 🖼️ **시각화**: 원본/마스크/인페인팅 결과를 한 눈에 비교할 수 있는 그리드 이미지 생성
- 📝 **간편한 실행**: Shell 스크립트를 통해 원하는 셋팅으로 실행

## 프로젝트 구조

```
gen_inpaint_dataset/
├── gen_mask.py              # 랜덤 마스크 생성
├── run_inpaint.py           # Stable Diffusion 인페인팅 실행
├── visualize_results.py     # 결과 시각화 그리드 생성
├── run_pipeline.sh          # 전체 파이프라인 실행 스크립트
├── run_visualization.sh     # 시각화 실행 스크립트
├── samples/                 # 입력 이미지 디렉토리
└── data/                    # 출력 데이터 디렉토리
    ├── masks/              # 생성된 마스크 이미지
    ├── results/            # 인페인팅 결과 이미지
    └── visualization/      # 비교 시각화 이미지
```

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/[your-username]/gen_inpaint_dataset.git
cd gen_inpaint_dataset
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. Hugging Face 로그인 (선택사항)
```bash
huggingface-cli login
```

## 사용 방법

### 빠른 시작 (전체 파이프라인 실행)

1. 입력 이미지를 `samples/` 디렉토리에 추가합니다.

2. 전체 파이프라인을 실행합니다:
```bash
bash run_pipeline.sh
```

이 스크립트는 다음 작업을 순차적으로 수행합니다:
- 마스크 생성
- 인페인팅 실행 (멀티 GPU)
- 완료 알림

### 개별 실행

#### 1. 마스크 생성만
```bash
python gen_mask.py --input_dir ./samples --mask_dir ./data/masks
```

#### 2. 인페인팅 실행
```bash
python run_inpaint.py \
    --input_dir ./samples \
    --mask_dir ./data/masks \
    --output_dir ./data/results \
    --num_gpus 2
```

#### 3. 결과 시각화
```bash
python visualize_results.py \
    --input_dir ./samples \
    --mask_dir ./data/masks \
    --result_dir ./data/results \
    --vis_dir ./data/visualization
```

또는 시각화 스크립트 사용:
```bash
bash run_visualization.sh
```

## 파라미터 설명

### gen_mask.py
- `--input_dir`: 원본 이미지가 있는 디렉토리
- `--mask_dir`: 마스크를 저장할 디렉토리

### run_inpaint.py
- `--input_dir`: 원본 이미지 디렉토리
- `--mask_dir`: 마스크 이미지 디렉토리
- `--output_dir`: 인페인팅 결과 저장 디렉토리
- `--num_gpus`: 사용할 GPU 개수 (기본값: 1)

### visualize_results.py
- `--input_dir`: 원본 이미지 디렉토리
- `--mask_dir`: 마스크 이미지 디렉토리
- `--result_dir`: 인페인팅 결과 디렉토리
- `--vis_dir`: 시각화 결과 저장 디렉토리
