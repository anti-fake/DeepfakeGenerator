# DeepfakeGenerator

> **DeepfakeGenerator**는 딥페이크(합성/조작) 연구를 위한 **데이터 생성 및 자동 라벨링(autolabel) 유틸리티 모음**입니다.  
> 현재는 특히 **이미지 인페인팅(inpainting) 기반 조작 데이터셋 생성 파이프라인**을 중심으로 구성되어 있습니다.
---

### 1) Autolabel (자동 라벨링)
- 데이터 생성 과정에서 필요한 **마스크/영역/메타데이터 라벨**을 자동으로 생성/정리하기 위한 스크립트들을 포함합니다.
- 예: `02_autolabel.py` :contentReference[oaicite:2]{index=2}

### 2) Image Inpainting Dataset Generator
- * Code: [`gen_inpaint_dataset`](https://github.com/anti-fake/DeepfakeGenerator/tree/main/gen_inpaint_dataset)
- * Description: 이미지 인페인팅(inpainting) 데이터셋을 생성하는 파이프라인
- * Main Source: [stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) 기반
