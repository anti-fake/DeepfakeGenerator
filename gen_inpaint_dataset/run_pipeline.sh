#!/bin/bash

# 경로 설정
INPUT_DIR="./samples"
MASK_DIR="./data/masks"
OUTPUT_DIR="./data/results"
NUM_GPUS=2

echo "Step 1: Generating Masks..."
python gen_mask.py --input_dir $INPUT_DIR --mask_dir $MASK_DIR

echo "Step 2: Running Inpainting..."
# CUDA_VISIBLE_DEVICES를 지정하여 특정 GPU만 할당할 수도 있습니다.
CUDA_VISIBLE_DEVICES=6,7 python run_inpaint.py \
    --input_dir $INPUT_DIR \
    --mask_dir $MASK_DIR \
    --output_dir $OUTPUT_DIR \
    --num_gpus $NUM_GPUS

echo "All process done!"