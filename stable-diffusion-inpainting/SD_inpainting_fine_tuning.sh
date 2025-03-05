#!/bin/bash

# Set environment variables for the training
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export CXR_DATA_ROOT="path/to/chest_xray_dataset"
export METADATA_PATH="./configs/cxr_prompt_SD_inpainting.json"
export OUTPUT_DIR="CXR-AnoFAIR-inpainting-model"

# Run the training with accelerate
accelerate launch ./stable-diffusion-inpainting/SD_inpainting_fine_tuning.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_root=$CXR_DATA_ROOT \
  --metadata_path=$METADATA_PATH \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=4 \
  --learning_rate=5e-5 \
  --gradient_accumulation_steps=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=50000 \
  --checkpointing_steps=5000
