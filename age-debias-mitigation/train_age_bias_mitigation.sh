#!/bin/bash

# This script runs the CXR-AnoFAIR training process with age distribution alignment

# Set the base model path - using standard Stable Diffusion v1-5
export BASE_MODEL="runwayml/stable-diffusion-v1-5"

# Output directory for saving models and logs
export OUTPUT_DIR="./outputs"

# Set attribute classifier path (pre-trained for age classification)
export ATTRIBUTE_CLASSIFIER_PATH="./models/cxr_age_classifier.pt"

# Set path for prompt templates
export PROMPT_TEMPLATES_PATH="./configs/cxr_prompt_bias_mitigation.json"

# Run the training with appropriate parameters
accelerate launch ./age-debias-mitigation/train_age_bias_mitigation.py \
  --mixed_precision="fp16" \
  --multi_gpu \
  --pretrained_model_name_or_path=$BASE_MODEL \
  --train_data_dir=$TRAIN_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --learning_rate=4e-5 \
  --distributional_alignment_weight=1.0 \
  --semantic_preservation_weight=1.0 \
  --diagnostic_preservation_weight=1.0 \
  --feature_similarity_weight=0.3 \
  --perceptual_weight=0.3 \
  --diagnostic_consistency_weight=0.4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --lora_rank=8 \
  --resolution=512 \
  --target_age_ratio=0.5 \
  --age_classifier_path=$ATTRIBUTE_CLASSIFIER_PATH \
  --prompt_templates_path=$PROMPT_TEMPLATES_PATH \
  --enable_xformers_memory_efficient_attention \
  --uncertainty_threshold=0.2 \
  --dynamic_weight_factor=0.4 \
  --seed=0
