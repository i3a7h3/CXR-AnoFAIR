### 1. Train for the Inpainting Model

#### **Training with CXR Diagnostic Preservation Loss**

Our **CXR Diagnostic Preservation Loss** combines feature similarity, perceptual loss, and diagnostic consistency to ensure preservation of critical pathological information during anonymization.

___Note: It needs at least 24GB VRAM.___


```bash

#!/bin/bash

# Set environment variables for the training
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export CXR_DATA_ROOT="path/to/chest_xray_dataset"
export METADATA_PATH="./configs/cxr_prompt_SD_inpainting.json"
export OUTPUT_DIR="cxr-anofair-model"

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
  --checkpointing_steps=5000 \
```

### **Important Args**

#### **General**

- `--pretrained_model_name_or_path` what model to train/initialize from
- `--instance_data_dir` path for CXR dataset that you want to train
- `--output_dir` where to save/log to
- `--instance_prompt` prompt template for training
- `--train_text_encoder` fine-tuning `text_encoder` with `unet` can give much better results

#### **Loss Components**

- `--cxr_dp_weight` Weight for the CXR Diagnostic Preservation Loss
- `--feature_weight` Weight for the feature similarity component
- `--perceptual_weight` Weight for the perceptual loss component
- `--diagnostic_weight` Weight for the diagnostic consistency component


