### 1. Train for the Inpainting Model

#### **Training with Full Fine-tuning Approach**

Our approach uses full fine-tuning of both U-Net and Text Encoder to adapt the Stable Diffusion Inpainting model for chest radiograph anonymization, preserving diagnostic features while removing identifying information.

___Note: It needs at least 24GB VRAM.___

```bash
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
```

### **Important Args**

#### **General**

- `--pretrained_model_name_or_path`: Base model to fine-tune from (Stable Diffusion Inpainting)
- `--data_root`: Path to the chest X-ray dataset directory
- `--metadata_path`: Path to the metadata JSON file containing disease, severity, and location information
- `--output_dir`: Directory to save the trained model and logs
- `--resolution`: Image resolution for training
- `--train_batch_size`: Batch size per GPU
- `--learning_rate`: Learning rate for training
- `--max_train_steps`: Total number of training steps

#### **Advanced Options**

- `--gradient_checkpointing`: Enable gradient checkpointing to reduce memory usage
- `--use_8bit_adam`: Use 8-bit precision for Adam optimizer to reduce memory consumption
- `--mixed_precision="fp16"`: Use mixed precision training to improve performance
- `--center_crop`: Enable center cropping of input images
- `--seed`: Set random seed for reproducibility



