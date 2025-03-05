### 1. Train for the Inpainting Model

#### **Training with CXR Diagnostic Preservation Loss**

Our **CXR Diagnostic Preservation Loss** combines feature similarity, perceptual loss, and diagnostic consistency to ensure preservation of critical pathological information during anonymization.

___Note: It needs at least 24GB VRAM.___


```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export INSTANCE_DIR="path-to-cxr-dataset"
export OUTPUT_DIR="path-to-save-model"

accelerate launch ./train_cxr_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --cxr_dp_weight=1.0 \
  --feature_weight=0.3 \
  --perceptual_weight=0.3 \
  --diagnostic_weight=0.4 \
  --instance_prompt="a chest x-ray showing [disease]" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50000
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


