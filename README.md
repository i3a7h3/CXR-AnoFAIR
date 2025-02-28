# CXR-AnoFAIR
CXR-AnoFAIR: Mitigating Attribute Bias in Chest Radiograph Anonymization using Stable Diffusion

This repository contains the implementation of the following paper:


<br>

## :open_book: Overview
<!-- ![overall_structure](./assets/Fig1.png) -->
<img src="./assets/Fig1.png" width="100%">

**CXR-AnoFAIR** presents example results of CXR anonymization with attribute bias mitigation.

<!-- ![result_examples](./assets/Fig2.png) -->
<img src="./assets/Fig2.png" width="100%">

We propose **CXR-AnoFAIR**, a novel framework for chest radiograph anonymization that addresses both privacy and fairness concerns. The pipeline begins with disease region detection, then applies a Stable Diffusion inpainting model guided by our multi-component CXR-Fair loss function. This loss combines **_ℒDA_**, **_ℒSP_**, and our proposed **_ℒCXR-DP_**, working alongside a LoRA adapter to efficiently control demographic attributes while preserving diagnostic information.

<br>

## :heavy_check_mark: Updates
- [02/2025] [Codebase](https://github.com/i3a7h3/CXR-AnoFAIR) for CXR-AnoFAIR released.
- [02/2025] [Training code](https://github.com/i3a7h3/CXR-AnoFAIR) for Stable Diffusion Inpainting released.
- [02/2025] [Training code](https://github.com/i3a7h3/CXR-AnoFAIR) for bias mitigation released.
- [02/2025] [Evaluation code](https://github.com/i3a7h3/CXR-AnoFAIR) for CXR-AnoFAIR released.

## :hammer: Setup

### Environment

```bash
conda create -n CXR-AnoFAIR python=3.10.13
conda activate CXR-AnoFAIR

git clone https://github.com/i3a7h3/CXR-AnoFAIR.git
pip install peft
pip install diffusers
pip install -r requirements.txt
```

<br>

## ⬇️: Disease Region Detection

### 1. Pretrained Model
1. Download the pretrained YOLOv8 model from [Google Drive](https://drive.google.com/file/d/1-0K1gMrjq0C7ssP4UDGePBr8CND4cAJC/view).
2. Download the annotations from [Google Drive](https://drive.google.com/file/d/1zAEiTPFPky7ilQ1Ou-Kzi9sPVdvx2nnt/view)

3. Put the model under `pretrained` folder as follows:
    ```
    CXR-AnoFAIR
    └── detection
        └── yolov8
            └── model_final.pt
        └── dataset
            └── annotations.coco
    ```

### 2. Disease Detection

1. We use YOLOv8 for chest disease detection. The model accurately localizes abnormal regions in chest radiographs, including various pathologies such as cardiomegaly, pneumothorax, pleural effusion, and lung opacities.

2. Prepare your chest X-ray images for input:
Place your images in the input folder.

```bash
python ./detection/test_disease_detector.py
```
Find the output masks in `./detection/output`

<br>

## :running_man: Train

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


### 2. Train for Bias Mitigation

Run the script below for training with CXR-Fair Loss for bias mitigation. 

```bash
export BASE_MODEL="path-to-inpainted-model"
export TRAIN_DATA="path-to-training-data"
export OUTPUT_DIR="path-to-bias-mitigated-model"

accelerate launch ./train_bias_mitigation.py \
  --pretrained_model_name_or_path=$BASE_MODEL \
  --train_data_dir=$TRAIN_DATA \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-5 \
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
  --seed="0"
```

### Important Args

#### **General**

- `--pretrained_model_name_or_path` what model to train/initialize from
- `--output_dir` where to save/log to
- `--seed` training seed (not set by default)

#### **CXR-Fair Loss Components**

- `--distributional_alignment_weight` Weight for the ℒ<sub>DA</sub> component
- `--semantic_preservation_weight` Weight for the ℒ<sub>SP</sub> component
- `--diagnostic_preservation_weight` Weight for the ℒ<sub>CXR-DP</sub> component
- `--feature_similarity_weight` Weight for feature similarity
- `--perceptual_weight` Weight for perceptual loss
- `--diagnostic_consistency_weight` Weight for diagnostic consistency

#### **Optimizers/learning rates**

- `--max_train_steps` How many train steps to take
- `--gradient_accumulation_steps` Gradient accumulation for larger batch size
- `--train_batch_size` Batch size per GPU
- `--checkpointing_steps` How often to save model
- `--gradient_checkpointing` For memory optimization
- `--learning_rate` Learning rate
- `--lora_rank` LoRA adapter rank

<br>

## ✈️ Inference

To run inference with our model, check out `inference.ipynb` for more details.

For your dataset, change the path of the input CXR image and disease mask.

<br>

## 📊 Evaluation

### Re-identification Rate

Evaluate privacy protection using the Siamese Neural Network method from PriCheXy-Net:
```bash
python ./evaluation/eval_reidentification.py
```

For more details, see the [PriCheXy-Net repository](https://github.com/kaipackhaeuser/PriCheXy-Net).

### Attribute Bias Evaluation

Quantify demographic bias reduction using the framework from AttrNzr:
```bash
python ./evaluation/eval_attribute_bias.py
```

For more details, see the [AttrNzr repository](https://github.com/A-Big-Brain/Attribute-Neutralizer-for-medical-AI-system/tree/Fairness).

### Diagnostic Preservation

Evaluate clinical utility using TorchXRayVision:
```bash
python ./evaluation/eval_diagnostic_preservation.py
```

For more details, see the [TorchXRayVision repository](https://github.com/mlmed/torchxrayvision).

<br>

## 📊 Results

Our framework significantly outperforms existing methods in terms of:
- **Anonymization effectiveness**: 27.8% reduction in re-identification rate compared to PriCheXy-Net
- **Bias mitigation**: Significant reduction in gender and age biases while maintaining high diagnostic fidelity
- **Image quality**: Superior generation quality with FID scores of 5.9, compared to 45.7 for the base model

<br>

## ❤️ Acknowledgements

We thank the authors for their great work:
- [PriCheXy-Net](https://github.com/kaipackhaeuser/PriCheXy-Net) for pioneering CXR anonymization techniques
- [AttrNzr](https://github.com/A-Big-Brain/Attribute-Neutralizer-for-medical-AI-system/tree/Fairness) for attribute bias mitigation methods
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) for CXR disease classification
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) for the Stable Diffusion implementation
- The authors of the [LoRA](https://github.com/microsoft/LoRA) method for efficient adaptation of pretrained models
