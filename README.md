# CXR-AnoFAIR
CXR-AnoFAIR: Mitigating Attribute Bias in Chest Radiograph Anonymization using Stable Diffusion

This repository contains the implementation of the following paper:


<br>

## :open_book: Overview
<!-- ![overall_structure](./assets/Fig1.png) -->
<img src="./assets/fig1.svg" width="100%">

**CXR-AnoFAIR** presents example results of CXR anonymization with attribute bias mitigation.

<!-- ![result_examples](./assets/Fig2.png) -->
<img src="./assets/fig2.svg" width="100%">

We propose **CXR-AnoFAIR**, a novel framework for chest radiograph anonymization that addresses both privacy and fairness concerns. The pipeline begins with disease region detection, then applies a Stable Diffusion inpainting model guided by our multi-component CXR-Fair loss function. This loss combines **_ℒDA_**(distributional alignment loss), **_ℒSP_**(semantic preservation loss), and our proposed **_ℒCXR-DP_**(CXR diagnostic preservation loss), working alongside a LoRA adapter to efficiently control demographic attributes while preserving diagnostic information.

<br>

## :heavy_check_mark: Updates
- [02/2025] [Codebase](https://github.com/i3a7h3/CXR-AnoFAIR) for CXR-AnoFAIR released.
- [02/2025] [Training code](https://github.com/i3a7h3/CXR-AnoFAIR) for Stable Diffusion Inpainting released.
- [02/2025] [Training code](https://github.com/i3a7h3/CXR-AnoFAIR) for bias mitigation released.

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

## 🔍 Overview of Pipeline
Our framework consists of two main components:

(1) Stable Diffusion Inpainting: Fine-tuned to preserve diagnostic information in chest radiographs 

(2) Bias Mitigation: Using CXR-Fair Loss to ensure demographic fairness

For the initial disease region detection, we use YOLOv8 to identify pathological areas that should be preserved during the anonymization process. This step helps maintain the diagnostic utility of the radiographs while allowing the model to focus on anonymizing non-pathological regions.
<br>
<br>


## :running_man: Train

| Experiment Name | Description |
|---|---|
| [(1) stable-diffusion-inpainting](stable-diffusion-inpainting/) | Fine-tune text encoder and U-Net to preserve diagnostic information in chest radiographs. |
| [(2-1) gender-debias-mitigation](gender-debias-mitigation/) | Finetune prompt LoRA on text encoder U-Net to jointly debias binary gender, to a perfectly balanced distribtion. |
| [(2-2) age-debias-mitigation](age-debias-mitigation/) | Finetune LoRA on text encoder U-Net to jointly debias binary age, to a perfectly balanced distribtion. |
| [(2-3) gender-age-debias-mitigation](gender-age-debias-mitigation/) | Finetune LoRA on text encoder U-Net to jointly debias binary gender and age, to a perfectly balanced distribtion. |


<br>
<br>

## 📊 Evaluation

### Bias Mitigation
Evaluate the effectiveness of demographic bias mitigation, we provide a script that evaluaties and analyzes the distribution of gender and age attributes:
```bash
python ./evaluate_bias_mitiagation.py \
    --gender_classifier_path /path/to/gender_classifier.pt \
    --age_classifier_path /path/to/age_classifier.pt \
    --images_dir /path/to/images \
    --save_dir ./results \
    --target_gender_ratio 0.5 \
    --target_age_ratio 0.5
```

### Re-identification Rate

Evaluate privacy protection using the Siamese Neural Network method from PriCheXy-Net: [PriCheXy-Net repository](https://github.com/kaipackhaeuser/PriCheXy-Net).

### Attribute Classification

Attribute classification using the framework from AttrNzr: [AttrNzr repository](https://github.com/A-Big-Brain/Attribute-Neutralizer-for-medical-AI-system/tree/Fairness).

### Diagnostic Preservation

Evaluate clinical utility using TorchXRayVision: [TorchXRayVision repository](https://github.com/mlmed/torchxrayvision).

<br>
<br>

## ✈️ Inference

To inference, Checkout - `./inference.py` for mode details.


For your dataset, change the path of the CXR-image and CXR-mask.

<br>

## ❤️ Acknowledgements

We thank the authors for their great work:
- [Fair Diffusion](https://github.com/sail-sg/finetune-fair-diffusion) for the distributional alignment techniques
- [AttrNzr](https://github.com/A-Big-Brain/Attribute-Neutralizer-for-medical-AI-system/tree/Fairness) for attribute bias mitigation methods
- [PriCheXy-Net](https://github.com/kaipackhaeuser/PriCheXy-Net) for pioneering CXR anonymization techniques
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) for CXR disease classification
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) for the Stable Diffusion implementation
