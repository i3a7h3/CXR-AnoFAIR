#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import json
import math
import yaml
import random
import logging
import argparse
import itertools
from datetime import datetime
from pathlib import Path
from packaging import version
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw

# Huggingface imports
import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, GradScalerKwargs

import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# PEFT for LoRA
from peft import LoraConfig, get_peft_model_state_dict

# Import torchxrayvision for the disease classifier
import torchxrayvision as xrv

# Set up logger
logger = get_logger(__name__)

class AgeClassifier(nn.Module):
    """Age classifier for CXR images (young vs old - threshold at 60 years)"""
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # Initialize model
        self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(2048, 2)  # 2 outputs for age (young < 60, old ≥ 60)
        
        # Load pretrained weights
        if pretrained_path is not None and os.path.exists(pretrained_path):
            self.model.load_state_dict(torch.load(pretrained_path))
        
        # Set to evaluation mode
        self.model.eval()
        
    def forward(self, x):
        return self.model(x)
    
    # Add this for consistency with the gender-age version
    def forward_age(self, x):
        return self.forward(x)

class PromptsDataset(torch.utils.data.Dataset):
    """Simple dataset for prompts"""
    def __init__(self, prompts):
        self.prompts = prompts
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, i):
        return self.prompts[i]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="CXR-AnoFAIR: Mitigating Age Bias in Chest Radiograph Anonymization")

    # Experiment setting
    parser.add_argument(
        '--proj_name', 
        default="CXR-AnoFAIR-Age",
        help="Project name",
        type=str, 
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        default=True,
        help="Whether to train the text encoder with LoRA",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        default=False,
        help="Whether to train the UNet with LoRA",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default="0", 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X steps for resuming training if needed",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--checkpointing_steps_long",
        type=int,
        default=1000,
        help="Save a permanent checkpoint every Y steps for evaluation",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to resume from checkpoint",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        default=True,
        help="Enable xformers for memory efficient attention"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="The dimension of the LoRA update matrices",
    )
    parser.add_argument(
        '--train_plot_every_n_iter', 
        help="Plot training stats every n iterations", 
        type=int, 
        default=20
    )
    parser.add_argument(
        '--evaluate_every_n_iter', 
        help="Evaluate model every n iterations", 
        type=int,
        default=200
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Where to report training results (currently only wandb supported)",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help="Allow TF32 precision for faster training on Ampere GPUs",
    )
    parser.add_argument(
        '--guidance_scale', 
        help="Diffusion model text guidance scale", 
        type=float, 
        default=7.5
    )
    parser.add_argument(
        "--target_age_ratio",
        type=float,
        default=0.75,
        help="Target ratio for young age (0.0 = all old, 0.5 = balanced, 1.0 = all young)",
    )

    # Loss weights
    parser.add_argument(
        '--distributional_alignment_weight', 
        default=1.0,
        help="Weight for the distributional alignment loss", 
        type=float, 
    )
    parser.add_argument(
        '--semantic_preservation_weight', 
        default=1.0,
        help="Weight for the semantic preservation loss", 
        type=float, 
    )
    parser.add_argument(
        '--diagnostic_preservation_weight', 
        default=1.0,
        help="Weight for the CXR diagnostic preservation loss", 
        type=float, 
    )
    # Sub-components of diagnostic preservation
    parser.add_argument(
        '--feature_similarity_weight', 
        default=0.3,
        help="Weight for feature similarity component", 
        type=float, 
    )
    parser.add_argument(
        '--perceptual_weight', 
        default=0.3,
        help="Weight for perceptual component", 
        type=float, 
    )
    parser.add_argument(
        '--diagnostic_consistency_weight', 
        default=0.4,
        help="Weight for diagnostic consistency component", 
        type=float, 
    )
    parser.add_argument(
        '--uncertainty_threshold', 
        help="Uncertainty threshold for distributional alignment loss", 
        type=float, 
        default=0.2
    )
    parser.add_argument(
        '--dynamic_weight_factor', 
        help="Factor for dynamic weighting", 
        type=float, 
        default=0.5
    )

    # Batch sizes
    parser.add_argument(
        '--train_images_per_prompt_GPU', 
        help="Number of images generated per prompt per GPU during training", 
        type=int, 
        default=8,
    )
    parser.add_argument(
        '--train_batch_size', 
        help="Training batch size per GPU", 
        type=int, 
        default=4
    )
    parser.add_argument(
        '--val_images_per_prompt_GPU', 
        help="Number of images generated per prompt per GPU during validation", 
        type=int, 
        default=8
    )
    parser.add_argument(
        '--val_batch_size', 
        help="Validation batch size per GPU", 
        type=int, 
        default=8
    )    

    # Data and output paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Logging directory",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Directory containing training data (not used in current implementation)",
        required=False
    )
    parser.add_argument(
        "--prompt_templates_path",
        type=str,
        default="./prompt_templates.json",
        help="Path to prompt templates JSON file",
    )
    parser.add_argument(
        '--age_classifier_path', 
        default=None,
        help="Path to pre-trained age classifier", 
        type=str,
        required=True, 
    )

    # Learning rate settings
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="LR scheduler type",
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of warmup steps for the LR scheduler"
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets in cosine_with_restarts scheduler",
    )
    parser.add_argument(
        "--lr_power", 
        type=float, 
        default=1.0, 
        help="Power factor for polynomial scheduler"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument('--use_8bit_adam', action='store_true', help='Use 8-bit Adam optimizer')

    # Advanced settings
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward pass",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing to save memory",
    )
    
    # Image size settings
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution for training",
    )
    parser.add_argument(
        "--img_size_small",
        type=int,
        default=224,
        help="Size to resize images for efficient processing",
    )

    # Config file
    parser.add_argument("--config", help="Config file path", type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Load config from file if specified
    if args.config:
        with open(args.config, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        args_dict = vars(args)
        for key, value in config_data.items():
            if key in args_dict:
                args_dict[key] = type(args_dict[key])(value)
        args = argparse.Namespace(**args_dict)

    return args

def make_grad_hook(coef):
    """Create a gradient hook that scales gradients by coef"""
    return lambda x: coef * x

def customized_all_gather(tensor, accelerator, return_tensor_other_processes=False):
    """Gather tensors from all processes and optionally return tensors from other processes"""
    tensor_all = [tensor.detach().clone() for i in range(accelerator.num_processes)]
    torch.distributed.all_gather(tensor_all, tensor)
    if return_tensor_other_processes:
        if accelerator.num_processes > 1:
            tensor_others = torch.cat([tensor_all[idx] for idx in range(accelerator.num_processes) 
                                       if idx != accelerator.local_process_index], dim=0)
        else:
            tensor_others = torch.empty([0,] + list(tensor_all[0].shape[1:]), 
                                        device=accelerator.device, dtype=tensor_all[0].dtype)
    tensor_all = torch.cat(tensor_all, dim=0)
    
    if return_tensor_other_processes:
        return tensor_all, tensor_others
    else:
        return tensor_all

def clean_checkpoint(ckpts_save_dir, name, checkpoints_total_limit):
    """Clean up old checkpoints to stay under checkpoints_total_limit"""
    checkpoints = os.listdir(ckpts_save_dir)
    checkpoints = [d for d in checkpoints if d.startswith(name)]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # Before saving the new checkpoint, we need to have at most checkpoints_total_limit - 1 checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(
            f"checkpoint name:{name}, {len(checkpoints)} checkpoints already exist, "
            f"removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(ckpts_save_dir, removing_checkpoint)
            if os.path.isdir(removing_checkpoint):
                try:
                    import shutil
                    shutil.rmtree(removing_checkpoint)
                except:
                    logger.warning(f"Failed to remove checkpoint: {removing_checkpoint}")

def get_clip_feat(images, clip_vision_model_w_proj, clip_img_mean, clip_img_std, normalize=True, to_high_precision=True):
    """Extract CLIP features from images
    
    Args:
        images: torch.Tensor shape [N,3,H,W], in range [-1,1]
        normalize: Whether to normalize features
        to_high_precision: Whether to convert to high precision
    
    Returns:
        embeds: CLIP embeddings
    """
    images_preprocessed = ((images + 1) * 0.5 - clip_img_mean) / clip_img_std
    embeds = clip_vision_model_w_proj(images_preprocessed).image_embeds
    
    if to_high_precision:
        embeds = embeds.to(torch.float32)
    if normalize:
        embeds = torch.nn.functional.normalize(embeds, dim=-1)
    return embeds

def get_dino_feat(images, dinov2_model, dinov2_img_mean, dinov2_img_std, normalize=True, to_high_precision=True):
    """Extract DINO features from images
    
    Args:
        images: torch.Tensor shape [N,3,H,W], in range [-1,1]
        normalize: Whether to normalize features
        to_high_precision: Whether to convert to high precision
    
    Returns:
        embeds: DINO embeddings
    """
    images_preprocessed = ((images + 1) * 0.5 - dinov2_img_mean) / dinov2_img_std
    embeds = dinov2_model(images_preprocessed)
    
    if to_high_precision:
        embeds = embeds.to(torch.float32)
    if normalize:
        embeds = torch.nn.functional.normalize(embeds, dim=-1)
    return embeds

@torch.no_grad()
def generate_dynamic_targets(probs, target_ratio, w_uncertainty=False):
    """Generate dynamic targets for the distributional alignment loss
    
    Args:
        probs: torch.Tensor [N,2], probabilities for binary attributes
                (young, old) where young is age < 60 and old is age >= 60
        target_ratio: Target distribution (percentage of young class - class 0)
        w_uncertainty: Whether to return uncertainty measures
    
    Returns:
        targets_all: Target classes
        uncertainty_all: Uncertainty of target classes (if w_uncertainty=True)
    """
    import scipy.stats
    
    idxs_2_rank = (probs != -1).all(dim=-1)
    probs_2_rank = probs[idxs_2_rank]

    # Rank by probability of being young (class 0)
    rank = torch.argsort(torch.argsort(probs_2_rank[:, 0]))
    # Assign target class based on target distribution - top target_ratio% will be young
    targets = (rank >= (rank.shape[0] * (1 - target_ratio))).long()

    targets_all = torch.ones([probs.shape[0]], dtype=torch.long, device=probs.device) * (-1)
    targets_all[idxs_2_rank] = targets
    
    if w_uncertainty:
        uncertainty = torch.ones([probs_2_rank.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
        uncertainty[targets == 0] = torch.tensor(
            1 - scipy.stats.binom.cdf(
                (rank[targets == 0]).cpu().numpy(), 
                probs_2_rank.shape[0], 
                1 - target_ratio
            )
        ).to(probs.dtype).to(probs.device)
        uncertainty[targets == 1] = torch.tensor(
            scipy.stats.binom.cdf(
                rank[targets == 1].cpu().numpy(), 
                probs_2_rank.shape[0], 
                target_ratio
            )
        ).to(probs.dtype).to(probs.device)
        
        uncertainty_all = torch.ones([probs.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
        uncertainty_all[idxs_2_rank] = uncertainty
        
        return targets_all, uncertainty_all
    else:
        return targets_all

def gen_dynamic_weights(age_indicators, targets, preds_age_ori, probs_age_ori, factor=0.5):
    """Generate dynamic weights for the loss based on age predictions"""
    weights = []
    for age_indicator, target, pred_age_ori, prob_age_ori in zip(
            age_indicators, targets, preds_age_ori, probs_age_ori):
        if not age_indicator:
            weights.append(factor)  # Lower weight for images without detected age
        else:
            if target == -1:
                weights.append(factor)
            elif target == pred_age_ori:
                weights.append(1.0)  # Full weight for correctly aligned age
            else:
                weights.append(factor)  # Lower weight for misaligned age

    weights = torch.tensor(weights, dtype=probs_age_ori.dtype, device=probs_age_ori.device)
    return weights

def plot_in_grid(images, save_to, age_indicators=None, age_preds=None, pred_class_probs=None):
    """Plot images in a grid for visualization
    
    Args:
        images: torch.Tensor [N,3,H,W]
        save_to: Path to save the visualization
        age_indicators: Boolean tensor indicating if age was detected
        age_preds: Predicted age
        pred_class_probs: Prediction probabilities
    """
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    # Create a figure
    batch_size = images.shape[0]
    rows = int(math.sqrt(batch_size))
    cols = math.ceil(batch_size / rows)
    
    # Convert images from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    
    # Create a grid
    grid = make_grid(images, nrow=cols)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    
    # Add information if provided
    if age_indicators is not None and age_preds is not None and pred_class_probs is not None:
        for i in range(batch_size):
            row = i // cols
            col = i % cols
            x = col * (images.shape[3] + 2) + images.shape[3] // 2
            y = row * (images.shape[2] + 2) + images.shape[2] // 2
            
            if age_indicators[i]:
                age_text = f"Age: {'Y' if age_preds[i]==0 else 'O'} ({pred_class_probs[i]:.2f})"
                plt.text(x, y, age_text, color='white', 
                         horizontalalignment='center', verticalalignment='center')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    plt.savefig(save_to)
    plt.close()

def model_sanity_print(model, state):
    """Print model parameters and gradients for debugging"""
    params = [p for p in model.parameters()]
    if len(params) > 0 and params[0].grad is not None:
        print(f"\t{params[0].device}; {state};\n\t\tparam[0]: {params[0].flatten()[0].item():.8f};"
              f"\tparam[0].grad: {params[0].grad.flatten()[0].item():.8f}")
    else:
        print(f"\t{params[0].device if len(params) > 0 else 'N/A'}; {state}; No gradients yet")

@torch.no_grad()
def generate_image_no_gradient(prompt, tokenizer, text_encoder, unet, vae, noises, num_denoising_steps, 
                              guidance_scale, device, weight_dtype):
    """Generate images without tracking gradients
    
    Args:
        prompt: Text prompt
        noises: Starting noise
        num_denoising_steps: Number of denoising steps
        which_text_encoder: Text encoder to use
        which_unet: UNet to use
    """
    # Create scheduler
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )
    
    N = noises.shape[0]
    prompts = [prompt] * N
    
    # Encode prompts
    prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
    prompts_token["input_ids"] = prompts_token["input_ids"].to(device)
    prompts_token["attention_mask"] = prompts_token["attention_mask"].to(device)

    prompt_embeds = text_encoder(
        prompts_token["input_ids"],
        prompts_token["attention_mask"],
    )
    prompt_embeds = prompt_embeds[0]

    # Encode negative prompts
    batch_size = prompt_embeds.shape[0]
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
    uncond_input["input_ids"] = uncond_input["input_ids"].to(device)
    uncond_input["attention_mask"] = uncond_input["attention_mask"].to(device)
    negative_prompt_embeds = text_encoder(
        uncond_input["input_ids"],
        uncond_input["attention_mask"],
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    # Combine positive and negative prompts
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds = prompt_embeds.to(weight_dtype)
    
    # Set up denoising
    noise_scheduler.set_timesteps(num_denoising_steps)
    latents = noises
    
    # Denoise
    for i, t in enumerate(noise_scheduler.timesteps):
        # Scale model input
        latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noises_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample
        noises_pred = noises_pred.to(torch.float32)
        
        # Perform guidance
        noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
        noises_pred = noises_pred_uncond + guidance_scale * (noises_pred_text - noises_pred_uncond)
        
        # Update latents
        latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

    # Decode latents
    latents = 1 / vae.config.scaling_factor * latents
    images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)  # in range [-1,1]
    
    return images

def generate_image_w_gradient(prompt, tokenizer, text_encoder, unet, vae, noises, num_denoising_steps, 
                             guidance_scale, device, weight_dtype):
    """Generate images with gradient tracking for training
    
    Similar to generate_image_no_gradient but allows gradients to flow through
    """
    # Create scheduler
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
    )
    
    # Set UNet to train mode to enable gradient checkpointing
    unet.train()
    
    N = noises.shape[0]
    prompts = [prompt] * N
    
    # Encode prompts
    prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
    prompts_token["input_ids"] = prompts_token["input_ids"].to(device)
    prompts_token["attention_mask"] = prompts_token["attention_mask"].to(device)

    prompt_embeds = text_encoder(
        prompts_token["input_ids"],
        prompts_token["attention_mask"],
    )
    prompt_embeds = prompt_embeds[0]

    # Encode negative prompts
    batch_size = prompt_embeds.shape[0]
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
    uncond_input["input_ids"] = uncond_input["input_ids"].to(device)
    uncond_input["attention_mask"] = uncond_input["attention_mask"].to(device)
    negative_prompt_embeds = text_encoder(
        uncond_input["input_ids"],
        uncond_input["attention_mask"],
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    # Combine positive and negative prompts
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]).to(weight_dtype)
    
    # Set up denoising
    noise_scheduler.set_timesteps(num_denoising_steps)
    
    # Calculate gradient coefficients for each step
    grad_coefs = []
    for i, t in enumerate(noise_scheduler.timesteps):
        alpha_prod_t = noise_scheduler.alphas_cumprod[t].sqrt()
        alpha_prod_t_prev = (1 - noise_scheduler.alphas_cumprod[t]).sqrt()
        grad_coef = alpha_prod_t * alpha_prod_t_prev / (1 - noise_scheduler.alphas[t])
        grad_coefs.append(grad_coef.item())
    
    # Normalize gradient coefficients
    grad_coefs = np.array(grad_coefs)
    grad_coefs /= (math.prod(grad_coefs) ** (1/len(grad_coefs)))
    
    # Start with noise
    latents = noises
    
    # Denoise
    for i, t in enumerate(noise_scheduler.timesteps):
        # Scale model input
        latent_model_input = torch.cat([latents.detach().to(weight_dtype)] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noises_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample
        noises_pred = noises_pred.to(torch.float32)
        
        # Perform guidance
        noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
        noises_pred = noises_pred_uncond + guidance_scale * (noises_pred_text - noises_pred_uncond)
        
        # Register gradient hook
        hook_fn = make_grad_hook(grad_coefs[i])
        noises_pred.register_hook(hook_fn)
        
        # Update latents
        latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

    # Decode latents
    latents = 1 / vae.config.scaling_factor * latents
    images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)  # in range [-1,1]
    
    return images

@torch.no_grad()
def evaluate_process(args, tokenizer, text_encoder, unet, vae, age_classifier, 
                    accelerator, weight_dtype, epoch, global_step):
    """Evaluate the model during training"""
    logger.info("Running validation...")
    
    # Load validation prompts
    with open(args.prompt_templates_path) as f:
        prompt_data = json.load(f)
    
    validation_prompts = []
    for template in prompt_data["prompt_templates_test"]:
        for disease in prompt_data.get("validation_diseases", ["No Finding", "Pneumonia", "Effusion"]):
            validation_prompts.append(template.format(disease=disease))
    
    # Create fixed noise for reproducibility
    torch.manual_seed(args.seed)
    val_noises = torch.randn(
        [len(validation_prompts), args.val_images_per_prompt_GPU, 4, 64, 64],
        dtype=torch.float32
    ).to(accelerator.device)
    
    # Initialize logs and images
    logs = []
    log_imgs = []
    
    # Generate and evaluate images for each prompt
    for prompt_idx, (prompt, noise) in enumerate(zip(validation_prompts, val_noises)):
        if accelerator.is_main_process:
            print(f"Evaluating prompt: {prompt}")
            
            logs_i = {
                "age_bias": [],
                "age_balanced": [],
            }
            log_imgs_i = {}
        
        # Generate images
        images = []
        for batch_idx in range(0, noise.shape[0], args.val_batch_size):
            batch_noise = noise[batch_idx:batch_idx + args.val_batch_size]
            batch_images = generate_image_no_gradient(
                prompt, tokenizer, text_encoder, unet, vae,
                batch_noise, 25, args.guidance_scale, 
                accelerator.device, weight_dtype
            )
            images.append(batch_images)
        
        images = torch.cat(images)
        
        # Evaluate age
        resized_images = F.interpolate(images, size=(224, 224), mode='bilinear')
        normalized_images = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )((resized_images + 1) / 2)
        
        age_logits = age_classifier(normalized_images)
        age_probs = F.softmax(age_logits, dim=-1)
        
        # Calculate age bias - difference between young and old predictions
        age_preds = age_logits.argmax(dim=-1)
        age_prob_diffs = age_probs[:, 0] - age_probs[:, 1]  # young - old
        
        # Gather results from all processes
        images_all = customized_all_gather(images, accelerator)
        age_preds_all = customized_all_gather(age_preds, accelerator)
        age_prob_diffs_all = customized_all_gather(age_prob_diffs, accelerator)
        age_probs_all = customized_all_gather(age_probs, accelerator)
        
        if accelerator.is_main_process:
            # Save visualization
            save_to = os.path.join(args.imgs_save_dir, f"eval_step{global_step}_prompt{prompt_idx}.jpg")
            plot_in_grid(
                images_all, save_to, 
                torch.ones_like(age_preds_all, dtype=torch.bool),
                age_preds_all, 
                age_probs_all[:, 0]  # Young probability
            )
            
            log_imgs_i["generated_images"] = [save_to]
            
            # Calculate bias metrics
            age_bias = age_prob_diffs_all.abs().mean().item()
            
            # Calculate balance metrics (closer to target_age_ratio is better)
            # For age, we want a specific distribution (e.g., 75% young, 25% old)
            young_ratio = age_preds_all.float().mean().item()
            age_balance = abs(young_ratio - args.target_age_ratio)
            
            logs_i["age_bias"].append(age_bias)
            logs_i["age_balanced"].append(1 - age_balance)  # Higher is better
            
            logs.append(logs_i)
            log_imgs.append(log_imgs_i)
    
    # Log results with wandb
    if accelerator.is_main_process and is_wandb_available():
        import wandb
        
        # Log metrics
        for prompt_idx, (prompt, logs_i) in enumerate(zip(validation_prompts, logs)):
            for key, values in logs_i.items():
                if isinstance(values, list):
                    wandb.log({f"eval_{key}_{prompt_idx}": np.mean(values)}, step=global_step)
                else:
                    wandb.log({f"eval_{key}_{prompt_idx}": values}, step=global_step)
        
        # Log average metrics across all prompts
        avg_metrics = {}
        for key in logs[0].keys():
            values = []
            for log_i in logs:
                if isinstance(log_i[key], list):
                    values.extend(log_i[key])
                else:
                    values.append(log_i[key])
            avg_metrics[f"eval_avg_{key}"] = np.mean(values)
        
        wandb.log(avg_metrics, step=global_step)
        
        # Log images
        for prompt_idx, (prompt, log_imgs_i) in enumerate(zip(validation_prompts, log_imgs)):
            for key, values in log_imgs_i.items():
                wandb.log({
                    f"eval_{key}_{prompt_idx}": [
                        wandb.Image(values[0], caption=prompt)
                    ]
                }, step=global_step)
    
    return logs

def main():
    args = parse_args()
    
    if not args.train_text_encoder and not args.train_unet:
        raise ValueError("At least one of --train_text_encoder and --train_unet must be True.")
    
    # Setup logging directory
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    # Configure gradient scaler for mixed precision
    kwargs = GradScalerKwargs(
        init_scale=2.**0,
        growth_interval=99999999, 
        backoff_factor=0.5,
        growth_factor=2,
    )
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set appropriate logging verbosity
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Set seed for reproducibility
    set_seed(args.seed, device_specific=True)
    
    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize folders and trackers
    now = datetime.now()
    timestring = f"{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}"
    folder_name = (f"CXR-AnoFAIR-Age_BS-{args.train_images_per_prompt_GPU*accelerator.num_processes}_"
                   f"DA-{args.distributional_alignment_weight}_SP-{args.semantic_preservation_weight}_"
                   f"DP-{args.diagnostic_preservation_weight}_loraR-{args.lora_rank}_{timestring}")
    
    args.imgs_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "imgs")
    args.ckpts_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "ckpts")
    
    if accelerator.is_main_process:
        os.makedirs(args.imgs_save_dir, exist_ok=True)
        os.makedirs(args.ckpts_save_dir, exist_ok=True)
        accelerator.init_trackers(
            args.proj_name, 
            init_kwargs = {
                "wandb": {
                    "name": folder_name, 
                    "dir": args.output_dir
                }
            }
        )
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet",
    )
    
    # Freeze models - we'll only train LoRA adapters
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()
    
    # Set up precision
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move models to device
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Create copies for evaluation
    if args.train_text_encoder:
        eval_text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
        )
        eval_text_encoder.requires_grad_(False)
        eval_text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    if args.train_unet:        
        eval_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="unet",
        )
        eval_unet.requires_grad_(False)
        eval_unet.to(accelerator.device, dtype=weight_dtype)
    
    # Enable xformers if requested
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xformers to at least 0.0.17. "
                    "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
            
            if args.train_unet:
                eval_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # Set up LoRA for UNet if requested
    if args.train_unet:
        # Apply LoRA to UNet using PEFT
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)
        
        # Only train the LoRA params
        unet.requires_grad_(False)
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
    
    # Set up LoRA for text encoder if requested
    if args.train_text_encoder:
        from peft import get_peft_model
        
        # Apply LoRA to Text Encoder
        text_encoder_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
        
        # Only train the LoRA params
        text_encoder.requires_grad_(False)
        for param in text_encoder.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
    
    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Set up parameters to optimize
    params_to_optimize = []
    if args.train_text_encoder:
        params_to_optimize.extend([p for p in text_encoder.parameters() if p.requires_grad])
    if args.train_unet:
        params_to_optimize.extend([p for p in unet.parameters() if p.requires_grad])
    
    # Set up optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            logger.warn("8-bit Adam not available, using standard Adam")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
        
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Load training prompts
    with open(args.prompt_templates_path) as f:
        prompt_data = json.load(f)
    
    train_prompts = []
    for template in prompt_data["prompt_templates_train"]:
        for disease in prompt_data.get("train_diseases", ["No Finding", "Pneumonia", "Effusion", "Cardiomegaly"]):
            train_prompts.append(template.format(disease=disease))
    
    # Create dataset
    train_dataset = PromptsDataset(prompts=train_prompts)
    args.num_update_steps_per_epoch = len(train_dataset)
    args.num_train_epochs = math.ceil(args.max_train_steps / args.num_update_steps_per_epoch)
    
    # Create dataloader indices - same across all processes
    random.seed(args.seed + 1)
    train_dataloader_idxs = []
    for epoch in range(args.num_train_epochs):
        idxs = list(range(len(train_dataset)))
        random.shuffle(idxs)
        train_dataloader_idxs.append(idxs)
    
    # Load utility models for training
    # 1. Age classifier
    age_classifier = AgeClassifier(args.age_classifier_path)
    age_classifier.to(accelerator.device, dtype=weight_dtype)
    age_classifier.eval()
    
    # 2. Feature extraction models - CLIP and DINO
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    clip_vision_model_w_proj = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    clip_vision_model_w_proj.to(accelerator.device, dtype=weight_dtype)
    clip_vision_model_w_proj.requires_grad_(False)
    
    # Get CLIP image normalization stats
    clip_img_mean = torch.tensor(clip_image_processor.image_mean).reshape([-1, 1, 1]).to(accelerator.device, dtype=weight_dtype)
    clip_img_std = torch.tensor(clip_image_processor.image_std).reshape([-1, 1, 1]).to(accelerator.device, dtype=weight_dtype)
    
    # Load DINO model for semantic preservation
    try:
        import torch.hub
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    except:
        logger.warning("Failed to load DINO model from torch hub. Using a simple CNN instead.")
        dinov2_model = torchvision.models.resnet18(pretrained=True)
        # Remove the classification head
        dinov2_model = nn.Sequential(*list(dinov2_model.children())[:-1])
    
    dinov2_model.to(accelerator.device, dtype=weight_dtype)
    dinov2_model.requires_grad_(False)
    dinov2_model.eval()
    
    # DINO image normalization stats
    dinov2_img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([-1, 1, 1]).to(accelerator.device, dtype=weight_dtype)
    dinov2_img_std = torch.tensor([0.229, 0.224, 0.225]).reshape([-1, 1, 1]).to(accelerator.device, dtype=weight_dtype)
    
    # 3. Disease classifier from torchxrayvision
    disease_classifier = xrv.models.DenseNet(weights="densenet121-res224-all")
    disease_classifier.to(accelerator.device, dtype=weight_dtype)
    disease_classifier.requires_grad_(False)
    disease_classifier.eval()
    
    # Set up loss
    CE_loss = nn.CrossEntropyLoss(reduction="none")
    
    # Set up LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Prepare models with accelerator
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    
    if args.train_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)
        
    if args.train_unet:
        unet = accelerator.prepare(unet)
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])

            resume_global_step = global_step
            first_epoch = global_step // args.num_update_steps_per_epoch
            resume_step = resume_global_step % args.num_update_steps_per_epoch
    else:
        global_step = 0
        first_epoch = 0
        resume_step = 0
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num prompts = {len(train_dataset)}")
    logger.info(f"  Num images per prompt = {args.train_images_per_prompt_GPU} (per GPU) * {accelerator.num_processes} (GPUs)")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Optimization steps = {args.max_train_steps}")
    
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True) if accelerator.is_main_process else None
    
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, data_idx in enumerate(train_dataloader_idxs[epoch]):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                continue
                
            # Run evaluation at the beginning of training
            if global_step == 0:
                evaluate_process(
                    args, tokenizer, text_encoder, unet, vae, age_classifier,
                    accelerator, weight_dtype, epoch, global_step
                )
            
            # Get prompt for this step
            prompt_i = train_dataset[data_idx]
            
            # Generate random noise
            torch.manual_seed(args.seed + global_step)
            noises_i = torch.randn(
                [args.train_images_per_prompt_GPU, 4, 64, 64],
                dtype=torch.float32
            ).to(accelerator.device)

            accelerator.wait_for_everyone()
            optimizer.zero_grad()
            
            # Print noise for debugging
            noises_i_all = [noises_i.detach().clone() for i in range(accelerator.num_processes)]
            torch.distributed.all_gather(noises_i_all, noises_i)
            if accelerator.is_main_process:
                now = datetime.now()
                accelerator.print(
                    f"{now.strftime('%Y/%m/%d - %H:%M:%S')} --- epoch: {epoch}, step: {step}, prompt: {prompt_i}\n" +
                    " ".join([f"\tprocess idx: {idx}; noise: {noises_i_all[idx].flatten()[-1].item():.4f};" 
                              for idx in range(len(noises_i_all))])
                    )
            
            if accelerator.is_main_process:
                logs_i = {
                    "loss_da": [],  # Distributional alignment loss
                    "loss_sp": [],  # Semantic preservation loss
                    "loss_dp": [],  # Diagnostic preservation loss
                    "loss": [],     # Total loss
                    "age_bias": [],
                    "age_balanced": [],
                }
                log_imgs_i = {}
            
            # Choose random number of denoising steps
            num_denoising_steps = random.choices(range(19, 24), k=1)[0]
            torch.distributed.broadcast_object_list([num_denoising_steps], src=0)
            
            with torch.no_grad():
                ################################################
                # step 1: generate all images using the diffusion model being finetuned
                images = []
                for j in range(0, noises_i.shape[0], args.val_batch_size):
                    noises_ij = noises_i[j:j + args.val_batch_size]
                    images_ij = generate_image_no_gradient(
                        prompt_i, tokenizer, text_encoder, unet, vae,
                        noises_ij, num_denoising_steps, args.guidance_scale, 
                        accelerator.device, weight_dtype
                    )
                    images.append(images_ij)
                images = torch.cat(images)

                # Process generated images for age classification
                resized_images = F.interpolate(images, size=(224, 224), mode='bilinear')
                normalized_images = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )((resized_images + 1) / 2)
                
                # Get age predictions
                age_logits = age_classifier(normalized_images)
                age_probs = F.softmax(age_logits, dim=-1)
                age_preds = age_logits.argmax(dim=-1)
                
                # Generate small versions for feature extraction
                images_small = F.interpolate(images, size=(args.img_size_small, args.img_size_small), mode='bilinear')
                clip_feats = get_clip_feat(images_small, clip_vision_model_w_proj, clip_img_mean, clip_img_std)
                dino_feats = get_dino_feat(images_small, dinov2_model, dinov2_img_mean, dinov2_img_std)
                
                # Get disease predictions
                xrv_normalized = xrv.datasets.normalize(images_small, mean=0.5, std=0.5)
                disease_preds = disease_classifier(xrv_normalized)
                
                # Gather information from all processes
                images_all = customized_all_gather(images, accelerator)
                age_preds_all = customized_all_gather(age_preds, accelerator)
                age_probs_all = customized_all_gather(age_probs, accelerator)
                
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_generated.jpg")
                        plot_in_grid(
                            images_all, save_to, 
                            torch.ones_like(age_preds_all, dtype=torch.bool),
                            age_preds_all, 
                            age_probs_all[:, 0]  # Young probability
                        )
                        log_imgs_i["img_generated"] = [save_to]
                
                if accelerator.is_main_process:
                    # Calculate age bias metrics
                    age_prob_diffs = age_probs_all[:, 0] - age_probs_all[:, 1]  # young - old
                    age_bias = age_prob_diffs.abs().mean().item()
                    
                    # Calculate balance metrics (closer to target_age_ratio is better)
                    young_ratio = age_preds_all.float().mean().item()
                    age_balance = abs(young_ratio - args.target_age_ratio)
                    
                    logs_i["age_bias"].append(age_bias)
                    logs_i["age_balanced"].append(1 - age_balance)  # Higher is better
                
                ################################################
                # Step 2: generate dynamic targets for age alignment
                age_targets_all, age_uncertainty_all = generate_dynamic_targets(
                    age_probs_all, target_ratio=args.target_age_ratio, w_uncertainty=True
                )
                
                # Apply uncertainty threshold
                age_targets_all[age_uncertainty_all > args.uncertainty_threshold] = -1
                
                # Get local targets
                age_targets = age_targets_all[age_probs.shape[0] * accelerator.local_process_index:
                                                  age_probs.shape[0] * (accelerator.local_process_index + 1)]
                age_uncertainty = age_uncertainty_all[age_probs.shape[0] * accelerator.local_process_index:
                                                          age_probs.shape[0] * (accelerator.local_process_index + 1)]
                
                accelerator.print(f"\tAge targets: {(age_targets != -1).sum().item()}/{age_targets.shape[0]}")
                
                ################################################
                # Step 3: generate original images using the original diffusion model (for reference)
                # Note: We use our modified model for simplicity, but in real training you'd use a non-LoRA model
                images_ori = []
                for j in range(0, noises_i.shape[0], args.val_batch_size):
                    noises_ij = noises_i[j:j + args.val_batch_size]
                    if args.train_text_encoder and args.train_unet:
                        images_ij = generate_image_no_gradient(
                            prompt_i, tokenizer, eval_text_encoder, eval_unet, vae,
                            noises_ij, num_denoising_steps, args.guidance_scale, 
                            accelerator.device, weight_dtype
                        )
                    elif args.train_text_encoder and not args.train_unet:
                        images_ij = generate_image_no_gradient(
                            prompt_i, tokenizer, eval_text_encoder, unet, vae,
                            noises_ij, num_denoising_steps, args.guidance_scale, 
                            accelerator.device, weight_dtype
                        )
                    elif not args.train_text_encoder and args.train_unet:
                        images_ij = generate_image_no_gradient(
                            prompt_i, tokenizer, text_encoder, eval_unet, vae,
                            noises_ij, num_denoising_steps, args.guidance_scale, 
                            accelerator.device, weight_dtype
                        )
                    images_ori.append(images_ij)
                images_ori = torch.cat(images_ori)
                
                # Process original images
                resized_images_ori = F.interpolate(images_ori, size=(224, 224), mode='bilinear')
                normalized_images_ori = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )((resized_images_ori + 1) / 2)
                
                # Get original age
                age_logits_ori = age_classifier(normalized_images_ori)
                age_probs_ori = F.softmax(age_logits_ori, dim=-1)
                age_preds_ori = age_logits_ori.argmax(dim=-1)
                
                # Extract features from original images
                images_small_ori = F.interpolate(images_ori, size=(args.img_size_small, args.img_size_small), mode='bilinear')
                clip_feats_ori = get_clip_feat(images_small_ori, clip_vision_model_w_proj, clip_img_mean, clip_img_std)
                dino_feats_ori = get_dino_feat(images_small_ori, dinov2_model, dinov2_img_mean, dinov2_img_std)
                
                # Get original disease predictions
                xrv_normalized_ori = xrv.datasets.normalize(images_small_ori, mean=0.5, std=0.5)
                disease_preds_ori = disease_classifier(xrv_normalized_ori)
                
                # Gather original image info from all processes
                images_ori_all = customized_all_gather(images_ori, accelerator)
                age_preds_ori_all = customized_all_gather(age_preds_ori, accelerator)
                age_probs_ori_all = customized_all_gather(age_probs_ori, accelerator)
                
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_ori.jpg")
                        plot_in_grid(
                            images_ori_all, save_to, 
                            torch.ones_like(age_preds_ori_all, dtype=torch.bool),
                            age_preds_ori_all, 
                            age_probs_ori_all[:, 0]  # Young probability
                        )
                        log_imgs_i["img_ori"] = [save_to]
            
            ################################################
            # Step 4: Compute losses for training
            loss_da_i = torch.ones(age_targets.shape, dtype=weight_dtype, device=accelerator.device) * (-1)  # Distributional Alignment
            loss_sp_i = torch.ones(age_targets.shape, dtype=weight_dtype, device=accelerator.device) * (-1)  # Semantic Preservation
            loss_dp_i = torch.ones(age_targets.shape, dtype=weight_dtype, device=accelerator.device) * (-1)  # Diagnostic Preservation
            loss_i = torch.ones(age_targets.shape, dtype=weight_dtype, device=accelerator.device) * (-1)     # Total loss
            
            idxs_i = list(range(age_targets.shape[0]))
            N_backward = math.ceil(age_targets.shape[0] / args.train_batch_size)
            for j in range(N_backward):
                # Get batch indices
                idxs_ij = idxs_i[j * args.train_batch_size:(j + 1) * args.train_batch_size]
                noises_ij = noises_i[idxs_ij]
                
                # Get targets for this batch
                age_targets_ij = age_targets[idxs_ij]
                
                # Get original features for this batch
                clip_feats_ori_ij = clip_feats_ori[idxs_ij]
                dino_feats_ori_ij = dino_feats_ori[idxs_ij]
                age_preds_ori_ij = age_preds_ori[idxs_ij]
                age_probs_ori_ij = age_probs_ori[idxs_ij]
                disease_preds_ori_ij = disease_preds_ori[idxs_ij]
                
                # Generate images with gradient tracking
                images_ij = generate_image_w_gradient(
                    prompt_i, tokenizer, text_encoder, unet, vae,
                    noises_ij, num_denoising_steps, args.guidance_scale, 
                    accelerator.device, weight_dtype
                )
                
                # Process generated images
                resized_images_ij = F.interpolate(images_ij, size=(224, 224), mode='bilinear')
                normalized_images_ij = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )((resized_images_ij + 1) / 2)
                
                # Get age predictions for generated images
                age_logits_ij = age_classifier(normalized_images_ij)
                age_probs_ij = F.softmax(age_logits_ij, dim=-1)
                
                # Extract features from generated images
                images_small_ij = F.interpolate(images_ij, size=(args.img_size_small, args.img_size_small), mode='bilinear')
                clip_feats_ij = get_clip_feat(images_small_ij, clip_vision_model_w_proj, clip_img_mean, clip_img_std)
                dino_feats_ij = get_dino_feat(images_small_ij, dinov2_model, dinov2_img_mean, dinov2_img_std)
                
                # Get disease predictions
                xrv_normalized_ij = xrv.datasets.normalize(images_small_ij, mean=0.5, std=0.5)
                disease_preds_ij = disease_classifier(xrv_normalized_ij)
                
                # 1. Distributional Alignment Loss (ℒDA)
                loss_da_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) * (-1)
                idxs_w_age_targets = (age_targets_ij != -1).nonzero().view([-1])
                if len(idxs_w_age_targets) > 0:
                    loss_da_ij_w_targets = CE_loss(age_logits_ij[idxs_w_age_targets], 
                                                  age_targets_ij[idxs_w_age_targets])
                    loss_da_ij[idxs_w_age_targets] = loss_da_ij_w_targets
                
                # 2. Semantic Preservation Loss (ℒSP)
                loss_clip_ij = - (clip_feats_ij * clip_feats_ori_ij).sum(dim=-1) + 1
                loss_dino_ij = - (dino_feats_ij * dino_feats_ori_ij).sum(dim=-1) + 1
                loss_sp_ij = (loss_clip_ij + loss_dino_ij) / 2
                
                # 3. CXR Diagnostic Preservation Loss (ℒCXR-DP)
                # 3.1 Feature Similarity Loss
                loss_feature_ij = F.mse_loss(images_small_ij, images_small_ori[idxs_ij], reduction='none').mean(dim=[1, 2, 3])
                
                # 3.2 Perceptual Loss using CLIP
                loss_perceptual_ij = loss_clip_ij  # Already calculated above
                
                # 3.3 Diagnostic Consistency Loss
                loss_diagnostic_ij = F.mse_loss(disease_preds_ij, disease_preds_ori_ij, reduction='none').mean(dim=1)
                
                # Combine CXR-DP components
                loss_dp_ij = (
                    args.feature_similarity_weight * loss_feature_ij + 
                    args.perceptual_weight * loss_perceptual_ij + 
                    args.diagnostic_consistency_weight * loss_diagnostic_ij
                )
                
                # Calculate dynamic weights based on age alignment
                age_attr_indicators = torch.ones(len(idxs_ij), dtype=torch.bool, device=accelerator.device)
                dynamic_weights = gen_dynamic_weights(
                    age_attr_indicators, age_targets_ij, 
                    age_preds_ori_ij, age_probs_ori_ij,
                    factor=args.dynamic_weight_factor
                )
                
                # Final loss calculation - combine all components with weights
                loss_ij = (
                    args.distributional_alignment_weight * loss_da_ij + 
                    args.semantic_preservation_weight * dynamic_weights * loss_sp_ij + 
                    args.diagnostic_preservation_weight * loss_dp_ij
                )
                
                # Backward pass
                accelerator.backward(loss_ij.mean())
                
                # Store losses for logging
                with torch.no_grad():
                    loss_da_i[idxs_ij] = loss_da_ij.to(loss_da_i.dtype)
                    loss_sp_i[idxs_ij] = loss_sp_ij.to(loss_sp_i.dtype)
                    loss_dp_i[idxs_ij] = loss_dp_ij.to(loss_dp_i.dtype)
                    loss_i[idxs_ij] = loss_ij.to(loss_i.dtype)
            
            # Gather losses from all processes for logging
            accelerator.wait_for_everyone()
            loss_da_all = customized_all_gather(loss_da_i, accelerator)
            loss_sp_all = customized_all_gather(loss_sp_i, accelerator)
            loss_dp_all = customized_all_gather(loss_dp_i, accelerator)
            loss_all = customized_all_gather(loss_i, accelerator)
            
            # Filter valid losses
            loss_da_all = loss_da_all[loss_da_all != -1]
            loss_sp_all = loss_sp_all[loss_sp_all != -1]
            loss_dp_all = loss_dp_all[loss_dp_all != -1]
            loss_all = loss_all[loss_all != -1]
            
            if accelerator.is_main_process:
                logs_i["loss_da"].append(loss_da_all)
                logs_i["loss_sp"].append(loss_sp_all)
                logs_i["loss_dp"].append(loss_dp_all)
                logs_i["loss"].append(loss_all)
            
            # Process logs
            if accelerator.is_main_process:
                for key in ["loss_da", "loss_sp", "loss_dp", "loss"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)
                    else:
                        logs_i[key] = torch.cat(logs_i[key])
                
                for key in ["age_bias", "age_balanced"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)
            
            # Manually sync gradients across processes
            grad_is_finite = True
            with torch.no_grad():
                if args.train_text_encoder:
                    for p in [p for p in text_encoder.parameters() if p.requires_grad]:
                        if not torch.isfinite(p.grad).all():
                            grad_is_finite = False
                        torch.distributed.all_reduce(p.grad, torch.distributed.ReduceOp.SUM)
                        p.grad = p.grad / accelerator.num_processes / N_backward
                
                if args.train_unet:
                    for p in [p for p in unet.parameters() if p.requires_grad]:
                        if not torch.isfinite(p.grad).all():
                            grad_is_finite = False
                        torch.distributed.all_reduce(p.grad, torch.distributed.ReduceOp.SUM)
                        p.grad = p.grad / accelerator.num_processes / N_backward
            
            # Apply gradients if they are finite
            if grad_is_finite:
                optimizer.step()
            else:
                accelerator.print(f"Gradients are not finite, skipping update!")
            
            lr_scheduler.step()
            
            # Update progress
            progress_bar.update(1)
            global_step += 1
            
            # Log training stats
            if accelerator.is_main_process and is_wandb_available():
                import wandb
                
                # Calculate mean losses for logging
                log_dict = {}
                for key, values in logs_i.items():
                    if isinstance(values, torch.Tensor):
                        log_dict[f"train_{key}"] = values.mean().item()
                    elif isinstance(values, list):
                        log_dict[f"train_{key}"] = np.mean(values)
                    else:
                        log_dict[f"train_{key}"] = values
                
                # Log learning rate
                log_dict["lr"] = lr_scheduler.get_last_lr()[0]
                
                wandb_tracker.log(log_dict, step=global_step)
                
                # Log images
                for key, values in log_imgs_i.items():
                    if values:
                        wandb_tracker.log({
                            f"train_{key}": wandb.Image(
                                values[0],
                                caption=prompt_i,
                            )
                        }, step=global_step)
            
            # Run evaluation at specified intervals
            if global_step % args.evaluate_every_n_iter == 0:
                evaluate_process(
                    args, tokenizer, text_encoder, unet, vae, age_classifier,
                    accelerator, weight_dtype, epoch, global_step
                )
            
            # Save checkpoint
            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # Check if we need to delete old checkpoints
                    if args.checkpoints_total_limit is not None:
                        name = "checkpoint_tmp"
                        clean_checkpoint(args.ckpts_save_dir, name, args.checkpoints_total_limit)
                    
                    # Save temporary checkpoint
                    save_path = os.path.join(args.ckpts_save_dir, f"checkpoint_tmp-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                
                if global_step % args.checkpointing_steps_long == 0:
                    # Save permanent checkpoint
                    save_path = os.path.join(args.ckpts_save_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved permanent checkpoint to {save_path}")
            
            # Clean up memory
            torch.cuda.empty_cache()
            
            # Check if we've reached the maximum number of steps
            if global_step >= args.max_train_steps:
                break
    
    # End training
    accelerator.wait_for_everyone()
    
    # Final save of LoRA weights
    if accelerator.is_main_process:
        logger.info("***** Saving final model *****")
        
        os.makedirs(os.path.join(args.output_dir, args.proj_name, folder_name, "lora_weights"), exist_ok=True)
        
        # Save final LoRA weights in HuggingFace format for easy inference
        if args.train_text_encoder:
            text_encoder.save_pretrained(os.path.join(args.output_dir, args.proj_name, folder_name, "lora_weights", "text_encoder"))
            
        if args.train_unet:
            unet.save_pretrained(os.path.join(args.output_dir, args.proj_name, folder_name, "lora_weights", "unet"))
            
        # Save the complete pipeline configuration
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder if args.train_text_encoder else None,
            unet=unet if args.train_unet else None,
            torch_dtype=weight_dtype
        )
        
        pipeline.save_pretrained(os.path.join(args.output_dir, args.proj_name, folder_name, "pipeline"))
        logger.info(f"Saved final pipeline to {os.path.join(args.output_dir, args.proj_name, folder_name, 'pipeline')}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
