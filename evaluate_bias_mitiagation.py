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

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import pickle as pkl
import glob
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Define classifier models for gender and age
class AttributeClassifier(nn.Module):
    """Attribute classifier for CXR images (gender or age)"""
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # Initialize model using ResNet50 as in AttNzr
        self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(2048, 2)  # 2 outputs for binary attributes
        
        # Load pretrained weights
        if pretrained_path is not None and os.path.exists(pretrained_path):
            self.model.load_state_dict(torch.load(pretrained_path))
        
        # Set to evaluation mode
        self.model.eval()
        
    def forward(self, x):
        return self.model(x)

def image_grid(imgs, rows, cols):
    """Create a grid of images"""
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def plot_in_grid(images, save_to, gender_preds=None, gender_probs=None, age_preds=None, age_probs=None):
    """
    Plot images in a grid with attribute predictions
    """
    images_to_plot = []
    for idx in range(images.shape[0]):
        img = images[idx]
        
        # Convert tensor to PIL image (normalize from [-1,1] to [0,1])
        if img.shape[0] == 1:  # Handle grayscale images
            img_pil = transforms.ToPILImage()(img.repeat(3, 1, 1)*0.5 + 0.5)
        else:  # Handle RGB images
            img_pil = transforms.ToPILImage()(img*0.5 + 0.5)
        
        # Create drawing object
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        
        # Add gender information if available
        if gender_preds is not None and gender_probs is not None:
            gender_pred = gender_preds[idx].item()
            gender_prob = gender_probs[idx].item()
            
            if gender_pred == 0:
                gender_text = f"Female: {gender_prob:.2f}"
                border_color = "red"
            else:
                gender_text = f"Male: {gender_prob:.2f}"
                border_color = "blue"
                
            # Add text and border
            draw.text((10, 10), gender_text, fill=border_color, font=font)
            img_pil = ImageOps.expand(img_pil, border=(5, 0, 0, 0), fill=border_color)
        
        # Add age information if available
        if age_preds is not None and age_probs is not None:
            age_pred = age_preds[idx].item()
            age_prob = age_probs[idx].item()
            
            if age_pred == 0:
                age_text = f"Young: {age_prob:.2f}"
                border_color = "green"
            else:
                age_text = f"Old: {age_prob:.2f}"
                border_color = "orange"
                
            # Add text and border
            y_pos = 30 if gender_preds is not None else 10
            draw.text((10, y_pos), age_text, fill=border_color, font=font)
            img_pil = ImageOps.expand(img_pil, border=(0, 5, 0, 0), fill=border_color)
        
        # Add image index
        draw.text((10, img_pil.height - 20), f"#{idx}", fill="white", font=font)
        
        images_to_plot.append(img_pil)
    
    # Create grid layout
    N_imgs = len(images_to_plot)
    N1 = int(math.sqrt(N_imgs))
    N2 = math.ceil(N_imgs / N1)

    # Pad with empty images if needed
    for i in range(N1*N2-N_imgs):
        images_to_plot.append(
            Image.new('RGB', color="black", size=images_to_plot[0].size)
        )
    
    # Create and save grid
    grid = image_grid(images_to_plot, N1, N2)
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    grid.save(save_to, quality=90)

def plot_attribute_distribution(values, labels, save_to, title="Distribution"):
    """Plot distribution of attribute values"""
    plt.figure(figsize=(10, 6))
    
    # Count occurrences
    unique_values, counts = np.unique(values, return_counts=True)
    
    # Create bar plot
    bars = plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), [labels[v] for v in unique_values])
    plt.ylabel("Count")
    plt.title(title)
    
    # Add percentage labels
    total = sum(counts)
    for i, count in enumerate(counts):
        percentage = count / total * 100
        plt.text(i, count + 0.5, f"{percentage:.1f}%", ha='center')
        
        # Color bars based on labels
        if labels[unique_values[i]] in ["Female", "Young"]:
            bars[i].set_color('skyblue')
        else:
            bars[i].set_color('salmon')
    
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

def calculate_bias_metrics(preds, target_ratio=0.5):
    """
    Calculate bias metrics from predictions
    
    Args:
        preds: Tensor of binary predictions
        target_ratio: Target ratio for class 1 (0.5 means balanced)
        
    Returns:
        Dictionary of bias metrics
    """
    # Get class counts
    class_counts = torch.bincount(preds, minlength=2)
    total = class_counts.sum().item()
    
    # Calculate ratios
    class_0_ratio = class_counts[0].item() / total
    class_1_ratio = class_counts[1].item() / total
    
    # Calculate bias (absolute difference between classes)
    bias = abs(class_1_ratio - class_0_ratio)
    
    # Calculate balance (how close to target ratio)
    balance = 1 - abs(class_1_ratio - target_ratio)
    
    return {
        'class_0_ratio': class_0_ratio,
        'class_1_ratio': class_1_ratio,
        'bias': bias,
        'balance': balance
    }

def load_images(image_paths, img_size, batch_size, device):
    """Load and preprocess CXR images in batches using torchxrayvision preprocessing"""
    all_images = []
    
    # Define transforms for chest X-rays
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(img_size)
    ])
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        
        for img_path in batch_paths:
            try:
                # Load image
                img = Image.open(img_path)
                # Convert to numpy array
                img_np = np.array(img)
                
                # Handle different channel configurations
                if len(img_np.shape) > 2:
                    # Convert RGB to grayscale by taking mean across channels
                    img_np = img_np.mean(axis=2)
                
                # Normalize to [-1024, 1024] range as required by torchxrayvision
                img_np = xrv.datasets.normalize(img_np, 255)
                
                # Add channel dimension if needed (needed for transform)
                if len(img_np.shape) == 2:
                    img_np = img_np[None, ...]
                
                # Apply transforms
                img_np = transform(img_np)
                
                # Convert to tensor
                img_tensor = torch.from_numpy(img_np).float()
                
                batch_imgs.append(img_tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if batch_imgs:
            # Convert batch to tensor
            batch_imgs = torch.stack(batch_imgs).to(device)
            all_images.append(batch_imgs)
    
    # Concatenate all batches if any were loaded
    if all_images:
        return torch.cat(all_images)
    else:
        return torch.empty(0, device=device)

def parse_args():
    parser = argparse.ArgumentParser(description="Bias evaluation script for CXR-AnoFAIR")

    parser.add_argument(
        "--gpu_id", 
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--gender_classifier_path",
        type=str,
        required=True,
        help="Path to pre-trained gender classifier",
    )
    parser.add_argument(
        "--age_classifier_path",
        type=str,
        required=True,
        help="Path to pre-trained age classifier",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images to evaluate",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=16,
        help="Batch size for processing images",
    )
    parser.add_argument(
        '--img_size', 
        type=int, 
        default=224,
        help="Size to resize images for processing",
    )
    parser.add_argument(
        '--target_gender_ratio',
        type=float,
        default=0.5,
        help="Target ratio for male gender (0.5 means balanced)",
    )
    parser.add_argument(
        '--target_age_ratio',
        type=float,
        default=0.5,
        help="Target ratio for young age (0.5 means balanced)",
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load attribute classifiers
    print(f"Loading gender classifier from {args.gender_classifier_path}")
    gender_classifier = AttributeClassifier(args.gender_classifier_path)
    gender_classifier.to(device)
    gender_classifier.eval()
    
    print(f"Loading age classifier from {args.age_classifier_path}")
    age_classifier = AttributeClassifier(args.age_classifier_path)
    age_classifier.to(device)
    age_classifier.eval()
    
    # Get all image paths
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.dcm']:
        image_paths.extend(glob.glob(os.path.join(args.images_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_paths)} images to evaluate")
    
    # Load and process images
    print("Processing images...")
    images = load_images(image_paths, args.img_size, args.batch_size, device)
    
    if images.shape[0] == 0:
        print("No images were successfully loaded. Exiting.")
        return
    
    # Store results
    results = {
        'gender_metrics': None,
        'age_metrics': None,
        'gender_counts': None,
        'age_counts': None,
        'combined_counts': None
    }
    
    # Evaluate gender distribution
    print("Evaluating gender distribution...")
    with torch.no_grad():
        # Preprocess for the classifier which expects normalized RGB images
        processed_images = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )(images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images)
        
        gender_logits = gender_classifier(processed_images)
        gender_probs = F.softmax(gender_logits, dim=1)
        gender_preds = gender_logits.argmax(dim=1)
        
        age_logits = age_classifier(processed_images)
        age_probs = F.softmax(age_logits, dim=1)
        age_preds = age_logits.argmax(dim=1)
    
    # Calculate gender metrics
    gender_metrics = calculate_bias_metrics(gender_preds, args.target_gender_ratio)
    results['gender_metrics'] = gender_metrics
    
    # Calculate age metrics
    age_metrics = calculate_bias_metrics(age_preds, args.target_age_ratio)
    results['age_metrics'] = age_metrics
    
    # Calculate and store counts for plotting
    gender_counts = torch.bincount(gender_preds, minlength=2).cpu().numpy()
    age_counts = torch.bincount(age_preds, minlength=2).cpu().numpy()
    
    results['gender_counts'] = gender_counts
    results['age_counts'] = age_counts
    
    # Calculate combined gender-age counts
    combined_labels = gender_preds * 2 + age_preds  # 0=Female-Young, 1=Female-Old, 2=Male-Young, 3=Male-Old
    combined_counts = torch.bincount(combined_labels, minlength=4).cpu().numpy()
    results['combined_counts'] = combined_counts
    
    # Plot gender distribution
    plot_attribute_distribution(
        gender_preds.cpu().numpy(),
        {0: "Female", 1: "Male"},
        os.path.join(args.save_dir, "gender_distribution.png"),
        title="Gender Distribution"
    )
    
    # Plot age distribution
    plot_attribute_distribution(
        age_preds.cpu().numpy(),
        {0: "Young", 1: "Old"},
        os.path.join(args.save_dir, "age_distribution.png"),
        title="Age Distribution"
    )
    
    # Plot combined distribution
    plt.figure(figsize=(10, 6))
    labels = ["Female-Young", "Female-Old", "Male-Young", "Male-Old"]
    bars = plt.bar(range(4), combined_counts)
    plt.xticks(range(4), labels, rotation=45)
    plt.ylabel("Count")
    plt.title("Combined Gender-Age Distribution")
    
    # Add percentage labels
    total = sum(combined_counts)
    for i, count in enumerate(combined_counts):
        percentage = count / total * 100
        plt.text(i, count + 0.5, f"{percentage:.1f}%", ha='center')
        
        # Color bars
        if i == 0:  # Female-Young
            bars[i].set_color('lightblue')
        elif i == 1:  # Female-Old
            bars[i].set_color('darkblue')
        elif i == 2:  # Male-Young
            bars[i].set_color('lightsalmon')
        else:  # Male-Old
            bars[i].set_color('darksalmon')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "combined_distribution.png"))
    plt.close()
    
    # Save visualization of images with predictions
    print("Generating visualization samples...")
    plot_in_grid(
        images[:min(16, len(images))],
        os.path.join(args.save_dir, "sample_predictions.png"),
        gender_preds=gender_preds[:min(16, len(images))],
        gender_probs=gender_probs.max(dim=1).values[:min(16, len(images))],
        age_preds=age_preds[:min(16, len(images))],
        age_probs=age_probs.max(dim=1).values[:min(16, len(images))]
    )
    
    # Save detailed results as CSV
    details = []
    for i in range(min(len(image_paths), len(gender_preds))):
        details.append({
            'image_path': image_paths[i],
            'gender_pred': 'Male' if gender_preds[i].item() == 1 else 'Female',
            'gender_confidence': gender_probs[i, gender_preds[i]].item(),
            'age_pred': 'Old' if age_preds[i].item() == 1 else 'Young',
            'age_confidence': age_probs[i, age_preds[i]].item()
        })
    
    pd.DataFrame(details).to_csv(os.path.join(args.save_dir, 'detailed_results.csv'), index=False)
    
    # Create summary report
    with open(os.path.join(args.save_dir, 'summary_report.txt'), 'w') as f:
        f.write("CXR-AnoFAIR Bias Evaluation Summary\n")
        f.write("================================\n\n")
        
        f.write("Gender Distribution Metrics:\n")
        f.write(f"  Female ratio: {gender_metrics['class_0_ratio']:.4f}\n")
        f.write(f"  Male ratio: {gender_metrics['class_1_ratio']:.4f}\n")
        f.write(f"  Gender bias: {gender_metrics['bias']:.4f}\n")
        f.write(f"  Gender balance score: {gender_metrics['balance']:.4f}\n\n")
        
        f.write("Age Distribution Metrics:\n")
        f.write(f"  Young ratio: {age_metrics['class_0_ratio']:.4f}\n")
        f.write(f"  Old ratio: {age_metrics['class_1_ratio']:.4f}\n")
        f.write(f"  Age bias: {age_metrics['bias']:.4f}\n")
        f.write(f"  Age balance score: {age_metrics['balance']:.4f}\n\n")
        
        f.write("Combined Distribution:\n")
        labels = ["Female-Young", "Female-Old", "Male-Young", "Male-Old"]
        for i, label in enumerate(labels):
            f.write(f"  {label}: {combined_counts[i]} ({combined_counts[i]/total*100:.1f}%)\n")
    
    # Save results as pickle file
    with open(os.path.join(args.save_dir, 'results.pkl'), 'wb') as f:
        pkl.dump(results, f)
    
    print(f"Evaluation complete! Results saved to {args.save_dir}")
    
    return results

if __name__ == "__main__":
    main()
