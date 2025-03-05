import PIL
import peft
import copy
import torch
import random
import os, sys
import argparse
import requests
from io import BytesIO
from IPython.display import display
from torchvision.ops import box_convert
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel

# Setup device and model
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "path-to-inpainting-model",  # Base CXR-inpainting model
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

# Load LoRA weights for bias mitigation
sd_pipe.load_lora_weights(
    "./CXR-AnoFAIR", 
    adapter_name="CXR-AnoFAIR"
)
sd_pipe.set_adapters(["CXR-AnoFAIR"], adapter_weights=[0.5])

# Inference function with random seed
def generate_image(image_path, mask_path, prompt, negative_prompt, pipe, seed=None):
    try:
        in_image = Image.open(image_path)
        in_mask = Image.open(mask_path)
    except IOError as e:
        print(f"Loading error: {e}")
        return None
    
    # Use random seed if none is provided
    if seed is None:
        seed = random.randint(0, 2147483647)
    
    print(f"Using seed: {seed}")
    generator = torch.Generator(device).manual_seed(seed)
    
    result = pipe(
        image=in_image, 
        mask_image=in_mask, 
        prompt=prompt,
        negative_prompt=negative_prompt, 
        generator=generator
    )
    return result.images[0]

# Example usage
image = 'path-to-cxr-image'
mask = "path-to-disease-mask"
prompt = "A chest X-ray showing pneumonia"
negative_prompt = "low resolution, low quality, blurry, noise, disfigured"

# Generate anonymized image with random seed
anonymized_image = generate_image(
    image_path=image, 
    mask_path=mask, 
    prompt=prompt,
    negative_prompt=negative_prompt, 
    pipe=sd_pipe
)

# Display or save the anonymized image
anonymized_image.save("anonymized_cxr.png")