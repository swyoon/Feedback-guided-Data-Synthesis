#!/usr/bin/env python3
"""
Minimal Working Example: Batch Image Generation with FeedbackGuidedDiffusion

This script demonstrates how to use the FeedbackGuidedDiffusion class from this repository
to generate a batch of images with various hyperparameters.
"""

import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
from utils_gen import get_pipe, get_classifier


def setup_models():
    """
    Initialize the diffusion pipeline and classifier for feedback guidance.
    Uses utility functions from utils_gen.py for consistency.
    
    Returns:
        pipe: FeedbackGuidedDiffusion pipeline
        fg_classifier: Classifier for feedback guidance (optional)
        fg_preprocessing: Preprocessing transforms for classifier
    """
    # Use the existing get_pipe function from utils_gen.py
    # Note: Make sure to update CLIP_MODEL_PATH and LDM_MODEL_PATH in utils_gen.py
    CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"
    LDM_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
    pipe = get_pipe(CLIP_MODEL_PATH, LDM_MODEL_PATH)
    
    # Create a dummy dataframe for get_classifier function compatibility
    # (the original function expects args and df parameters)
    class Args:
        pass
    
    args = Args()
    df = pd.DataFrame()  # Empty dataframe - not used by get_classifier
    
    # Use the existing get_classifier function from utils_gen.py
    fg_classifier, fg_preprocessing = get_classifier(args, df)
    
    return pipe, fg_classifier, fg_preprocessing

def generate_batch_images(
    pipe, 
    prompts, 
    fg_classifier=None, 
    fg_preprocessing=None,
    output_dir="generated_images",
    cfg=7.5,
    fg_criterion="entropy",
    fg_scale=0.03,
    cls_index=281
):
    """
    Generate a batch of images using the FeedbackGuidedDiffusion pipeline.
    
    Args:
        pipe: FeedbackGuidedDiffusion pipeline
        prompts: List of text prompts
        fg_classifier: Classifier for feedback guidance (optional)
        fg_preprocessing: Preprocessing for classifier
        output_dir: Directory to save generated images
        cfg: Classifier-free guidance scale (5.0-15.0 recommended)
        fg_criterion: "loss" or "entropy" for feedback guidance
        fg_scale: Feedback guidance scale (0.01-0.1 recommended)
        cls_index: ImageNet class index for feedback guidance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generated_images = []
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: '{prompt}'")
        
        try:
            # Create generator for reproducible results
            generator = torch.Generator(device=pipe.device)
            generator.manual_seed(42 + i)  # Different seed for each image
            
            # Generate image
            if fg_classifier is not None and fg_scale > 0:
                # Generate with feedback guidance
                result = pipe(
                    prompt=prompt,
                    cfg=cfg,
                    fg_criterion=fg_criterion,
                    fg_scale=fg_scale,
                    fg_classifier=fg_classifier,
                    fg_preprocessing=fg_preprocessing,
                    generator=generator,
                    num_inference_steps=30,
                    num_images_per_prompt=1,
                    cls_index=cls_index,
                    guidance_freq=5
                )
            else:
                # Generate without feedback guidance
                result = pipe(
                    prompt=prompt,
                    cfg=cfg,
                    fg_criterion=fg_criterion,
                    fg_scale=0,  # Disable feedback guidance
                    fg_classifier=None,
                    fg_preprocessing=None,
                    generator=generator,
                    num_inference_steps=30,
                    num_images_per_prompt=1,
                    cls_index=None,
                    guidance_freq=5
                )
            
            # Extract image and criteria value
            if isinstance(result, tuple):
                image, criteria_final_value = result
            else:
                image = result[0] if isinstance(result, list) else result
                criteria_final_value = -10000  # Default value when no feedback
            
            print(f"  Feedback criteria value: {criteria_final_value}")
            
            # Save image
            if isinstance(image, list):
                image = image[0]
            
            # Resize and save
            image = image.resize((256, 256))
            output_path = os.path.join(output_dir, f"image_{i:03d}.png")
            image.save(output_path)
            print(f"  Saved: {output_path}")
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"image_{i:03d}.txt")
            with open(metadata_path, 'w') as file:
                file.write(str(criteria_final_value))
            
            generated_images.append(image)
            
        except Exception as e:
            print(f"  Error generating image: {e}")
            continue
    
    print(f"\nGenerated {len(generated_images)} images successfully!")
    return generated_images


def compute_entropy_for_images(image_dir, fg_classifier, fg_preprocessing):
    """
    Compute predictive entropy for all images in a directory.
    
    Args:
        image_dir: Directory containing generated images
        fg_classifier: Classifier model
        fg_preprocessing: Preprocessing transforms for classifier
        
    Returns:
        List of entropy values for each image
    """
    entropy_values = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    fg_classifier.eval()
    fg_classifier.to(torch.float32)
    
    print(f"Computing entropy for {len(image_files)} images in {image_dir}...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            
            # Apply ImageNet preprocessing: ToTensor + normalization
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0).cuda()
            # Apply classifier preprocessing (ImageNet normalization)
            image_preprocessed = fg_preprocessing(image_tensor)
            
            # Get classifier predictions
            with torch.no_grad():
                logits = fg_classifier(image_preprocessed)
                
                # Compute entropy: H = -sum(p * log(p))
                prob = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                entropy = -(prob * log_prob).sum(dim=-1).item()
                
                entropy_values.append(entropy)
                print(f"  {img_file}: entropy = {entropy:.4f}")
                
        except Exception as e:
            print(f"  Error processing {img_file}: {e}")
            continue
    
    return entropy_values

# Set memory efficient attention if using limited GPU memory
# torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == "__main__":
    print("Setting up models...")
    pipe, fg_classifier, fg_preprocessing = setup_models()
    
    # Example prompts
    prompts = [
        "a photo of a cat",
        "a photo of a dog", 
        "a photo of a bird",
        "a photo of a horse",
        "a photo of a tiger"
    ]
    
    print(f"Generating {len(prompts)} images with feedback guidance...")
    images_with_feedback = generate_batch_images(
        pipe, prompts, fg_classifier, fg_preprocessing, 
        output_dir="with_feedback",
        fg_scale=0.1  # Enable feedback guidance
    )
    
    print(f"\nGenerating {len(prompts)} images without feedback guidance...")
    images_without_feedback = generate_batch_images(
        pipe, prompts, fg_classifier=None, fg_preprocessing=None,
        output_dir="without_feedback", 
        fg_scale=0  # Disable feedback guidance
    )
    
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Generated {len(images_with_feedback)} images with feedback guidance!")
    print(f"Generated {len(images_without_feedback)} images without feedback guidance!")
    print("Check the 'with_feedback' and 'without_feedback' directories for comparison.")
    
    # Compute and compare predictive entropy
    print(f"\n=== COMPUTING PREDICTIVE ENTROPY ===")
    
    # Compute entropy for images with feedback guidance
    entropy_with_feedback = compute_entropy_for_images(
        "with_feedback", fg_classifier, fg_preprocessing
    )
    
    # Compute entropy for images without feedback guidance  
    entropy_without_feedback = compute_entropy_for_images(
        "without_feedback", fg_classifier, fg_preprocessing
    )
    
    # Compare entropy values
    print(f"\n=== ENTROPY COMPARISON ===")
    avg_entropy_with = np.mean(entropy_with_feedback)
    avg_entropy_without = np.mean(entropy_without_feedback)
    std_entropy_with = np.std(entropy_with_feedback)
    std_entropy_without = np.std(entropy_without_feedback)
    
    print(f"Images WITH feedback guidance:")
    print(f"  Average entropy: {avg_entropy_with:.4f} ± {std_entropy_with:.4f}")
    print(f"  Individual values: {[f'{x:.4f}' for x in entropy_with_feedback]}")
    
    print(f"\nImages WITHOUT feedback guidance:")
    print(f"  Average entropy: {avg_entropy_without:.4f} ± {std_entropy_without:.4f}")
    print(f"  Individual values: {[f'{x:.4f}' for x in entropy_without_feedback]}")
    
    difference = avg_entropy_with - avg_entropy_without
    print(f"\nEntropy difference (with - without): {difference:.4f}")
       