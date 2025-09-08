import argparse
import numpy as np
import cv2
import json
import models
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import os
from math import ceil
from PIL import Image
from pathlib import Path
import time


def load_image(image_path, transform=None):
    """Load and preprocess a single image"""
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def multi_scale_predict(model, image_A, image_B, scales, num_classes, flip=False):
    """Multi-scale prediction function"""
    H, W = (image_A.size(2), image_A.size(3))
    upsize = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w = upsize[0] - H, upsize[1] - W
    image_A = F.pad(image_A, pad=(0, pad_w, 0, pad_h), mode='reflect')
    image_B = F.pad(image_B, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image_A.shape[2], image_A.shape[3]))

    for scale in scales:
        scaled_img_A = F.interpolate(image_A, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_img_B = F.interpolate(image_B, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(A_l=scaled_img_A, B_l=scaled_img_B))
        
        if flip:
            fliped_img_A = scaled_img_A.flip(-1)
            fliped_img_B = scaled_img_B.flip(-1)
            fliped_predictions = upsample(model(A_l=fliped_img_A, B_l=fliped_img_B))
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)
    
    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def main():
    args = parse_arguments()

    # Load config
    assert args.config, "Config file path is required"
    config = json.load(open(args.config))
    scales = [1.0, 1.25]  # Multi-scale inference
    
    # Setup
    num_classes = 2
    palette = get_voc_pallete(num_classes)
    
    # Define image transforms (adjust based on your training preprocessing)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Load model
    config['model']['supervised'] = True
    config['model']['semi'] = False
    model = models.Consistency_ResNet50_CD(num_classes=num_classes, conf=config['model'], testing=True)
    print(f'\nModel: {model}\n')
    
    checkpoint = torch.load(args.model, weights_only=False)
    model = torch.nn.DataParallel(model)
    
    try:
        print("Loading the state dictionary...")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.eval()
    model.cuda()
    
    # Create output directory - Enhanced with better error handling
    if args.save:
        print(f"Checking/creating output directory: {args.output_dir}")
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"✓ Output directory ready: {args.output_dir}")
            # Test write permissions
            test_file = os.path.join(args.output_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("✓ Write permissions confirmed")
        except Exception as e:
            print(f"✗ Error with output directory: {e}")
            return
    
    # Load input images
    print(f"Loading images:")
    print(f"T1: {args.image_t1}")
    print(f"T2: {args.image_t2}")
    
    # Check if images exist
    if not os.path.exists(args.image_t1):
        print(f"✗ T1 image not found: {args.image_t1}")
        return
    if not os.path.exists(args.image_t2):
        print(f"✗ T2 image not found: {args.image_t2}")
        return
    print("✓ Both images found")
    
    try:
        image_A = load_image(args.image_t1, transform)
        image_B = load_image(args.image_t2, transform)
        print("✓ Images loaded successfully")
    except Exception as e:
        print(f"✗ Error loading images: {e}")
        return
    
    # Add batch dimension and move to GPU
    image_A = image_A.unsqueeze(0).cuda()
    image_B = image_B.unsqueeze(0).cuda()
    
    print("Running inference...")
    start_time = time.time()
    
    # Predict
    with torch.no_grad():
        output = multi_scale_predict(model, image_A, image_B, scales, num_classes, flip=args.flip)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.3f} seconds")
    
    # Get prediction
    prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
    
    # Get confidence/probability map
    confidence = np.max(output, axis=0)
    
    # Save results with enhanced debugging
    if args.save:
        print(f"Saving results to: {args.output_dir}")
        
        # Get base name for output files
        base_name = Path(args.image_t1).stem + "_" + Path(args.image_t2).stem
        print(f"Base filename: {base_name}")
        
        try:
            # Save colorized prediction
            prediction_colored = colorize_mask(prediction, palette)
            pred_path = os.path.join(args.output_dir, f"{base_name}_prediction.png")
            prediction_colored.save(pred_path)
            print(f"✓ Saved colorized prediction: {pred_path}")
        except Exception as e:
            print(f"✗ Error saving colorized prediction: {e}")
        
        try:
            # Save binary mask
            binary_mask = prediction * 255  # Convert to 0-255 range
            binary_path = os.path.join(args.output_dir, f"{base_name}_binary.png")
            success = cv2.imwrite(binary_path, binary_mask)
            if success:
                print(f"✓ Saved binary mask: {binary_path}")
            else:
                print(f"✗ Failed to save binary mask: {binary_path}")
        except Exception as e:
            print(f"✗ Error saving binary mask: {e}")
        
        try:
            # Save confidence map
            confidence_normalized = (confidence * 255).astype(np.uint8)
            conf_path = os.path.join(args.output_dir, f"{base_name}_confidence.png")
            success = cv2.imwrite(conf_path, confidence_normalized)
            if success:
                print(f"✓ Saved confidence map: {conf_path}")
            else:
                print(f"✗ Failed to save confidence map: {conf_path}")
        except Exception as e:
            print(f"✗ Error saving confidence map: {e}")
        
        # Optionally save raw prediction probabilities
        if args.save_probs:
            try:
                prob_path = os.path.join(args.output_dir, f"{base_name}_probabilities.npy")
                np.save(prob_path, output)
                print(f"✓ Saved raw probabilities: {prob_path}")
            except Exception as e:
                print(f"✗ Error saving probabilities: {e}")
                
        # Verify output directory contents
        try:
            if os.path.exists(args.output_dir):
                files = os.listdir(args.output_dir)
                print(f"Files in output directory ({len(files)} files): {files}")
                # Show file sizes
                for file in files:
                    file_path = os.path.join(args.output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"  - {file}: {size} bytes")
            else:
                print(f"✗ Output directory does not exist: {args.output_dir}")
        except Exception as e:
            print(f"✗ Error checking output directory: {e}")
    else:
        print("⚠️  Save flag not set. Use --save to save results.")
    
    # Print statistics
    change_pixels = np.sum(prediction == 1)
    total_pixels = prediction.shape[0] * prediction.shape[1]
    change_percentage = (change_pixels / total_pixels) * 100
    
    print(f"\nResults:")
    print(f"Image size: {prediction.shape}")
    print(f"Changed pixels: {change_pixels:,}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Change percentage: {change_percentage:.2f}%")
    print(f"Average confidence: {confidence.mean():.3f}")
    
    # Show prediction statistics
    unique_values, counts = np.unique(prediction, return_counts=True)
    print(f"Prediction value distribution:")
    for val, count in zip(unique_values, counts):
        print(f"  Value {val}: {count:,} pixels ({count/total_pixels*100:.2f}%)")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Change Detection Inference on Custom Image Pair')
    parser.add_argument('--config', default="configs/config_carto.json", type=str,
                        help='Path to the config file (e.g., configs/config_carto.json)')
    parser.add_argument('--model', default='saved/Cartosat/SemiCD_Cartosat_semi_5/best_model.pth', type=str,
                        help='Path to the trained .pth model')
    parser.add_argument('--image_t1', default="break/A_png/OUTPUT_1_4.png", type=str,
                        help='Path to the first image (T1)')
    parser.add_argument('--image_t2', default="break/B_png/OUTPUT_1_4.png", type=str,
                        help='Path to the second image (T2)')
    parser.add_argument('--output_dir', default='results', type=str,
                        help='Directory to save output results')
    parser.add_argument('--save', action='store_true', 
                        help='Save prediction results')
    parser.add_argument('--flip', action='store_true',
                        help='Use horizontal flip augmentation during inference')
    parser.add_argument('--save_probs', action='store_true',
                        help='Save raw probability maps as .npy files')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()