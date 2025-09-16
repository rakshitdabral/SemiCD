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


def get_overlay_color(color_name):
    """Return BGR tuple for overlay color"""
    colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)
    }
    return colors.get(color_name.lower(), (0, 0, 255))  # default red


def main():
    args = parse_arguments()

    # Load config
    assert args.config, "Config file path is required"
    config = json.load(open(args.config))
    scales = [1.0, 1.25]  # Multi-scale inference

    # Setup
    num_classes = 2
    palette = get_voc_pallete(num_classes)

    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
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

    # Create output directory
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load input images
    if not os.path.exists(args.image_t1):
        print(f"✗ T1 image not found: {args.image_t1}")
        return
    if not os.path.exists(args.image_t2):
        print(f"✗ T2 image not found: {args.image_t2}")
        return

    try:
        image_A = load_image(args.image_t1, transform)
        image_B = load_image(args.image_t2, transform)
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
    confidence = np.max(output, axis=0)

    # Save results
    if args.save:
        base_name = Path(args.image_t1).stem + "_" + Path(args.image_t2).stem

        # Save colorized prediction
        prediction_colored = colorize_mask(prediction, palette)
        pred_path = os.path.join(args.output_dir, f"{base_name}_prediction.png")
        prediction_colored.save(pred_path)

        # Save binary mask
        binary_mask = prediction * 255
        binary_path = os.path.join(args.output_dir, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary_mask)

        # Save confidence map
        confidence_normalized = (confidence * 255).astype(np.uint8)
        conf_path = os.path.join(args.output_dir, f"{base_name}_confidence.png")
        cv2.imwrite(conf_path, confidence_normalized)

        # Save raw probabilities if requested
        if args.save_probs:
            prob_path = os.path.join(args.output_dir, f"{base_name}_probabilities.npy")
            np.save(prob_path, output)

        # Create overlay visualization + composite image
        try:
            t1_original = cv2.imread(args.image_t1)
            t2_original = cv2.imread(args.image_t2)
            h, w = t2_original.shape[:2]

            mask_resized = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)

            overlay = np.zeros_like(t2_original, dtype=np.uint8)
            overlay[mask_resized == 1] = get_overlay_color(args.overlay_color)

            blended = cv2.addWeighted(t2_original, 0.7, overlay, 0.3, 0)

            # Color mask for visualization
            mask_color = np.zeros_like(t2_original, dtype=np.uint8)
            mask_color[mask_resized == 1] = get_overlay_color(args.overlay_color)

            # Resize T1 to match T2 dimensions
            t1_resized = cv2.resize(t1_original, (w, h))

            # Concatenate horizontally: T1 | T2 | Mask | Overlay
            composite = np.concatenate((t1_resized, t2_original, mask_color, blended), axis=1)

            composite_path = os.path.join(args.output_dir, f"{base_name}_composite.png")
            cv2.imwrite(composite_path, composite)
            print(f"✓ Saved composite visualization: {composite_path}")
        except Exception as e:
            print(f"✗ Error creating composite visualization: {e}")

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

    unique_values, counts = np.unique(prediction, return_counts=True)
    print(f"Prediction value distribution:")
    for val, count in zip(unique_values, counts):
        print(f"  Value {val}: {count:,} pixels ({count/total_pixels*100:.2f}%)")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Change Detection Inference with Overlay')
    parser.add_argument('--config', default="configs/config_carto.json", type=str)
    parser.add_argument('--model', default='saved/Cartosat/SemiCD_Cartosat_semi_5/best_model.pth', type=str)
    parser.add_argument('--image_t1', default="break/A_png/OUTPUT_2_3.png", type=str)
    parser.add_argument('--image_t2', default="break/B_png/OUTPUT_2_3.png", type=str)
    parser.add_argument('--output_dir', default='results', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--save_probs', action='store_true')
    parser.add_argument('--overlay_color', default='red', type=str,
                        help='Overlay color for mask (red, green, blue). Default: red')
    return parser.parse_args()


if __name__ == '__main__':
    main()