"""
test.py - Testing/Evaluation Script for Vietnamese Scene Text Recognition

Features:
- Load trained checkpoint
- Run inference on test set
- Calculate comprehensive metrics
- Print detailed results and examples
- Support for single image inference
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2

# Import from project modules
from model.model import CRNN, CTCLoss
from dataset.dataset import DictGuidedDataset
from config.utils import (
    CharacterSet,
    ctc_decode,
    calculate_accuracy,
    collate_fn,
    AverageMeter, 
    levenshtein_distance
)


def evaluate_dataset(model: nn.Module,
                    dataloader: DataLoader,
                    device: torch.device,
                    charset: CharacterSet,
                    verbose: bool = True) -> dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained CRNN model
        dataloader: Test data loader
        device: Device to use
        charset: Character set for decoding
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_correct = []
    
    total_time = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels, label_lengths, texts) in enumerate(dataloader):
            batch_start = time.time()
            
            images = images.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            log_probs = model(images)
            
            # Decode predictions
            preds = ctc_decode(log_probs, charset)
            
            # Store results
            all_preds.extend(preds)
            all_targets.extend(texts)
            
            for pred, target in zip(preds, texts):
                all_correct.append(pred.lower() == target.lower())
            
            total_time += time.time() - batch_start
            num_samples += batch_size
            
            if verbose and (batch_idx + 1) % 20 == 0:
                acc = sum(all_correct) / len(all_correct) * 100
                print(f"  Evaluated {num_samples} samples, Current Accuracy: {acc:.2f}%")
    
    # Calculate metrics
    metrics = calculate_accuracy(all_preds, all_targets, case_sensitive=False)
    metrics['total_samples'] = num_samples
    metrics['inference_time'] = total_time
    metrics['avg_time_per_sample'] = total_time / max(num_samples, 1)
    
    return metrics, all_preds, all_targets


def inference_single_image(model: nn.Module,
                          image_path: str,
                          charset: CharacterSet,
                          device: torch.device,
                          img_height: int = 32,
                          img_width: int = 128) -> str:
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        charset: Character set
        device: Device to use
        img_height: Target height
        img_width: Target width
        
    Returns:
        Predicted text
    """
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Resize
    image = cv2.resize(image, (img_width, img_height))
    
    # Convert to tensor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float().unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        log_probs = model(image)
        pred = ctc_decode(log_probs, charset)[0]
    
    return pred


def print_detailed_results(preds: list, 
                           targets: list,
                           max_examples: int = 50):
    """
    Print detailed prediction results.
    
    Args:
        preds: List of predictions
        targets: List of ground truth texts
        max_examples: Maximum examples to print
    """
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    # Separate correct and incorrect
    correct_examples = []
    incorrect_examples = []
    
    for pred, target in zip(preds, targets):
        is_correct = pred.lower() == target.lower()
        edit_dist = levenshtein_distance(pred.lower(), target.lower())
        
        example = {
            'pred': pred,
            'target': target,
            'edit_dist': edit_dist,
            'correct': is_correct
        }
        
        if is_correct:
            correct_examples.append(example)
        else:
            incorrect_examples.append(example)
    
    # Sort incorrect by edit distance
    incorrect_examples.sort(key=lambda x: x['edit_dist'])
    
    # Print some correct examples
    print(f"\n✓ CORRECT PREDICTIONS ({len(correct_examples)}/{len(preds)}):")
    print("-"*50)
    for i, ex in enumerate(correct_examples[:10]):
        print(f"  {i+1}. '{ex['target']}'")
    if len(correct_examples) > 10:
        print(f"  ... and {len(correct_examples)-10} more")
    
    # Print incorrect examples
    print(f"\n✗ INCORRECT PREDICTIONS ({len(incorrect_examples)}/{len(preds)}):")
    print("-"*50)
    
    # Close predictions (low edit distance)
    close_incorrect = [ex for ex in incorrect_examples if ex['edit_dist'] <= 3]
    if close_incorrect:
        print("\n  Near-correct (edit distance ≤ 3):")
        for i, ex in enumerate(close_incorrect[:15]):
            print(f"    GT: '{ex['target']}' | Pred: '{ex['pred']}' (dist={ex['edit_dist']})")
    
    # Far predictions
    far_incorrect = [ex for ex in incorrect_examples if ex['edit_dist'] > 3]
    if far_incorrect:
        print(f"\n  Further off (edit distance > 3): {len(far_incorrect)} examples")
        for i, ex in enumerate(far_incorrect[:10]):
            print(f"    GT: '{ex['target']}' | Pred: '{ex['pred']}' (dist={ex['edit_dist']})")
    
    # Statistics by text length
    print("\n" + "="*70)
    print("ACCURACY BY TEXT LENGTH")
    print("="*70)
    
    length_stats = {}
    for pred, target in zip(preds, targets):
        length = len(target)
        bucket = f"{(length-1)//5*5+1}-{(length-1)//5*5+5}"
        
        if bucket not in length_stats:
            length_stats[bucket] = {'correct': 0, 'total': 0}
        
        length_stats[bucket]['total'] += 1
        if pred.lower() == target.lower():
            length_stats[bucket]['correct'] += 1
    
    for bucket in sorted(length_stats.keys()):
        stats = length_stats[bucket]
        acc = stats['correct'] / stats['total'] * 100
        print(f"  Length {bucket}: {acc:.1f}% ({stats['correct']}/{stats['total']})")


def main():
    parser = argparse.ArgumentParser(description='Test Vietnamese OCR Model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='data/dataset',
                       help='Path to dataset folder')
    parser.add_argument('--split', type=str, default='test',
                       choices=['test', 'unseen', 'train'],
                       help='Dataset split to evaluate')
    
    # Model arguments (should match training)
    parser.add_argument('--backbone', type=str, default='vgg',
                       choices=['vgg', 'resnet'],
                       help='CNN backbone type')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='LSTM hidden size')
    parser.add_argument('--img-height', type=int, default=32,
                       help='Input image height')
    parser.add_argument('--img-width', type=int, default=128,
                       help='Input image width')
    
    # Other arguments
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed results')
    parser.add_argument('--image', type=str, default=None,
                       help='Single image path for inference')
    
    args = parser.parse_args()
    
    # Print header
    print("="*70)
    print("VIETNAMESE SCENE TEXT RECOGNITION - EVALUATION")
    print("="*70)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get config from checkpoint if available
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        args.backbone = ckpt_args.get('backbone', args.backbone)
        args.hidden_size = ckpt_args.get('hidden_size', args.hidden_size)
        args.img_height = ckpt_args.get('img_height', args.img_height)
        args.img_width = ckpt_args.get('img_width', args.img_width)
    
    # Initialize character set
    dict_path = Path(args.data_root) / 'general_dict.txt'
    charset = CharacterSet(str(dict_path) if dict_path.exists() else None)
    
    # Override charset size from checkpoint if available
    charset_size = checkpoint.get('charset_size', len(charset))
    print(f"Character set size: {charset_size}")
    
    # Create model
    model = CRNN(
        num_classes=charset_size,
        img_height=args.img_height,
        img_width=args.img_width,
        hidden_size=args.hidden_size,
        backbone=args.backbone
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'val_accuracy' in checkpoint:
        print(f"  Checkpoint accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    # Single image inference
    if args.image:
        print(f"\n{'='*50}")
        print(f"SINGLE IMAGE INFERENCE")
        print(f"{'='*50}")
        print(f"Image: {args.image}")
        
        try:
            pred = inference_single_image(
                model, args.image, charset, device,
                args.img_height, args.img_width
            )
            print(f"Prediction: {pred}")
        except Exception as e:
            print(f"Error: {e}")
        
        return
    
    # Dataset evaluation
    print(f"\nEvaluating on {args.split} set...")
    
    # Create dataset
    test_dataset = DictGuidedDataset(
        data_root=args.data_root,
        split=args.split,
        charset=charset,
        img_height=args.img_height,
        img_width=args.img_width,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\nRunning evaluation...")
    metrics, preds, targets = evaluate_dataset(
        model, test_loader, device, charset, verbose=True
    )
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"  Split: {args.split}")
    print(f"  Total Samples: {metrics['total_samples']}")
    print(f"  Word Accuracy: {metrics['word_accuracy']:.2f}%")
    print(f"  Character Accuracy: {metrics['char_accuracy']:.2f}%")
    print(f"  Character Error Rate (CER): {metrics['cer']:.4f}")
    print(f"  Total Inference Time: {metrics['inference_time']:.2f}s")
    print(f"  Avg Time per Sample: {metrics['avg_time_per_sample']*1000:.2f}ms")
    print("="*70)
    
    # Print sample predictions
    print("\nSample Predictions:")
    print("-"*50)
    np.random.seed(42)
    indices = np.random.choice(len(preds), min(20, len(preds)), replace=False)
    
    for i in indices:
        pred = preds[i]
        target = targets[i]
        match = "✓" if pred.lower() == target.lower() else "✗"
        print(f"  {match} GT: '{target}' | Pred: '{pred}'")
    
    # Detailed results
    if args.detailed:
        print_detailed_results(preds, targets)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
