"""
train.py - Training Script for Vietnamese Scene Text Recognition

Features:
- Training loop with CTC loss
- Learning rate scheduling
- Checkpoint saving
- Logging and visualization
- Early stopping
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Import from project modules
from model.model import CRNN, CTCLoss
from dataset.dataset import DictGuidedDataset
from config.utils import (
    CharacterSet, 
    ctc_decode, 
    calculate_accuracy,
    collate_fn,
    AverageMeter,
    save_checkpoint,
    normalize_text
)


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    charset: CharacterSet,
                    epoch: int) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: CRNN model
        dataloader: Training data loader
        criterion: CTC loss
        optimizer: Optimizer
        device: Device to use
        charset: Character set for decoding
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    loss_meter = AverageMeter('Loss')
    batch_time = AverageMeter('BatchTime')
    
    start_time = time.time()
    num_batches = len(dataloader)
    
    for batch_idx, (images, labels, label_lengths, texts) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        batch_size = images.size(0)
        
        # Forward pass
        log_probs = model(images)  # [T, B, num_classes]
        
        # Input lengths (all same for fixed width)
        seq_length = log_probs.size(0)
        input_lengths = torch.IntTensor([seq_length] * batch_size)
        
        # Calculate loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Update meters
        loss_meter.update(loss.item(), batch_size)
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
            print(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"Time: {batch_time.avg:.3f}s/batch")
    
    return {'loss': loss_meter.avg}


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             charset: CharacterSet,
             max_samples: int = 500) -> dict:
    """
    Validate model on validation set.
    
    Args:
        model: CRNN model
        dataloader: Validation data loader
        criterion: CTC loss
        device: Device to use
        charset: Character set for decoding
        max_samples: Maximum samples to evaluate
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    all_preds = []
    all_targets = []
    
    sample_count = 0
    
    with torch.no_grad():
        for images, labels, label_lengths, texts in dataloader:
            if sample_count >= max_samples:
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            batch_size = images.size(0)
            
            # Forward pass
            log_probs = model(images)
            seq_length = log_probs.size(0)
            input_lengths = torch.IntTensor([seq_length] * batch_size)
            
            # Calculate loss
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss_meter.update(loss.item(), batch_size)
            
            # Decode predictions
            preds = ctc_decode(log_probs, charset)
            
            all_preds.extend(preds)
            all_targets.extend(texts)
            
            sample_count += batch_size
    
    # Calculate metrics
    metrics = calculate_accuracy(all_preds, all_targets, case_sensitive=False)
    metrics['loss'] = loss_meter.avg
    
    return metrics, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='Train Vietnamese OCR Model')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='data/dataset',
                       help='Path to dataset folder')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='vgg',
                       choices=['vgg', 'resnet'],
                       help='CNN backbone type')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='LSTM hidden size')
    parser.add_argument('--img-height', type=int, default=32,
                       help='Input image height')
    parser.add_argument('--img-width', type=int, default=128,
                       help='Input image width')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("VIETNAMESE SCENE TEXT RECOGNITION - TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize character set
    dict_path = Path(args.data_root) / 'general_dict.txt'
    charset = CharacterSet(str(dict_path) if dict_path.exists() else None)
    print(f"Character set size: {len(charset)}")
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_dataset = DictGuidedDataset(
        data_root=args.data_root,
        split='train',
        charset=charset,
        img_height=args.img_height,
        img_width=args.img_width,
        augment=True
    )
    
    val_dataset = DictGuidedDataset(
        data_root=args.data_root,
        split='test',
        charset=charset,
        img_height=args.img_height,
        img_width=args.img_width,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = CRNN(
        num_classes=len(charset),
        img_height=args.img_height,
        img_width=args.img_width,
        hidden_size=args.hidden_size,
        backbone=args.backbone
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    criterion = CTCLoss(blank=0)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Resume from checkpoint
    start_epoch = 1
    best_accuracy = 0.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch-1}, best accuracy: {best_accuracy:.2f}%")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-"*40)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, charset, epoch
        )
        
        # Validate
        val_metrics, preds, targets = validate(
            model, val_loader, criterion, device, charset
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch summary
        print(f"\n  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Word Accuracy: {val_metrics['word_accuracy']:.2f}%")
        print(f"  Char Accuracy: {val_metrics['char_accuracy']:.2f}%")
        print(f"  CER: {val_metrics['cer']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Print sample predictions
        print("\n  Sample Predictions:")
        for i in range(min(5, len(preds))):
            pred = preds[i]
            target = targets[i]
            match = "✓" if pred.lower() == target.lower() else "✗"
            print(f"    {match} GT: '{target}' | Pred: '{pred}'")
        
        # Save checkpoint
        is_best = val_metrics['word_accuracy'] > best_accuracy
        if is_best:
            best_accuracy = val_metrics['word_accuracy']
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['word_accuracy'],
            'charset_size': len(charset),
            'args': vars(args)
        }
        
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint(checkpoint, str(checkpoint_path), is_best)
        
        if is_best:
            print(f"\n  *** New best accuracy: {best_accuracy:.2f}% ***")
    
    # Final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Word Accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'charset_size': len(charset),
        'args': vars(args)
    }, final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
