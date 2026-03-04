import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

from models.attention_ocr import AttentionOCR
from datasets.dataset import VinAIDataset, ResizeNormalize
from utils.label_converter import AttentionLabelConverter


def compute_accuracy(pred_indices, target_indices, converter):
    """Tính Character Accuracy và Word Accuracy cho Attention model."""
    pred_texts = converter.decode(pred_indices)
    # Target indices ở đây là output của Dataset (đã có tone marker encoding)
    # Tuy nhiên decode cần tensor chuẩn, ta giả định target_indices là chuỗi indices
    target_texts = converter.decode(target_indices)
    
    char_correct = 0
    char_total = 0
    word_correct = 0
    
    for pred, gt in zip(pred_texts, target_texts):
        pred = pred.strip()
        gt = gt.strip()
        
        if pred == gt:
            word_correct += 1
        
        for p_char, g_char in zip(pred, gt):
            if p_char == g_char:
                char_correct += 1
        char_total += max(len(pred), len(gt))
    
    char_acc = char_correct / max(char_total, 1)
    word_acc = word_correct / max(len(target_indices), 1)
    return char_acc, word_acc


def train():
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device} (Attention OCR)")

    # Khởi tạo converter (105 ký tự)
    converter = AttentionLabelConverter(max_len=config.get('max_text_length', 25))
    num_class = len(converter.alphabet)

    transform = ResizeNormalize(size=(config['img_width'], config['img_height']))

    # Khởi tạo dataset với converter để nhãn được tiền xử lý sang indices
    train_dataset = VinAIDataset(
        root=config['data_root'],
        img_folder=config['train_images'],
        label_folder=config['train_labels'],
        transform=transform,
        converter=converter
    )

    val_dataset = VinAIDataset(
        root=config['data_root'],
        img_folder=config.get('val_images', 'test_image'),
        label_folder=config['train_labels'],
        transform=transform,
        converter=converter
    )

    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Khởi tạo model Attention OCR
    model = AttentionOCR(
        img_h=config['img_height'], 
        nc=1, 
        n_class=num_class,
        nh=config['rnn_hidden_size']
    ).to(device)
    
    # Load checkpoint nếu có
    start_epoch = 0
    checkpoint_to_resume = config.get('resume_checkpoint', None)
    if checkpoint_to_resume and os.path.exists(checkpoint_to_resume):
        model.load_state_dict(torch.load(checkpoint_to_resume, map_location=device, weights_only=True))
        print(f"Đã load checkpoint từ: {checkpoint_to_resume}")

    # Loss: CrossEntropy (NLLLoss do model trả về log_softmax)
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )

    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])

    best_word_acc = 0.0

    for epoch in range(start_epoch, config['num_epochs']):
        # ============================
        # TRAINING PHASE
        # ============================
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for images, target_indices in pbar:
            images = images.to(device)
            target_indices = target_indices.to(device) # [B, T]
            
            # Forward Attention OCR (với teacher forcing)
            # outputs: [T, B, n_class]
            outputs = model(images, target_indices)
            
            # Tính loss
            # outputs: [T, B, C] -> reshaped [T*B, C]
            # targets: [B, T] -> transposed [T, B] -> reshaped [T*B]
            target_transposed = target_indices.transpose(0, 1).contiguous().view(-1)
            loss = criterion(outputs.view(-1, num_class), target_transposed)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        
        # ============================
        # VALIDATION PHASE
        # ============================
        model.eval()
        val_loss = 0
        total_word_acc = 0
        
        with torch.no_grad():
            for images, target_indices in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]  "):
                images = images.to(device)
                target_indices = target_indices.to(device)
                
                # Validation mode: model tự decode (không targets) -> [B, MaxLen]
                # Nhưng để tính Loss, ta cần forward kiểu train (có targets)
                # Hoặc chỉ tính Accuracy
                
                # Tính Loss (teacher forcing off hoặc dùng targets để tính loss)
                outputs_for_loss = model(images, target_indices)
                target_transposed = target_indices.transpose(0, 1).contiguous().view(-1)
                loss = criterion(outputs_for_loss.view(-1, num_class), target_transposed)
                val_loss += loss.item()
                
                # Tính Accuracy (decode độc lập)
                pred_indices = model(images) # [B, MaxLen]
                _, word_acc = compute_accuracy(pred_indices, target_indices, converter)
                total_word_acc += word_acc
        
        avg_val_loss = val_loss / len(val_loader)
        avg_word_acc = total_word_acc / len(val_loader)
        
        print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Word Acc: {avg_word_acc:.2%}")
        
        scheduler.step(avg_val_loss)
        
        # Lưu checkpoint
        if avg_word_acc > best_word_acc:
            best_word_acc = avg_word_acc
            torch.save(model.state_dict(), f"{config['checkpoint_dir']}/attention_ocr_best.pth")
        
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save(model.state_dict(), f"{config['checkpoint_dir']}/attention_ocr_epoch_{epoch+1}.pth")

    print(f"\nTraining hoàn tất! Best Word Accuracy: {best_word_acc:.2%}")

if __name__ == "__main__":
    train()
