import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

from models.transformer_ocr import TransformerOCR
from datasets.transformer_dataset import Vocab, TransformerOCRDataset, collate_fn, VIETNAMESE_CHARS


def train():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Architecture: VGG19-BN + Transformer")

    vocab = Vocab(VIETNAMESE_CHARS)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    train_dataset = TransformerOCRDataset(
        root=config['data_root'], img_folder=config['train_images'],
        label_folder=config['train_labels'], vocab=vocab,
        img_height=config.get('img_height', 32), augment=True
    )
    val_dataset = TransformerOCRDataset(
        root=config['data_root'], img_folder=config.get('val_images', 'test_image'),
        label_folder=config['train_labels'], vocab=vocab,
        img_height=config.get('img_height', 32), augment=False
    )

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=config.get('num_workers', 0), collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=config.get('num_workers', 0), collate_fn=collate_fn)

    model = TransformerOCR(
        vocab_size=vocab_size, d_model=256, nhead=8,
        num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, max_seq_length=128,
        pos_dropout=0.1, trans_dropout=0.1,
        cnn_pretrained=True, cnn_dropout=0.5,
        ss=[(2,2),(2,2),(2,1),(2,1),(1,1)],
        ks=[(2,2),(2,2),(2,1),(2,1),(1,1)],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    resume_path = config.get('resume_checkpoint', None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed from: {resume_path}")

    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.PAD)
    lr = config.get('learning_rate', 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    num_epochs = config.get('num_epochs', 50)
    best_word_acc = 0.0
    save_interval = config.get('save_interval', 5)

    for epoch in range(start_epoch, num_epochs):
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for img_batch, label_batch in pbar:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            tgt_input = label_batch[:, :-1].transpose(0, 1)
            tgt_output = label_batch[:, 1:]
            tgt_key_padding_mask = (label_batch[:, :-1] == Vocab.PAD)

            output = model(img_batch, tgt_input, tgt_key_padding_mask)
            loss = criterion(output.contiguous().view(-1, vocab_size), tgt_output.contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct_words = 0
        total_words = 0

        with torch.no_grad():
            for img_batch, label_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]  "):
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)

                tgt_input = label_batch[:, :-1].transpose(0, 1)
                tgt_output = label_batch[:, 1:]
                tgt_key_padding_mask = (label_batch[:, :-1] == Vocab.PAD)

                output = model(img_batch, tgt_input, tgt_key_padding_mask)
                loss = criterion(output.contiguous().view(-1, vocab_size), tgt_output.contiguous().view(-1))
                val_loss += loss.item()

                pred_indices, _ = model.greedy_decode(img_batch, max_len=config.get('max_text_length', 25))
                pred_texts = vocab.batch_decode(pred_indices.tolist())
                target_texts = vocab.batch_decode(label_batch.tolist())
                for pred, gt in zip(pred_texts, target_texts):
                    if pred.strip() == gt.strip():
                        correct_words += 1
                    total_words += 1

        avg_val_loss = val_loss / len(val_loader)
        word_acc = correct_words / max(total_words, 1)

        print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Word Acc: {word_acc:.2%} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(avg_val_loss)

        if word_acc > best_word_acc:
            best_word_acc = word_acc
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'best_word_acc': best_word_acc, 'vocab_chars': vocab.chars},
                       os.path.join(checkpoint_dir, 'transformer_ocr_best.pth'))
            print(f"  ★ Best model saved (Word Acc: {best_word_acc:.2%})")

        if (epoch + 1) % save_interval == 0:
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'vocab_chars': vocab.chars},
                       os.path.join(checkpoint_dir, f'transformer_ocr_epoch_{epoch+1}.pth'))

    print(f"\nTraining done! Best Word Accuracy: {best_word_acc:.2%}")


if __name__ == "__main__":
    train()
