import sys
import os
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.transformer_ocr import TransformerOCR
from datasets.transformer_dataset import Vocab, TransformerOCRDataset, collate_fn, VIETNAMESE_CHARS


def cnn_seq_len(w):
    w = (w - 2) // 2 + 1
    w = (w - 2) // 2 + 1
    return w * 2


def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else config.get('eval_checkpoint', 'checkpoints/transformer_ocr_best.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint không tìm thấy: {checkpoint_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict    = ckpt['model_state_dict']
        saved_epoch   = ckpt.get('epoch', '?')
        saved_acc     = ckpt.get('best_word_acc', None)
        vocab_chars   = ckpt.get('vocab_chars', VIETNAMESE_CHARS)
    else:
        state_dict  = ckpt
        saved_epoch = '?'
        saved_acc   = None
        vocab_chars = VIETNAMESE_CHARS

    if saved_acc is not None:
        print(f"Saved at epoch {saved_epoch} | Word Acc khi train: {saved_acc:.2%}")

    vocab = Vocab(vocab_chars)
    vocab_size = len(vocab)

    enc_layers = sum(1 for k in state_dict if k.startswith('transformer.transformer.encoder.layers.')
                     and k.endswith('.self_attn.in_proj_weight'))
    dec_layers = sum(1 for k in state_dict if k.startswith('transformer.transformer.decoder.layers.')
                     and k.endswith('.self_attn.in_proj_weight'))
    ff_shape   = state_dict['transformer.transformer.encoder.layers.0.linear1.weight'].shape
    dim_ff     = ff_shape[0]
    print(f"Detected: encoder={enc_layers} layers, decoder={dec_layers} layers, dim_feedforward={dim_ff}")

    model = TransformerOCR(
        vocab_size=vocab_size, d_model=256, nhead=8,
        num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
        dim_feedforward=dim_ff, max_seq_length=128,
        pos_dropout=0.1, trans_dropout=0.0,
        cnn_pretrained=False, cnn_dropout=0.0,
        ss=[(2,2),(2,2),(2,1),(2,1),(1,1)],
        ks=[(2,2),(2,2),(2,1),(2,1),(1,1)],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_dataset = TransformerOCRDataset(
        root=config['data_root'],
        img_folder=config.get('val_images', 'test_image'),
        label_folder=config['train_labels'],
        vocab=vocab,
        img_height=config.get('img_height', 32),
        augment=False
    )

    batch_size = config.get('batch_size', 32)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    correct_words = 0
    total_words   = 0
    correct_chars = 0
    total_chars   = 0
    wrong_samples = []

    print(f"\nEvaluating {len(val_dataset)} samples...")
    with torch.no_grad():
        for img_batch, label_batch, img_widths in tqdm(val_loader):
            img_batch   = img_batch.to(device)
            label_batch = label_batch.to(device)

            src_len = cnn_seq_len(img_batch.shape[3])
            src_key_padding_mask = torch.zeros(img_batch.size(0), src_len, dtype=torch.bool, device=device)
            for i, w in enumerate(img_widths):
                feat_valid = cnn_seq_len(w.item())
                if feat_valid < src_len:
                    src_key_padding_mask[i, feat_valid:] = True

            pred_indices, _ = model.greedy_decode(
                img_batch,
                max_len=config.get('max_text_length', 25),
                src_key_padding_mask=src_key_padding_mask
            )
            pred_texts   = vocab.batch_decode(pred_indices.tolist())
            target_texts = vocab.batch_decode(label_batch.tolist())

            for pred, gt in zip(pred_texts, target_texts):
                pred = pred.strip()
                gt   = gt.strip()

                if pred == gt:
                    correct_words += 1
                elif len(wrong_samples) < 10:
                    wrong_samples.append((gt, pred))

                correct_chars += sum(p == g for p, g in zip(pred, gt))
                total_chars   += max(len(gt), 1)
                total_words   += 1

    word_acc = correct_words / max(total_words, 1)
    char_acc = correct_chars / max(total_chars, 1)

    print(f"\n{'='*50}")
    print(f"  Word Accuracy : {word_acc:.4%}  ({correct_words}/{total_words})")
    print(f"  Char Accuracy : {char_acc:.4%}  ({correct_chars}/{total_chars})")
    print(f"{'='*50}")

    if wrong_samples:
        print(f"\nMột số mẫu sai (GT → Pred):")
        for gt, pred in wrong_samples:
            print(f"  GT  : {gt}")
            print(f"  Pred: {pred}")
            print()


if __name__ == '__main__':
    main()
