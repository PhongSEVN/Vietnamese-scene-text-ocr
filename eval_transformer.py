import torch
import yaml
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.transformer_ocr import TransformerOCR
from datasets.transformer_dataset import Vocab, TransformerOCRDataset, collate_fn, VIETNAMESE_CHARS


def evaluate():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocab(VIETNAMESE_CHARS)
    vocab_size = len(vocab)

    model = TransformerOCR(
        vocab_size=vocab_size, d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=1024, max_seq_length=128,
        cnn_pretrained=False, cnn_dropout=0.0,
        ss=[(2,2),(2,2),(2,1),(2,1),(1,1)],
        ks=[(2,2),(2,2),(2,1),(2,1),(1,1)],
    ).to(device)

    checkpoint_path = config.get('eval_checkpoint', 'checkpoints/transformer_ocr_best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found at '{checkpoint_path}'")
        return

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded: {checkpoint_path}")

    test_folder = config.get('unseen_test_images', config.get('val_images', 'test_image'))
    test_dataset = TransformerOCRDataset(
        root=config['data_root'], img_folder=test_folder,
        label_folder=config['train_labels'], vocab=vocab,
        img_height=config.get('img_height', 32), augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 16),
                             shuffle=False, num_workers=config.get('num_workers', 0), collate_fn=collate_fn)

    correct_words = 0
    total_words = 0
    char_correct = 0
    char_total = 0

    with torch.no_grad():
        for i, (img_batch, label_batch, _) in enumerate(tqdm(test_loader)):
            img_batch = img_batch.to(device)
            pred_indices, _ = model.greedy_decode(img_batch, max_len=config.get('max_text_length', 25))

            pred_texts = vocab.batch_decode(pred_indices.tolist())
            target_texts = vocab.batch_decode(label_batch.tolist())

            for pred, gt in zip(pred_texts, target_texts):
                pred, gt = pred.strip(), gt.strip()
                if i < 3:
                    print(f"  Pred: '{pred}' | GT: '{gt}'")
                if pred == gt:
                    correct_words += 1
                for p, g in zip(pred, gt):
                    if p == g:
                        char_correct += 1
                char_total += len(gt)
                total_words += 1

    print(f"\n{'='*40}")
    print(f"Word Accuracy : {correct_words/max(total_words,1)*100:.2f}%  ({correct_words}/{total_words})")
    print(f"Char Accuracy : {char_correct/max(char_total,1)*100:.2f}%")
    print(f"{'='*40}")


if __name__ == "__main__":
    evaluate()
