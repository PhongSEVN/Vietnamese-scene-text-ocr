import torch
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

from models.attention_ocr import AttentionOCR
from datasets.dataset import VinAIDataset, ResizeNormalize
from utils.label_converter import AttentionLabelConverter


def evaluate():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy trên thiết bị: {device}")

    # Khởi tạo converter và model (giống train.py)
    converter = AttentionLabelConverter(max_len=config.get('max_text_length', 25))
    num_class = len(converter.alphabet)

    model = AttentionOCR(
        img_h=config['img_height'],
        nc=1,
        n_class=num_class,
        nh=config['rnn_hidden_size']
    ).to(device)

    # Load checkpoint: dùng eval_checkpoint trong config, fallback về best
    model_path = config.get('eval_checkpoint', 'checkpoints/attention_ocr_best.pth')
    if not os.path.exists(model_path):
        print(f"[Lỗi] Không tìm thấy checkpoint tại: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Đã load checkpoint: {model_path}")

    # Load dataset (unseen_test hoặc test_image)
    transform = ResizeNormalize(size=(config['img_width'], config['img_height']))
    test_folder = config.get('unseen_test_images', config.get('val_images', 'test_image'))
    dataset = VinAIDataset(
        root=config['data_root'],
        img_folder=test_folder,
        label_folder=config['train_labels'],
        transform=transform,
        converter=converter
    )
    loader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=False,
                        num_workers=config.get('num_workers', 0))
    print(f"Test samples: {len(dataset)} từ '{test_folder}'")

    correct = 0
    total = 0
    char_correct = 0
    char_total = 0

    print("\n--- ĐÁNH GIÁ MÔ HÌNH ATTENTION OCR ---")
    with torch.no_grad():
        for i, (images, target_indices) in enumerate(tqdm(loader)):
            images = images.to(device)
            target_indices = target_indices.to(device)

            # Inference mode: model tự decode, trả về [B, MaxLen]
            pred_indices = model(images)

            pred_texts = converter.decode(pred_indices)
            target_texts = converter.decode(target_indices)

            for pred, gt in zip(pred_texts, target_texts):
                pred = pred.strip()
                gt = gt.strip()

                if i < 3:  # In mẫu 3 batch đầu để kiểm tra
                    print(f"  Dự đoán: '{pred}' | Thực tế: '{gt}'")

                if pred == gt:
                    correct += 1

                for p_char, g_char in zip(pred, gt):
                    if p_char == g_char:
                        char_correct += 1
                char_total += max(len(pred), len(gt))
                total += 1

    word_acc = correct / max(total, 1) * 100
    char_acc = char_correct / max(char_total, 1) * 100
    print(f"\n{'='*40}")
    print(f"Word Accuracy : {word_acc:.2f}%  ({correct}/{total})")
    print(f"Char Accuracy : {char_acc:.2f}%")
    print(f"{'='*40}")


if __name__ == "__main__":
    evaluate()
