import torch
from torch.utils.data import DataLoader
import yaml
from models.crnn import CRNN
from datasets.dataset import VinAIDataset, ResizeNormalize
from utils.label_converter import CTCLabelConverter
from tqdm import tqdm

def evaluate():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    converter = CTCLabelConverter(config['char_list'])
    num_class = len(config['char_list']) + 1

    # Load Model
    model = CRNN(img_h=config['img_height'], nc=1, n_class=num_class, nh=config['rnn_hidden_size']).to(device)
    # Giả sử load checkpoint mới nhất
    model_path = "checkpoints/crnn_epoch_10.pth"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    # Load Data
    transform = ResizeNormalize(size=(config['img_width'], config['img_height']))
    dataset = VinAIDataset(root=config['data_root'], img_folder='test_image', label_folder='labels', transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    
    print("\n--- ĐÁNH GIÁ MÔ HÌNH ---")
    print("In mẫu 100 kết quả đầu tiên để phân tích lỗi:")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(device)
            preds = model(images)
            
            pred_texts = converter.decode_greedy(preds)
            
            pred = pred_texts[0].strip()
            target = labels[0].strip()
            
            if i < 100:
                print(f"  -> {i+1}. Dự đoán: '{pred}' | Thực tế: '{target}'")
            
            if pred == target:
                correct += 1
            total += 1

    print(f"Độ chính xác cấp độ từ (Word Accuracy): {correct / total * 100:.2f}%")

if __name__ == "__main__":
    evaluate()
