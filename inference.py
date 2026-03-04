import torch
import cv2
import yaml
import os
import numpy as np
from models.crnn import CRNN
from utils.label_converter import CTCLabelConverter
from utils.nlp_postprocess import NLPPostProcessor
from datasets.dataset import ResizeNormalize

class OCRInference:
    def __init__(self, config_path, model_path, enable_nlp=True):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.converter = CTCLabelConverter(self.config['char_list'])
        num_class = len(self.config['char_list']) + 1
        
        self.model = CRNN(img_h=self.config['img_height'], nc=1, n_class=num_class, 
                         nh=self.config['rnn_hidden_size']).to(self.device)
        
        # Load weight
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = ResizeNormalize(size=(self.config['img_width'], self.config['img_height']))

        # Khởi tạo NLP Post-Processing
        self.enable_nlp = enable_nlp
        self.nlp_processor = None
        if enable_nlp:
            project_root = os.path.dirname(os.path.abspath(config_path))
            dict_paths = [
                os.path.join(project_root, "data", "dataset", "general_dict.txt"),
                os.path.join(project_root, "data", "dataset", "vn_dictionary.txt"),
                os.path.join(project_root, "data", "dataset", "dict_guided_dictionary.txt"),  # 91K từ dict-guided CVPR-2021
            ]
            self.nlp_processor = NLPPostProcessor(
                dictionary_paths=dict_paths,
                enable_spell_correction=True,
                enable_word_segmentation=True,
                enable_confidence_filter=True,
            )

    def predict(self, image_path):
        """Dự đoán từ ảnh đơn lẻ (đã được cắt chứa chữ)."""
        img = cv2.imread(image_path)
        if img is None: return "Không thể đọc ảnh"
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device) # Add batch dim
        
        with torch.no_grad():
            preds = self.model(img_tensor)
            
        result = self.converter.decode_greedy(preds)
        raw_text = result[0]
        
        # NLP Post-Processing: chuẩn hóa kết quả
        if self.enable_nlp and self.nlp_processor:
            processed_text = self.nlp_processor.process(raw_text)
            return processed_text
        
        return raw_text

if __name__ == "__main__":
    ocr = OCRInference("config.yaml", "checkpoints/crnn_epoch_10.pth")
    # print(ocr.predict("path/to/cropped_word.jpg"))

