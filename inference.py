import torch
import cv2
import yaml
import os
import numpy as np

from models.attention_ocr import AttentionOCR
from utils.label_converter import AttentionLabelConverter
from utils.nlp_postprocess import NLPPostProcessor
from datasets.dataset import ResizeNormalize


class OCRInference:
    def __init__(self, config_path, model_path=None, enable_nlp=True):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Converter và model
        self.converter = AttentionLabelConverter(max_len=self.config.get('max_text_length', 25))
        num_class = len(self.converter.alphabet)

        self.model = AttentionOCR(
            img_h=self.config['img_height'],
            nc=1,
            n_class=num_class,
            nh=self.config['rnn_hidden_size']
        ).to(self.device)

        # Load checkpoint (ưu tiên argument, rồi config, rồi default)
        if model_path is None:
            model_path = self.config.get('eval_checkpoint', 'checkpoints/attention_ocr_best.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {model_path}")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[Inference] Loaded model: {model_path} | Device: {self.device}")

        self.transform = ResizeNormalize(size=(self.config['img_width'], self.config['img_height']))

        # NLP Post-Processing
        self.enable_nlp = enable_nlp
        self.nlp_processor = None
        if enable_nlp:
            project_root = os.path.dirname(os.path.abspath(config_path))
            dict_paths = [
                os.path.join(project_root, "data", "dataset", "general_dict.txt"),
                os.path.join(project_root, "data", "dataset", "vn_dictionary.txt"),
                os.path.join(project_root, "data", "dataset", "dict_guided_dictionary.txt"),
            ]
            # Chỉ load nếu ít nhất 1 file từ điển tồn tại
            available_dicts = [p for p in dict_paths if os.path.exists(p)]
            if available_dicts:
                self.nlp_processor = NLPPostProcessor(
                    dictionary_paths=available_dicts,
                    enable_spell_correction=True,
                    enable_word_segmentation=True,
                    enable_confidence_filter=True,
                )
            else:
                print("[Inference] Không tìm thấy từ điển, bỏ qua NLP post-processing.")

    def predict(self, image_path):
        """Dự đoán từ đường dẫn ảnh (ảnh đã được cắt chứa chữ)."""
        img = cv2.imread(image_path)
        if img is None:
            return "Không thể đọc ảnh"

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1, 1, H, W]

        with torch.no_grad():
            # Inference mode (không truyền targets) → [1, MaxLen] indices
            pred_indices = self.model(img_tensor)

        raw_text = self.converter.decode(pred_indices)[0].strip()

        if self.enable_nlp and self.nlp_processor:
            raw_text = self.nlp_processor.process(raw_text)

        return raw_text

    def predict_crop(self, crop_bgr):
        """Dự đoán từ numpy array (BGR crop) — dùng khi tích hợp trong pipeline."""
        img_tensor = self.transform(crop_bgr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_indices = self.model(img_tensor)

        raw_text = self.converter.decode(pred_indices)[0].strip()

        if self.enable_nlp and self.nlp_processor:
            raw_text = self.nlp_processor.process(raw_text)

        return raw_text


if __name__ == "__main__":
    ocr = OCRInference("config.yaml")
    # Ví dụ:
    # print(ocr.predict("path/to/cropped_word.jpg"))
