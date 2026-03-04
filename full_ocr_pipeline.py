import torch
import cv2
import numpy as np
import yaml
import easyocr
import os
from models.attention_ocr import AttentionOCR
from utils.label_converter import AttentionLabelConverter
from utils.nlp_postprocess import NLPPostProcessor
from datasets.dataset import ResizeNormalize

class FullOCRPipeline:
    def __init__(self, config_path, model_path, enable_nlp=True):
        # 1. Load cấu hình
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. Khởi tạo bộ nhận dạng Attention OCR
        self.converter = AttentionLabelConverter(max_len=self.config.get('max_text_length', 25))
        num_class = len(self.converter.alphabet) # 105
        self.model = AttentionOCR(img_h=self.config['img_height'], nc=1, n_class=num_class, 
                                 nh=self.config['rnn_hidden_size']).to(self.device)
        
        # Load trọng số
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print(f"Cảnh báo: Không tìm thấy file trọng số tại {model_path}")
            
        self.model.eval()
        
        # 3. Khởi tạo bộ phát hiện (Detection) từ EasyOCR
        self.detector = easyocr.Reader(['vi'], gpu=torch.cuda.is_available())
        
        # Tiền xử lý cho Recognition
        self.transform = ResizeNormalize(size=(self.config['img_width'], self.config['img_height']))
        
        # 4. Khởi tạo bộ xử lý NLP Post-Processing
        self.enable_nlp = enable_nlp
        self.nlp_processor = None
        if enable_nlp:
            project_root = os.path.dirname(os.path.abspath(config_path))
            dict_paths = [
                os.path.join(project_root, "data", "dataset", "general_dict.txt"),
                os.path.join(project_root, "data", "dataset", "vn_dictionary.txt"),
                os.path.join(project_root, "data", "dataset", "dict_guided_dictionary.txt"),
            ]
            self.nlp_processor = NLPPostProcessor(
                dictionary_paths=dict_paths,
                enable_spell_correction=True,
                enable_word_segmentation=True,
                enable_confidence_filter=True,
            )

    def recognize_crop(self, crop):
        """
        Dự đoán văn bản sử dụng Attention Decoder.
        """
        img_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Attention model trả về indices [B, MaxLen] khi eval
            indices = self.model(img_tensor)
        
        # Giải mã indices sang văn bản tiếng Việt
        decoded_texts = self.converter.decode(indices)
        text = decoded_texts[0]
        
        # Vì Attention OCR dự đoán trực tiếp sequence, ta có thể lấy "confidence" 
        # (Trong ví dụ này ta giả định 1.0 hoặc tính trung bình softmax nếu cần)
        confidence = 0.95 
        
        return text.strip(), confidence


    def run_ocr(self, image_path):
        # Đọc ảnh gốc
        full_img = cv2.imread(image_path)
        if full_img is None: return "Lỗi: Không thể mở ảnh."

        # BƯỚC 1: DETECTION + RECOGNITION bằng EasyOCR
        # EasyOCR readtext trả về: [(bbox, text, confidence), ...]
        # Ta lấy cả text của EasyOCR để dùng làm fallback
        dt_results = self.detector.readtext(full_img, 
                                            paragraph=False, 
                                            width_ths=0.01, 
                                            add_margin=0.15)
        
        # BƯỚC 2: XỬ LÝ TỪNG VÙNG CHỮ
        final_results = []
        raw_results = []
        display_img = full_img.copy()
        h_orig, w_orig = full_img.shape[:2]
        
        # Ngưỡng confidence tối thiểu của CRNN để được chấp nhận
        CRNN_MIN_CONF = 0.50
        # EasyOCR thắng khi conf của nó cao hơn CRNN ít nhất bao nhiêu
        EASYOCR_MARGIN = 0.15

        for res in dt_results:
            bbox = res[0]
            easyocr_text = res[1]        # Text mà EasyOCR đã nhận dạng
            easyocr_conf = res[2]        # Confidence của EasyOCR (0→1)

            x_min = int(min([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            x_max = int(max([p[0] for p in bbox]))
            y_max = int(max([p[1] for p in bbox]))

            x1, y1 = max(0, x_min), max(0, y_min)
            x2, y2 = min(w_orig, x_max), min(h_orig, y_max)

            crop = full_img[y1:y2, x1:x2]
            if crop.size == 0: continue

            # BƯỚC 3: RECOGNITION - Kết hợp CRNN + EasyOCR
            crnn_text, crnn_conf = self.recognize_crop(crop)

            # Logic chọn kết quả thông minh:
            # Dùng EasyOCR khi:
            #   a) CRNN không đủ tự tin (conf < CRNN_MIN_CONF), hoặc
            #   b) EasyOCR tự tin hơn CRNN rõ rệt (vượt margin)
            use_easyocr = (
                (crnn_conf < CRNN_MIN_CONF or not crnn_text.strip())
                or (easyocr_conf - crnn_conf > EASYOCR_MARGIN and easyocr_text.strip())
            )

            if use_easyocr and easyocr_text.strip():
                chosen_text = easyocr_text
                source = "EasyOCR"
            elif crnn_text.strip():
                chosen_text = crnn_text
                source = "CRNN"
            else:
                chosen_text = easyocr_text
                source = "Fallback"
            
            if chosen_text.strip():
                raw_results.append(f"{chosen_text}")
                
                # BƯỚC 4: NLP POST-PROCESSING
                if self.enable_nlp and self.nlp_processor:
                    processed_text = self.nlp_processor.process(chosen_text)
                else:
                    processed_text = chosen_text
                
                if processed_text.strip():
                    final_results.append(processed_text)
                
                # In chi tiết để debug
                print(f"  [{source}] CRNN='{crnn_text}'(conf={crnn_conf:.2f}) | "
                      f"EasyOCR='{easyocr_text}'(conf={easyocr_conf:.2f}) → '{processed_text}'")
                
                # Vẽ khung
                color = (0, 255, 0) if source == "CRNN" else (0, 165, 255)  # Xanh=CRNN, Cam=EasyOCR
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_img, source, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # HIỂN THỊ
        output_path = "ocr_result_viewer.jpg"
        cv2.imwrite(output_path, display_img)
        print(f"\nĐã lưu ảnh kết quả tại: {output_path}")
        
        try:
            os.startfile(output_path)
        except Exception as e:
            print(f"Không thể tự động mở ảnh: {e}. Bạn hãy mở file {output_path} thủ công nhé.")
        
        # Nối toàn bộ text
        raw_full_text = " ".join(raw_results)
        processed_full_text = " ".join(final_results)
        
        # BƯỚC 5: NLP POST-PROCESSING lần cuối cho toàn văn bản
        if self.enable_nlp and self.nlp_processor:
            processed_full_text = self.nlp_processor.process(processed_full_text)
        
        print(f"\n[RAW]           : {raw_full_text}")
        print(f"[NLP PROCESSED] : {processed_full_text}")
        
        return processed_full_text

if __name__ == "__main__":
    # Đảm bảo đường dẫn .pth chính xác
    model_path = "checkpoints/crnn_epoch_100.pth"
    pipeline = FullOCRPipeline("config.yaml", model_path)
    
    # Đổi tên ảnh bạn muốn test tại đây
    image_to_test = "data/dataset/test_image/im1492.jpg"
    
    print(f"Đang thực hiện OCR cho: {image_to_test}...")
    result_text = pipeline.run_ocr(image_to_test)
    
    print("\n" + "="*30)
    print("VĂN BẢN NHẬN DẠNG ĐƯỢC:")
    print(result_text)
    print("="*30)