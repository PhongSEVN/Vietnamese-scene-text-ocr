"""
Full OCR Pipeline: EasyOCR (Detection) + VietOCR (Recognition)
==============================================================
Pipeline:
  1. Ảnh đầu vào (full image)
  2. EasyOCR phát hiện vùng chữ (CHỈ detection, bỏ qua kết quả recognition)
  3. Crop từng vùng chữ theo bounding box
  4. VietOCR nhận dạng text tiếng Việt cho từng crop
  5. NLP Post-Processing (chuẩn hóa + sửa lỗi chính tả)
  6. Trả về danh sách kết quả sắp xếp trên-xuống, trái-phải
"""

import cv2
import numpy as np
import os
import torch
import easyocr
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from utils.nlp_postprocess import NLPPostProcessor


class FullOCRPipeline:
    """
    Pipeline OCR hoàn chỉnh cho tiếng Việt.
    - Detection: EasyOCR (chỉ dùng phần detect, KHÔNG dùng phần recognition)
    - Recognition: VietOCR (pretrained Transformer OCR chuyên tiếng Việt)
    - Post-Processing: NLP chuẩn hóa + sửa lỗi chính tả
    """

    def __init__(self, vietocr_model='vgg_transformer', enable_nlp=True):
        """
        Khởi tạo pipeline.

        Args:
            vietocr_model: Tên model VietOCR pretrained.
                           Lựa chọn: 'vgg_transformer', 'vgg_seq2seq', 'resnet_transformer'
                           - vgg_transformer: Chính xác nhất (88%), tốc độ trung bình
                           - vgg_seq2seq: Nhanh nhất (12ms), chính xác 87%
            enable_nlp: Bật/tắt NLP Post-Processing
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Thiết bị: {self.device}")

        # =============================================
        # 1. KHỞI TẠO EASYOCR (CHỈ DETECTION)
        # =============================================
        print("[1/3] Đang khởi tạo EasyOCR Detector...")
        self.detector = easyocr.Reader(
            ['vi'],
            gpu=torch.cuda.is_available(),
            recognizer=False  # CHỈ dùng detector, KHÔNG load recognizer -> tiết kiệm RAM
        )
        print("  ✓ EasyOCR Detector sẵn sàng (chỉ detection, không recognition)")

        # =============================================
        # 2. KHỞI TẠO VIETOCR (RECOGNITION)
        # =============================================
        print(f"[2/3] Đang khởi tạo VietOCR ({vietocr_model})...")
        config = Cfg.load_config_from_name(vietocr_model)
        config['device'] = self.device
        config['cnn']['pretrained'] = True
        config['predictor']['beamsearch'] = False  # Greedy decoding (nhanh hơn)

        self.recognizer = Predictor(config)
        print(f"VietOCR ({vietocr_model}) sẵn sàng")
        print(f"    - Backbone: {vietocr_model.split('_')[0].upper()}")
        print(f"    - Decoder: {vietocr_model.split('_')[1].upper()}")
        print(f"    - Pretrained trên 10 triệu ảnh tiếng Việt")

        # =============================================
        # 3. KHỞI TẠO NLP POST-PROCESSING
        # =============================================
        self.enable_nlp = enable_nlp
        self.nlp_processor = None
        if enable_nlp:
            print("[3/3] Đang khởi tạo NLP Post-Processor...")
            project_root = os.path.dirname(os.path.abspath(__file__))
            dict_paths = [
                os.path.join(project_root, "data", "dataset", "general_dict.txt"),
                os.path.join(project_root, "data", "dataset", "vn_dictionary.txt"),
                os.path.join(project_root, "data", "dataset", "dict_guided_dictionary.txt"),
            ]
            existing_paths = [p for p in dict_paths if os.path.exists(p)]
            if existing_paths:
                self.nlp_processor = NLPPostProcessor(
                    dictionary_paths=existing_paths,
                    enable_spell_correction=True,
                    enable_word_segmentation=True,
                    enable_confidence_filter=True,
                )
            else:
                print("  ⚠ Không tìm thấy từ điển, bỏ qua NLP Post-Processing")
                self.enable_nlp = False
        else:
            print("[3/3] NLP Post-Processing: TẮT")

        print("\n" + "=" * 50)
        print("PIPELINE SẴN SÀNG!")
        print("=" * 50 + "\n")

    def _detect_text_regions(self, image):
        """
        Bước 1: Sử dụng EasyOCR để phát hiện vùng chữ.
        Chỉ trả về bounding boxes, KHÔNG dùng kết quả recognition của EasyOCR.

        Args:
            image: Ảnh BGR (numpy array) đọc từ cv2.imread

        Returns:
            List of bounding boxes: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # detect() chỉ trả về bounding boxes, không recognition
        horizontal_list, free_list = self.detector.detect(
            image,
            width_ths=0.7,
            mag_ratio=1.5  # Phóng to ảnh lên 1.5x để detect chữ nhỏ tốt hơn
        )

        bboxes = []

        # horizontal_list chứa danh sách [x_min, x_max, y_min, y_max]
        if horizontal_list and len(horizontal_list[0]) > 0:
            for box in horizontal_list[0]:
                x_min, x_max, y_min, y_max = box
                bboxes.append({
                    'bbox': [[x_min, y_min], [x_max, y_min],
                             [x_max, y_max], [x_min, y_max]],
                    'x_min': int(x_min), 'y_min': int(y_min),
                    'x_max': int(x_max), 'y_max': int(y_max)
                })

        # free_list chứa các polygon phức tạp (chữ xiên, cong)
        if free_list and len(free_list[0]) > 0:
            for polygon in free_list[0]:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                bboxes.append({
                    'bbox': polygon,
                    'x_min': int(min(xs)), 'y_min': int(min(ys)),
                    'x_max': int(max(xs)), 'y_max': int(max(ys))
                })

        return bboxes

    def _crop_and_recognize(self, image, bbox_info):
        """
        Bước 2 & 3: Crop ảnh theo bounding box rồi đưa vào VietOCR nhận dạng.

        Args:
            image: Ảnh gốc BGR (numpy array)
            bbox_info: Dict chứa thông tin bounding box

        Returns:
            (text, confidence): Chuỗi text nhận dạng được và độ tự tin
        """
        h, w = image.shape[:2]

        # Crop ảnh theo bounding box (có padding an toàn)
        pad = 2  # thêm 2px padding để tránh cắt sát viền chữ
        x1 = max(0, bbox_info['x_min'] - pad)
        y1 = max(0, bbox_info['y_min'] - pad)
        x2 = min(w, bbox_info['x_max'] + pad)
        y2 = min(h, bbox_info['y_max'] + pad)

        crop_bgr = image[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return "", 0.0

        # Chuyển từ BGR (OpenCV) sang RGB (PIL) — VietOCR yêu cầu PIL Image
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        # VietOCR predict
        try:
            text, prob = self.recognizer.predict(pil_img, return_prob=True)
            confidence = float(prob) if prob is not None else 0.0
        except Exception as e:
            print(f"  ⚠ Lỗi VietOCR: {e}")
            text, confidence = "", 0.0

        return text.strip(), confidence

    def _sort_results(self, results):
        """
        Sắp xếp kết quả theo thứ tự đọc tự nhiên:
        Trên → Dưới, Trái → Phải.
        Các box cùng hàng (chênh y < 15px) sẽ được sắp theo x.
        """
        if not results:
            return results

        # Sắp xếp sơ bộ theo y_min
        results.sort(key=lambda r: r['y_min'])

        # Gom các box cùng hàng (chênh lệch y < ngưỡng)
        LINE_THRESHOLD = 15  # pixel
        lines = []
        current_line = [results[0]]

        for i in range(1, len(results)):
            if abs(results[i]['y_min'] - current_line[0]['y_min']) < LINE_THRESHOLD:
                current_line.append(results[i])
            else:
                lines.append(current_line)
                current_line = [results[i]]
        lines.append(current_line)

        # Sắp xếp mỗi hàng theo x_min (trái sang phải)
        sorted_results = []
        for line in lines:
            line.sort(key=lambda r: r['x_min'])
            sorted_results.extend(line)

        return sorted_results

    def run_ocr(self, image_path):
        """
        Chạy toàn bộ pipeline OCR.

        Args:
            image_path: Đường dẫn tới ảnh cần OCR

        Returns:
            List[dict]: Danh sách kết quả, mỗi phần tử có dạng:
                {
                    "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                    "text": "recognized text",
                    "confidence": 0.95
                }
        """
        # Đọc ảnh
        full_img = cv2.imread(image_path)
        if full_img is None:
            print(f"Lỗi: Không thể mở ảnh tại '{image_path}'")
            return []

        h_img, w_img = full_img.shape[:2]
        print(f"Ảnh: {image_path} ({w_img}x{h_img})")

        # ===== BƯỚC 1: DETECTION (EasyOCR) =====
        print("\n[DETECTION] Đang tìm vùng chữ bằng EasyOCR...")
        bboxes = self._detect_text_regions(full_img)
        print(f"  → Tìm thấy {len(bboxes)} vùng chữ")

        if not bboxes:
            print("  ⚠ Không phát hiện được vùng chữ nào!")
            return []

        # ===== BƯỚC 2 & 3: CROP + RECOGNITION (VietOCR) =====
        print("\n[RECOGNITION] Đang nhận dạng bằng VietOCR...")
        results = []

        for i, bbox_info in enumerate(bboxes):
            text, confidence = self._crop_and_recognize(full_img, bbox_info)

            if text.strip():
                # NLP Post-Processing
                processed_text = text
                if self.enable_nlp and self.nlp_processor:
                    processed_text = self.nlp_processor.process(text)

                result = {
                    'bbox': bbox_info['bbox'],
                    'text': processed_text.strip() if processed_text.strip() else text,
                    'text_raw': text,
                    'confidence': confidence,
                    'x_min': bbox_info['x_min'],
                    'y_min': bbox_info['y_min'],
                    'x_max': bbox_info['x_max'],
                    'y_max': bbox_info['y_max'],
                }
                results.append(result)

                print(f"  [{i+1:2d}] '{text}' (conf={confidence:.2f})"
                      + (f" → NLP: '{processed_text}'" if processed_text != text else ""))

        # ===== BƯỚC 4: SẮP XẾP =====
        results = self._sort_results(results)

        # ===== BƯỚC 5: VISUALIZATION =====
        display_img = full_img.copy()
        for r in results:
            x1, y1, x2, y2 = r['x_min'], r['y_min'], r['x_max'], r['y_max']
            color = (0, 255, 0)  # Xanh lá
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)

            # Vẽ text lên ảnh (dùng font latin, không hiển thị tiếng Việt đầy đủ)
            label = f"{r['confidence']:.0%}"
            cv2.putText(display_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        output_path = "ocr_result_viewer.jpg"
        cv2.imwrite(output_path, display_img)
        print(f"\n  Đã lưu ảnh kết quả: {output_path}")

        try:
            os.startfile(output_path)
        except Exception:
            pass

        # ===== IN KẾT QUẢ =====
        full_text = " ".join([r['text'] for r in results])
        print(f"\n{'='*50}")
        print(f"VĂN BẢN NHẬN DẠNG ĐƯỢC ({len(results)} vùng):")
        print(f"{'='*50}")
        for i, r in enumerate(results):
            print(f"  {i+1}. '{r['text']}' (conf={r['confidence']:.2f})")
        print(f"{'='*50}")
        print(f"TOÀN VĂN: {full_text}")
        print(f"{'='*50}")

        # Trả về format chuẩn
        return [
            {
                'bbox': r['bbox'],
                'text': r['text'],
                'confidence': r['confidence']
            }
            for r in results
        ]


if __name__ == "__main__":
    # Khởi tạo pipeline
    # Lựa chọn model VietOCR:
    #   - 'vgg_transformer': Transformer decoder, chính xác nhất (88%)
    #   - 'vgg_seq2seq': Attention Seq2Seq, nhanh nhất (12ms)
    pipeline = FullOCRPipeline(
        vietocr_model='vgg_transformer',
        enable_nlp=True
    )

    # Đổi đường dẫn ảnh bạn muốn test tại đây
    image_to_test = "data/dataset/test_image/im1215.jpg"

    print(f"\nĐang thực hiện OCR cho: {image_to_test}...")
    results = pipeline.run_ocr(image_to_test)

    print(f"\nJSON Output:")
    import json
    print(json.dumps(results, ensure_ascii=False, indent=2))