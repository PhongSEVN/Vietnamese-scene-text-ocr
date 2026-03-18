"""
Script chuyển đổi MCOCR2020 → format dataset hiện tại.
"""

import os
import csv
import ast
import numpy as np
import cv2

_BASE = os.path.dirname(os.path.abspath(__file__))
MCOCR_CSV     = os.path.join(_BASE, "data", "data loại khác", "MCOCR2020", "mcocr_public", "mcocr_train_data", "mcocr_train_df.csv")
MCOCR_IMG_DIR = os.path.join(_BASE, "data", "data loại khác", "MCOCR2020", "mcocr_public", "mcocr_train_data", "train_images")
OUT_IMG_DIR   = os.path.join(_BASE, "data", "dataset", "train_images")
OUT_LABEL_DIR = os.path.join(_BASE, "data", "dataset", "labels")

MAX_TEXT_LEN = 25   # bỏ text dài hơn max_text_length trong config
PADDING      = 4    # pixel padding quanh bbox khi crop
START_IDX    = 10001  # tránh collision với gt_1..gt_2000

# ────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)

    saved        = 0
    skip_text    = 0
    skip_img     = 0
    skip_crop    = 0

    with open(MCOCR_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"MCOCR invoice images: {len(rows)}")

    for row in rows:
        img_path = os.path.join(MCOCR_IMG_DIR, row["img_id"])
        if not os.path.exists(img_path):
            skip_img += 1
            continue

        try:
            polygons = ast.literal_eval(row["anno_polygons"])
        except Exception:
            continue

        texts = row["anno_texts"].split("|||")
        if len(polygons) != len(texts):
            continue

        img = None  # lazy load — chỉ đọc ảnh khi có text hợp lệ

        for poly, text in zip(polygons, texts):
            text = text.strip()

            # Lọc text không hợp lệ
            if not text or text == "###" or len(text) > MAX_TEXT_LEN:
                skip_text += 1
                continue

            bbox = poly.get("bbox")  # [x, y, w, h]
            if not bbox or len(bbox) < 4:
                continue

            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if w <= 0 or h <= 0:
                continue

            if img is None:
                # Dùng np.fromfile để đọc path Unicode trên Windows
                buf = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    skip_img += 1
                    break

            img_h, img_w = img.shape[:2]
            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(img_w, x + w + PADDING)
            y2 = min(img_h, y + h + PADDING)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                skip_crop += 1
                continue

            idx       = START_IDX + saved
            out_img   = os.path.join(OUT_IMG_DIR,   f"mcocr_{idx}.jpg")
            out_label = os.path.join(OUT_LABEL_DIR, f"gt_{idx}.txt")

            # Dùng imencode + tofile để ghi path Unicode trên Windows
            ok, buf = cv2.imencode(".jpg", crop)
            if not ok:
                continue
            np.array(buf).tofile(out_img)

            ch, cw = crop.shape[:2]
            # Label format: x1,y1,x2,y2,x3,y3,x4,y4,text  (4 góc theo chiều kim đồng hồ)
            with open(out_label, "w", encoding="utf-8") as lf:
                lf.write(f"0,0,{cw},0,{cw},{ch},0,{ch},{text}\n")

            saved += 1

    total_after = 43045 + saved  # 43045 = số mẫu gốc
    ratio = saved / total_after * 100

    print(f"\n{'='*45}")
    print(f"Crops saved      : {saved}")
    print(f"Skipped (text)   : {skip_text}  (dài > {MAX_TEXT_LEN} hoặc rỗng)")
    print(f"Skipped (image)  : {skip_img}")
    print(f"Skipped (crop)   : {skip_crop}  (bbox quá nhỏ)")
    print(f"{'='*45}")
    print(f"Dataset gốc      : 43,045 mẫu")
    print(f"MCOCR thêm vào   : {saved:,} mẫu")
    print(f"Tỷ lệ MCOCR      : {ratio:.1f}%")
    print(f"Tổng              : {total_after:,} mẫu")
    print(f"{'='*45}")
    print(f"\nAnh/label đã lưu vào:")
    print(f"  {OUT_IMG_DIR}/mcocr_10001.jpg ... mcocr_{START_IDX+saved-1}.jpg")
    print(f"  {OUT_LABEL_DIR}/gt_10001.txt  ... gt_{START_IDX+saved-1}.txt")


if __name__ == "__main__":
    main()
