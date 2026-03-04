import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class VinAIDataset(Dataset):
    def __init__(self, root, img_folder, label_folder, transform=None, converter=None):
        """
        root       : Thư mục chứa data (data/dataset)
        img_folder : Thư mục ảnh VÀO (train_images | test_image)
        label_folder: Thư mục nhãn (labels)
        transform  : Transform áp dụng lên crop ảnh
        converter  : Converter để chuyển đổi chữ sang indices (None sẽ trả về string)
        """
        self.root = root
        self.img_folder = os.path.join(root, img_folder)
        self.label_folder = os.path.join(root, label_folder)
        self.transform = transform
        self.converter = converter

        self.data_list = []
        self._load_annotations()

    def _load_annotations(self):
        """Đọc file gt_*.txt và tạo danh sách word crops."""
        # ... (giữ nguyên logic load)
        if not os.path.exists(self.label_folder):
            print(f"Lỗi: Không tìm thấy thư mục nhãn tại {self.label_folder}")
            return

        if not os.path.exists(self.img_folder):
            print(f"Lỗi: Không tìm thấy thư mục ảnh tại {self.img_folder}")
            return

        existing_imgs = set()
        for fname in os.listdir(self.img_folder):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                try:
                    num = int(fname.replace('im', '').split('.')[0])
                    existing_imgs.add(num)
                except ValueError:
                    pass

        label_files = sorted(
            [f for f in os.listdir(self.label_folder) if f.endswith('.txt')],
            key=lambda x: int(x.replace('gt_', '').replace('.txt', ''))
        )

        loaded = 0
        for label_file in label_files:
            num_part = label_file.replace('gt_', '').replace('.txt', '')
            img_num = int(num_part)

            if img_num not in existing_imgs:
                continue

            img_name = f"im{img_num:04d}.jpg"
            img_path = os.path.join(self.img_folder, img_name)

            try:
                with open(os.path.join(self.label_folder, label_file), 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 9:
                            continue

                        coords = [int(p) for p in parts[:8]]
                        text = ",".join(parts[8:]).strip()

                        if text in ("###", "") or text.startswith("###"):
                            continue

                        self.data_list.append({
                            'img_path': img_path,
                            'coords': coords,
                            'text': text
                        })
                        loaded += 1
            except Exception as e:
                print(f"Lỗi đọc file {label_file}: {e}")

        print(f"[Dataset] Đã load: {loaded} samples từ '{os.path.basename(self.img_folder)}'")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = cv2.imread(item['img_path'])

        if image is None:
            return torch.zeros((1, 32, 128)), "" if self.converter is None else torch.zeros(25).long()

        x_coords = item['coords'][0::2]
        y_coords = item['coords'][1::2]
        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

        h_orig, w_orig = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)

        crop = image[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            crop = np.zeros((32, 128, 3), dtype=np.uint8)

        if self.transform:
            crop = self.transform(crop)

        label = item['text']
        if self.converter:
            # Mã hóa nhãn sang định dạng 105 ký tự
            indices = self.converter.encode([label])[0]
            return crop, indices
        
        return crop, label



class ResizeNormalize:
    def __init__(self, size=(128, 32)):
        self.size = size  # (W, H)

    def __call__(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        target_w, target_h = self.size
        h, w = img.shape

        # Tránh lỗi chia cho 0
        if w == 0 or h == 0:
            return torch.zeros((1, target_h, target_w))

        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Lấy màu nền từ góc ảnh (tránh hardcode nền trắng)
        bg_color = int(np.median([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]))
        canvas = np.full((target_h, target_w), bg_color, dtype=np.uint8)

        # Căn giữa ảnh trong canvas (cả dọc lẫn ngang)
        offset_y = (target_h - new_h) // 2
        offset_x = (target_w - new_w) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = img

        img = canvas.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).unsqueeze(0)
