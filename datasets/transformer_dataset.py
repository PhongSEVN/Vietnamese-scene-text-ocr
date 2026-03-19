import os
import re
import cv2
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset


VIETNAMESE_CHARS = (
    'aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợ'
    'pqrstuùủũúụưừửữứựvwxyỳỷỹýỵz'
    'AÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬBCDĐEÈẺẼÉẸÊỀỂỄẾỆFGHIÌỈĨÍỊJKLMNOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢ'
    'PQRSTUÙỦŨÚỤƯỪỬỮỨỰVWXYỲỶỸÝỴZ'
    '0123456789'
    ' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
)


class Vocab:
    PAD = 0
    SOS = 1
    EOS = 2
    MASK = 3

    def __init__(self, chars=None):
        if chars is None:
            chars = VIETNAMESE_CHARS
        self.chars = chars
        self.c2i = {c: i + 4 for i, c in enumerate(chars)}
        self.i2c = {i + 4: c for i, c in enumerate(chars)}
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, text):
        return [self.SOS] + [self.c2i[c] for c in text if c in self.c2i] + [self.EOS]

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        first = 1 if self.SOS in ids else 0
        last = ids.index(self.EOS) if self.EOS in ids else None
        return ''.join([self.i2c.get(i, '') for i in ids[first:last]])

    def batch_decode(self, arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.tolist()
        return [self.decode(ids) for ids in arr]

    def __len__(self):
        return len(self.c2i) + 4


class TransformerOCRDataset(Dataset):
    def __init__(self, root, img_folder, label_folder, vocab,
                 img_height=32, img_min_width=32, img_max_width=512, augment=False):
        self.vocab = vocab
        self.img_height = img_height
        self.img_min_width = img_min_width
        self.img_max_width = img_max_width
        self.augment = augment

        img_dir = os.path.join(root, img_folder)
        label_dir = os.path.join(root, label_folder)

        self.samples = []
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)

            nums = re.findall(r'\d+', os.path.splitext(img_file)[0])
            if nums:
                label_file = f"gt_{int(nums[-1])}.txt"
            else:
                label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(',')
                        if len(parts) >= 9:
                            coords = list(map(int, parts[:8]))
                            text = ','.join(parts[8:]).strip()
                            if text and text != '###':
                                self.samples.append({'img_path': img_path, 'coords': coords, 'text': text})
            except Exception:
                continue

        print(f"[Dataset] Loaded: {len(self.samples)} samples from '{img_folder}'")

    def _resize_image(self, img):
        h, w = img.shape[:2]
        new_w = int(self.img_height * float(w) / float(h))
        new_w = math.ceil(new_w / 10) * 10
        new_w = max(new_w, self.img_min_width)
        new_w = min(new_w, self.img_max_width)
        return cv2.resize(img, (new_w, self.img_height))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample['img_path'])

        if img is None:
            img = np.zeros((self.img_height, self.img_min_width, 3), dtype=np.uint8)
            text = ''
        else:
            coords = sample['coords']
            xs, ys = coords[0::2], coords[1::2]
            h_img, w_img = img.shape[:2]
            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(w_img, max(xs))
            y2 = min(h_img, max(ys))
            crop = img[y1:y2, x1:x2]
            img = crop if crop.size > 0 else np.zeros((self.img_height, self.img_min_width, 3), dtype=np.uint8)
            text = sample['text']

        if self.augment and img.size > 0:
            try:
                img = np.ascontiguousarray(img)
                import random
                
                if random.random() < 0.3:
                    h, w = img.shape[:2]
                    src_pts = np.float32([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
                    offset = random.randint(1, 5)
                    dst_pts = np.float32([
                        [random.randint(0, offset), random.randint(0, offset)],
                        [w-1 - random.randint(0, offset), random.randint(0, offset)],
                        [random.randint(0, offset), h-1 - random.randint(0, offset)],
                        [w-1 - random.randint(0, offset), h-1 - random.randint(0, offset)]
                    ])
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    img = cv2.warpPerspective(img, M, (w, h))

                if random.random() < 0.2:
                    h, w = img.shape[:2]
                    angle = random.uniform(-5, 5)
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    img = cv2.warpAffine(img, M, (w, h))

                if random.random() < 0.4:
                    alpha = random.uniform(0.7, 1.3)
                    beta = random.uniform(-30, 30)
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                
                if random.random() < 0.3:
                    if random.random() < 0.5:
                        noise = np.random.normal(0, 5, img.shape)
                        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    else:
                        img = cv2.GaussianBlur(img, (3, 3), 0)
            except Exception:
                pass

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._resize_image(img)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img_tensor = torch.FloatTensor(img.transpose(2, 0, 1))
        label_ids = self.vocab.encode(text)

        return img_tensor, torch.LongTensor(label_ids)


def collate_fn(batch):
    imgs, labels = zip(*batch)

    max_w = max(img.shape[2] for img in imgs)
    padded_imgs = []
    img_widths = []
    for img in imgs:
        c, h, w = img.shape
        pad_val = (0.0 - 0.485) / 0.229
        padded = torch.full((c, h, max_w), pad_val)
        padded[:, :, :w] = img
        padded_imgs.append(padded)
        img_widths.append(w)
    img_batch = torch.stack(padded_imgs, 0)

    max_len = max(len(label) for label in labels)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    return img_batch, padded_labels, torch.tensor(img_widths, dtype=torch.long)
