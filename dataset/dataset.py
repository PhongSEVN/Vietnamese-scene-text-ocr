"""
dataset.py - Vietnamese Scene Text Recognition Dataset

Handles:
- Loading images from train/test folders
- Parsing DICT-Guided format labels (x1,y1,x2,y2,x3,y3,x4,y4,text)
- Image preprocessing and augmentation
- Text encoding for CTC loss
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class DictGuidedDataset(Dataset):
    """
    Vietnamese Scene Text Recognition Dataset
    
    Dataset structure:
    - train_images/: im0001.jpg - im1200.jpg (training)
    - test_image/: im1201.jpg - im1500.jpg (test)
    - unseen_test_images/: im1501.jpg - im2000.jpg (unseen test)
    - labels/: gt_1.txt - gt_2000.txt
    
    Label format: x1,y1,x2,y2,x3,y3,x4,y4,text
    - x1,y1...x4,y4: polygon coordinates
    - text: ground truth (### means ignore)
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 charset = None,
                 img_height: int = 32,
                 img_width: int = 128,
                 max_text_length: int = 25,
                 augment: bool = False):
        """
        Initialize dataset.
        
        Args:
            data_root: Path to dataset folder (e.g., data/dataset)
            split: 'train', 'test', or 'unseen'
            charset: CharacterSet instance for encoding
            img_height: Target image height
            img_width: Target image width  
            max_text_length: Maximum text length
            augment: Whether to apply augmentation
        """
        self.data_root = Path(data_root)
        self.split = split
        self.charset = charset
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_length = max_text_length
        self.augment = augment and (split == 'train')
        
        # Set image directory based on split
        if split == 'train':
            self.img_dir = self.data_root / 'train_images'
            self.id_range = range(1, 1201)  # im0001 - im1200
        elif split == 'test':
            self.img_dir = self.data_root / 'test_image'
            self.id_range = range(1201, 1501)  # im1201 - im1500
        else:  # unseen
            self.img_dir = self.data_root / 'unseen_test_images'
            self.id_range = range(1501, 2001)  # im1501 - im2000
        
        self.labels_dir = self.data_root / 'labels'
        
        # Load all samples
        self.samples = self._load_samples()
        
        print(f"[Dataset] Loaded {len(self.samples)} samples from {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """
        Load all valid samples (image path + cropped text regions).
        
        Returns:
            List of sample dictionaries with 'image', 'text', 'bbox' keys
        """
        samples = []
        
        for img_id in self.id_range:
            # Image path
            img_name = f"im{img_id:04d}.jpg"
            img_path = self.img_dir / img_name
            
            if not img_path.exists():
                continue
            
            # Label path
            label_path = self.labels_dir / f"gt_{img_id}.txt"
            
            if not label_path.exists():
                continue
            
            # Parse label file
            text_boxes = self._parse_label(label_path)
            
            # Create sample for each valid text box
            for box in text_boxes:
                if box['text'] != '###' and len(box['text']) > 0:
                    # Filter out too long texts
                    if len(box['text']) <= self.max_text_length:
                        samples.append({
                            'image_path': str(img_path),
                            'bbox': box['bbox'],
                            'text': box['text']
                        })
        
        return samples
    
    def _parse_label(self, label_path: Path) -> List[Dict]:
        """
        Parse label file.
        
        Args:
            label_path: Path to gt_X.txt file
            
        Returns:
            List of text boxes with bbox and text
        """
        text_boxes = []
        
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 9:
                    continue
                
                try:
                    # Parse 8 coordinates
                    coords = [int(parts[i]) for i in range(8)]
                    bbox = np.array(coords).reshape(4, 2)
                    
                    # Text is everything after the 8 coordinates
                    text = ','.join(parts[8:])
                    
                    text_boxes.append({
                        'bbox': bbox,
                        'text': text
                    })
                except (ValueError, IndexError):
                    continue
        
        return text_boxes
    
    def _crop_and_resize(self, 
                         image: np.ndarray, 
                         bbox: np.ndarray) -> np.ndarray:
        """
        Crop text region from image using perspective transform.
        
        Args:
            image: Full image
            bbox: 4 corner points
            
        Returns:
            Cropped and resized image
        """
        # Get bounding rectangle
        x_coords = bbox[:, 0]
        y_coords = bbox[:, 1]
        x_min, x_max = max(0, int(x_coords.min())), int(x_coords.max())
        y_min, y_max = max(0, int(y_coords.min())), int(y_coords.max())
        
        # Ensure valid crop region
        h, w = image.shape[:2]
        x_max = min(x_max, w)
        y_max = min(y_max, h)
        
        if x_max <= x_min or y_max <= y_min:
            # Return empty image if invalid
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        # Simple crop (bounding box approach)
        cropped = image[y_min:y_max, x_min:x_max]
        
        if cropped.size == 0:
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        # Resize to target size (keep aspect ratio, pad)
        resized = self._resize_pad(cropped)
        
        return resized
    
    def _resize_pad(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while keeping aspect ratio and pad to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        
        # Calculate scale to fit height
        scale = self.img_height / h
        new_w = int(w * scale)
        
        # Limit width
        if new_w > self.img_width:
            new_w = self.img_width
            scale = new_w / w
        
        # Resize
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image (white background)
        padded = np.full((self.img_height, self.img_width, 3), 255, dtype=np.uint8)
        
        # Center the image
        y_offset = (self.img_height - new_h) // 2
        x_offset = 0  # Align to left for text
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Random rotation (small angle)
        if random.random() > 0.7:
            angle = random.uniform(-5, 5)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), 
                                   borderMode=cv2.BORDER_REPLICATE)
        
        # Random Gaussian blur
        if random.random() > 0.7:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        return image
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor and normalize.
        
        Args:
            image: Image as numpy array (H, W, C) in BGR
            
        Returns:
            Tensor (C, H, W) normalized to [0, 1]
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Transpose to (C, H, W)
        image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], int, str]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, encoded_label, label_length, raw_text)
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        
        if image is None:
            # Return dummy sample if image load fails
            image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            # Crop text region
            image = self._crop_and_resize(image, sample['bbox'])
        
        # Apply augmentation
        if self.augment:
            image = self._augment(image)
        
        # Convert to tensor
        image_tensor = self._to_tensor(image)
        
        # Encode text
        text = sample['text']
        if self.charset:
            encoded_label = self.charset.encode(text)
        else:
            encoded_label = []
        
        label_length = len(encoded_label)
        
        return image_tensor, encoded_label, label_length, text


class WholeImageDataset(Dataset):
    """
    Alternative dataset that uses whole images (not cropped word regions).
    Simpler approach for demonstration.
    
    Each sample is one text box label - we extract the word region.
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 charset = None,
                 img_height: int = 32,
                 img_width: int = 128,
                 max_text_length: int = 25):
        """Initialize dataset."""
        
        self.data_root = Path(data_root)
        self.charset = charset
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_length = max_text_length
        
        # Determine split ranges
        if split == 'train':
            self.img_dir = self.data_root / 'train_images'
            id_range = range(1, 1201)
        elif split == 'test':
            self.img_dir = self.data_root / 'test_image'
            id_range = range(1201, 1501)
        else:
            self.img_dir = self.data_root / 'unseen_test_images'
            id_range = range(1501, 2001)
        
        self.labels_dir = self.data_root / 'labels'
        self.samples = self._load_samples(id_range)
        
        print(f"[WholeImageDataset] Loaded {len(self.samples)} samples from {split}")
    
    def _load_samples(self, id_range) -> List[Tuple[str, str]]:
        """Load (image_path, text) pairs."""
        samples = []
        
        for img_id in id_range:
            img_name = f"im{img_id:04d}.jpg"
            img_path = self.img_dir / img_name
            
            if not img_path.exists():
                continue
            
            label_path = self.labels_dir / f"gt_{img_id}.txt"
            if not label_path.exists():
                continue
            
            # Parse labels
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) < 9:
                        continue
                    
                    try:
                        coords = [int(parts[i]) for i in range(8)]
                        bbox = np.array(coords).reshape(4, 2)
                        text = ','.join(parts[8:])
                        
                        if text != '###' and 0 < len(text) <= self.max_text_length:
                            samples.append((str(img_path), bbox, text))
                    except:
                        continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, bbox, text = self.samples[idx]
        
        # Load and crop
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            # Crop
            x_coords = bbox[:, 0]
            y_coords = bbox[:, 1]
            x_min = max(0, int(x_coords.min()))
            y_min = max(0, int(y_coords.min()))
            x_max = min(image.shape[1], int(x_coords.max()))
            y_max = min(image.shape[0], int(y_coords.max()))
            
            if x_max > x_min and y_max > y_min:
                image = image[y_min:y_max, x_min:x_max]
            else:
                image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        # Resize
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # Convert to tensor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Encode
        encoded = self.charset.encode(text) if self.charset else []
        
        return image, encoded, len(encoded), text


# Test
if __name__ == "__main__":
    from utils import CharacterSet
    
    charset = CharacterSet()
    
    # Test dataset
    dataset = DictGuidedDataset(
        data_root="data/dataset",
        split='train',
        charset=charset,
        augment=False
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get sample
    if len(dataset) > 0:
        img, label, length, text = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Label: {label[:10]}... (length={length})")
        print(f"Text: {text}")
        
        # Decode
        decoded = charset.decode(label, remove_duplicates=False)
        print(f"Decoded: {decoded}")
