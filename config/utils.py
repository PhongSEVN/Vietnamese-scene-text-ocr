"""
utils.py - Utility functions for Vietnamese Scene Text Recognition

Contains:
- Character set handling
- CTC decoding
- Metrics calculation (Word Accuracy, Character Error Rate)
- Text encoding/decoding
"""

import os
import re
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch


class CharacterSet:
    """
    Vietnamese Character Set for OCR
    
    Handles mapping between characters and indices for CTC loss.
    Index 0 is reserved for CTC blank token.
    """
    
    def __init__(self, dict_path: Optional[str] = None):
        """
        Initialize character set.
        
        Args:
            dict_path: Optional path to dictionary file with characters
        """
        # Base Vietnamese character set
        # Includes: uppercase, lowercase, digits, punctuation, Vietnamese diacritics
        self.chars = self._build_vietnamese_charset()
        
        # If dictionary provided, augment with additional chars
        if dict_path and os.path.exists(dict_path):
            self._load_from_dict(dict_path)
        
        # CTC blank at index 0
        self.blank_idx = 0
        
        # Build mappings
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.idx_to_char[0] = ''  # blank token
        
    def _build_vietnamese_charset(self) -> str:
        """Build comprehensive Vietnamese character set."""
        # Basic ASCII
        ascii_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ascii_lower = 'abcdefghijklmnopqrstuvwxyz'
        digits = '0123456789'
        punctuation = ' .,!?:;-()[]{}\'\"@#$%&*/\\+=<>_~`|'
        
        # Vietnamese specific characters (uppercase)
        vn_upper = 'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ'
        
        # Vietnamese specific characters (lowercase)
        vn_lower = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
        
        return ascii_upper + ascii_lower + digits + punctuation + vn_upper + vn_lower
    
    def _load_from_dict(self, dict_path: str):
        """Load additional characters from dictionary file."""
        additional_chars = set()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                for c in line.strip():
                    if c not in self.chars:
                        additional_chars.add(c)
        
        if additional_chars:
            self.chars += ''.join(sorted(additional_chars))
    
    def __len__(self) -> int:
        """Total number of classes including blank."""
        return len(self.chars) + 1
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of indices.
        
        Args:
            text: Input string
            
        Returns:
            List of character indices
        """
        encoded = []
        for c in text:
            if c in self.char_to_idx:
                encoded.append(self.char_to_idx[c])
            # Skip unknown characters
        return encoded
    
    def decode(self, indices: List[int], remove_duplicates: bool = True) -> str:
        """
        Decode list of indices to text.
        
        Args:
            indices: List of character indices
            remove_duplicates: Whether to apply CTC decoding (remove duplicates and blanks)
            
        Returns:
            Decoded string
        """
        if remove_duplicates:
            # CTC decoding: remove consecutive duplicates, then remove blanks
            decoded = []
            prev_idx = -1
            for idx in indices:
                if idx != prev_idx:
                    if idx != self.blank_idx:
                        decoded.append(idx)
                    prev_idx = idx
            indices = decoded
        
        return ''.join(self.idx_to_char.get(idx, '') for idx in indices)


def ctc_decode(log_probs: torch.Tensor, 
               charset: CharacterSet,
               beam_width: int = 0) -> List[str]:
    """
    Decode CTC output to strings.
    
    Args:
        log_probs: Model output [T, B, C] or [B, T, C]
        charset: CharacterSet instance
        beam_width: If > 0, use beam search. Otherwise use greedy.
        
    Returns:
        List of decoded strings
    """
    # Ensure shape is [T, B, C]
    if log_probs.dim() == 3:
        if log_probs.shape[0] < log_probs.shape[1]:
            log_probs = log_probs.permute(1, 0, 2)
    
    batch_size = log_probs.shape[1]
    decoded_texts = []
    
    # Greedy decoding
    for b in range(batch_size):
        probs = log_probs[:, b, :]  # [T, C]
        indices = probs.argmax(dim=-1).cpu().numpy().tolist()
        text = charset.decode(indices, remove_duplicates=True)
        decoded_texts.append(text)
    
    return decoded_texts


def calculate_accuracy(predictions: List[str], 
                       targets: List[str],
                       case_sensitive: bool = False) -> Dict[str, float]:
    """
    Calculate various accuracy metrics.
    
    Args:
        predictions: List of predicted strings
        targets: List of ground truth strings
        case_sensitive: Whether to consider case
        
    Returns:
        Dictionary with metrics:
        - word_accuracy: Percentage of exactly matching words
        - char_accuracy: Character-level accuracy
        - cer: Character Error Rate
    """
    if not case_sensitive:
        predictions = [p.lower() for p in predictions]
        targets = [t.lower() for t in targets]
    
    total = len(predictions)
    if total == 0:
        return {'word_accuracy': 0.0, 'char_accuracy': 0.0, 'cer': 1.0}
    
    # Word accuracy (exact match)
    correct_words = sum(1 for p, t in zip(predictions, targets) if p == t)
    word_accuracy = correct_words / total
    
    # Character-level metrics
    total_chars = 0
    correct_chars = 0
    edit_distance_sum = 0
    
    for pred, target in zip(predictions, targets):
        total_chars += len(target)
        
        # Count matching characters (simple approach)
        min_len = min(len(pred), len(target))
        correct_chars += sum(1 for i in range(min_len) if pred[i] == target[i])
        
        # Edit distance for CER
        edit_distance_sum += levenshtein_distance(pred, target)
    
    char_accuracy = correct_chars / max(total_chars, 1)
    cer = edit_distance_sum / max(total_chars, 1)
    
    return {
        'word_accuracy': word_accuracy * 100,
        'char_accuracy': char_accuracy * 100, 
        'cer': cer
    }


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalize_text(text: str) -> str:
    """
    Normalize Vietnamese text for training/evaluation.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles variable-length sequences.
    
    Args:
        batch: List of (image, label_indices, label_length, raw_text) tuples
        
    Returns:
        Batched tensors
    """
    images, labels, label_lengths, raw_texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Pad labels to same length
    max_label_len = max(len(l) for l in labels)
    padded_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            label = label.tolist()
        padded = label + [0] * (max_label_len - len(label))
        padded_labels.append(padded)
    
    labels = torch.LongTensor(padded_labels)
    label_lengths = torch.IntTensor(label_lengths)
    
    return images, labels, label_lengths, raw_texts


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = 'meter'):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: dict, 
                   filepath: str,
                   is_best: bool = False):
    """
    Save training checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    import shutil
    
    torch.save(state, filepath)
    
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        shutil.copy(filepath, best_path)


def load_checkpoint(filepath: str, 
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


# Test the charset
if __name__ == "__main__":
    charset = CharacterSet()
    print(f"Character set size: {len(charset)}")
    print(f"Sample chars: {charset.chars[:50]}...")
    
    # Test encode/decode
    test_text = "Xin chào Việt Nam!"
    encoded = charset.encode(test_text)
    decoded = charset.decode(encoded, remove_duplicates=False)
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded[:20]}...")
    print(f"Decoded: {decoded}")
    
    # Test metrics
    preds = ["hello", "world", "VietNam"]
    targets = ["hello", "word", "vietnam"]
    metrics = calculate_accuracy(preds, targets, case_sensitive=False)
    print(f"\nMetrics: {metrics}")
