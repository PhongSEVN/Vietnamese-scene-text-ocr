"""
Module NLP Post-Processing cho OCR tiếng Việt.
Chuẩn hóa và sửa lỗi kết quả dự đoán từ mô hình CRNN.

Pipeline xử lý:
  1. Unicode Normalization (NFC)
  2. Xử lý khoảng trắng & dấu câu
  3. Sửa lỗi chính tả (SymSpell + từ điển tiếng Việt)
  4. Tách từ tiếng Việt (underthesea word_tokenize)
  5. Chuẩn hóa dấu tiếng Việt (compose dấu đúng vị trí)
"""

import re
import os
import unicodedata
from typing import List, Optional


# ============================================================
# 1. UNICODE & TEXT NORMALIZATION
# ============================================================

class VietnameseTextNormalizer:
    """Chuẩn hóa text tiếng Việt: Unicode, khoảng trắng, dấu câu."""

    # Bảng mapping Unicode tổ hợp → precomposed phổ biến bị lỗi
    # (Một số font/hệ thống trả về dạng NFD thay vì NFC)
    COMMON_UNICODE_FIXES = {
        '\u0065\u0309': 'ẻ',  # e + hook above
        '\u0061\u0309': 'ả',  # a + hook above  
        '\u006F\u0303': 'õ',  # o + tilde
        '\u0075\u0301': 'ú',  # u + acute
    }

    # Ký tự đặc biệt thường bị OCR nhận sai
    OCR_CHAR_FIXES = {
        '|': 'l',   # pipe -> chữ l
        '0': {       # số 0 có thể bị nhầm với chữ O (xử lý theo context)
            'default': '0',
        },
        '1': {       # số 1 có thể bị nhầm với chữ l hoặc I
            'default': '1',
        },
    }

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Chuẩn hóa Unicode về dạng NFC (precomposed)."""
        return unicodedata.normalize('NFC', text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Chuẩn hóa khoảng trắng: loại bỏ thừa, trim."""
        # Thay thế nhiều khoảng trắng liên tiếp bằng 1
        text = re.sub(r'\s+', ' ', text)
        # Xóa khoảng trắng trước dấu câu
        text = re.sub(r'\s+([.,!?;:)])', r'\1', text)
        # Thêm khoảng trắng sau dấu câu nếu thiếu
        text = re.sub(r'([.,!?;:])([^\s\d.,!?;:)])', r'\1 \2', text)
        return text.strip()

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Chuẩn hóa dấu câu: loại bỏ dấu lặp, sửa dấu sai."""
        # Loại bỏ dấu câu lặp: "..." -> "...", "!!" -> "!"
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        # Sửa dấu ngoặc sai
        text = text.replace('（', '(').replace('）', ')')
        return text

    @staticmethod
    def remove_invalid_chars(text: str) -> str:
        """Loại bỏ ký tự điều khiển và ký tự không hợp lệ."""
        # Giữ lại: chữ cái (bao gồm tiếng Việt), số, dấu câu phổ biến, khoảng trắng
        cleaned = []
        for char in text:
            cat = unicodedata.category(char)
            # Giữ: Letter, Number, Punctuation, Space, Symbol thông thường
            if cat.startswith(('L', 'N', 'P', 'Z', 'S')):
                cleaned.append(char)
            elif char in (' ', '\t', '\n'):
                cleaned.append(char)
        return ''.join(cleaned)

    def normalize(self, text: str) -> str:
        """Chạy toàn bộ pipeline chuẩn hóa text."""
        if not text:
            return text
        text = self.normalize_unicode(text)
        text = self.remove_invalid_chars(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_whitespace(text)
        return text


# ============================================================
# 2. VIETNAMESE DIACRITICS HANDLER (Xử lý dấu tiếng Việt)
# ============================================================

class VietnameseDiacriticsHandler:
    """
    Xử lý và chuẩn hóa dấu tiếng Việt.
    Sửa các lỗi phổ biến do OCR gây ra với dấu thanh.
    """

    # Quy tắc đặt dấu tiếng Việt (kiểu mới - đặt dấu ở nguyên âm đôi/ba)
    # Ví dụ: "hoà" thay vì "hòa" (tùy quy tắc cũ/mới)
    
    # Bảng các nguyên âm tiếng Việt
    VOWELS = set('aàảãáạăằẳẵắặâầẩẫấậeèẻẽéẹêềểễếệiìỉĩíịoòỏõóọôồổỗốộơờởỡớợuùủũúụưừửữứựyỳỷỹýỵ')
    VOWELS_UPPER = set('AÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬEÈẺẼÉẸÊỀỂỄẾỆIÌỈĨÍỊOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢUÙỦŨÚỤƯỪỬỮỨỰYỲỶỸÝỴ')
    
    # Mapping nguyên âm gốc → các dạng có dấu
    VOWEL_BASE_MAP = {
        'a': 'aàảãáạ', 'ă': 'ăằẳẵắặ', 'â': 'âầẩẫấậ',
        'e': 'eèẻẽéẹ', 'ê': 'êềểễếệ',
        'i': 'iìỉĩíị',
        'o': 'oòỏõóọ', 'ô': 'ôồổỗốộ', 'ơ': 'ơờởỡớợ',
        'u': 'uùủũúụ', 'ư': 'ưừửữứự',
        'y': 'yỳỷỹýỵ',
    }

    # Các lỗi dấu thường gặp từ OCR
    COMMON_DIACRITIC_ERRORS = {
        # Dấu bị đặt sai vị trí hoặc bị nhầm
        'ướ': 'ướ',  # Chuẩn hóa
        'ườ': 'ườ',
        'ưở': 'ưở',
    }

    @staticmethod
    def get_base_vowel(char: str) -> Optional[str]:
        """Lấy nguyên âm gốc (không dấu thanh) của một ký tự."""
        # Decompose ký tự để tách dấu
        decomposed = unicodedata.normalize('NFD', char)
        if decomposed:
            base = decomposed[0]
            # Tìm nguyên âm gốc
            base_lower = base.lower()
            if base_lower in 'aăâeêioôơuưy':
                return base
        return None

    @staticmethod  
    def fix_common_diacritic_errors(text: str) -> str:
        """Sửa các lỗi dấu tiếng Việt phổ biến."""
        # Chuẩn hóa NFC trước
        text = unicodedata.normalize('NFC', text)

        # Sửa lỗi ký tự bị tách dấu (combining characters còn sót)
        # Ví dụ: 'e' + combining acute -> 'é'
        text = unicodedata.normalize('NFC', text)
        
        return text


# ============================================================
# 3. SPELL CORRECTION (Sửa lỗi chính tả tiếng Việt)
# ============================================================

class VietnameseSpellCorrector:
    """
    Sửa lỗi chính tả tiếng Việt sử dụng:
    - Từ điển tiếng Việt phổ biến
    - Khoảng cách Levenshtein (edit distance)
    - Quy tắc ngữ âm tiếng Việt
    """

    # Từ điển "âm tiết" tiếng Việt hợp lệ (syllables)
    # Tiếng Việt có khoảng ~7000 âm tiết hợp lệ
    
    # Các lỗi ngữ âm phổ biến trong OCR tiếng Việt
    PHONETIC_CONFUSION_PAIRS = {
        # Phụ âm đầu hay bị nhầm
        ('d', 'đ'),    # d ↔ đ
        ('g', 'q'),    # g ↔ q  
        ('gi', 'di'),  # gi ↔ di
        ('tr', 'ch'),  # tr ↔ ch
        ('s', 'x'),    # s ↔ x
        ('r', 'g'),    # r ↔ g (trong một số trường hợp)
        ('l', 'n'),    # l ↔ n
        
        # Nguyên âm hay bị nhầm  
        ('ă', 'â'),    # ă ↔ â
        ('ơ', 'ơ'),    # ơ ↔ ở
        ('ư', 'u'),    # ư ↔ u
        
        # Phụ âm cuối hay bị nhầm
        ('n', 'ng'),   # n ↔ ng
        ('t', 'c'),    # t ↔ c
        ('nh', 'ng'),  # nh ↔ ng
    }

    def __init__(self, dictionary_paths: Optional[List[str]] = None):
        """
        Khởi tạo spell corrector.
        
        Args:
            dictionary_paths: Danh sách đường dẫn tới các file từ điển tiếng Việt.
                            Mỗi file có format: mỗi dòng chứa 1 từ/cụm từ.
                            Hỗ trợ cả format có TAB tần suất: từ<TAB>tần_suất
        """
        self.word_freq = {}
        self.syllable_set = set()
        self._symspell = None
        
        # Hỗ trợ cả input đơn lẻ (string) và danh sách
        if isinstance(dictionary_paths, str):
            dictionary_paths = [dictionary_paths]
        
        loaded_any = False
        if dictionary_paths:
            for dict_path in dictionary_paths:
                if dict_path and os.path.exists(dict_path):
                    self._load_dictionary(dict_path)
                    loaded_any = True
                else:
                    print(f"[NLP Post-Process] Cảnh báo: Không tìm thấy từ điển tại {dict_path}")
        
        if loaded_any:
            self._init_symspell()
            print(f"[NLP Post-Process] TỔNG CỘNG: {len(self.word_freq)} từ, "
                  f"{len(self.syllable_set)} âm tiết duy nhất.")
        else:
            print("[NLP Post-Process] Không load được từ điển nào.")
            print("[NLP Post-Process] Sẽ chỉ sử dụng các rule-based corrections.")

    def _load_dictionary(self, path: str):
        """Load từ điển tiếng Việt từ file. Hỗ trợ format: mỗi dòng 1 từ hoặc từ<TAB>tần_suất."""
        count_before = len(self.word_freq)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Hỗ trợ cả format có TAB tần suất và không có
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        try:
                            freq = int(parts[1].strip())
                        except ValueError:
                            freq = 1
                    else:
                        word = parts[0].strip()
                        freq = 1
                    
                    if not word:
                        continue
                    
                    word_lower = word.lower()
                    # Cộng dồn tần suất nếu từ đã tồn tại
                    self.word_freq[word_lower] = self.word_freq.get(word_lower, 0) + freq
                    
                    # Thêm từng âm tiết vào set  
                    for syllable in word_lower.split():
                        # Loại bỏ dấu câu đi kèm khi thêm vào syllable set
                        clean_syllable = syllable.strip('.,!?;:()[]{}"\'-/')
                        if clean_syllable:
                            self.syllable_set.add(clean_syllable)
            
            count_new = len(self.word_freq) - count_before
            filename = os.path.basename(path)
            print(f"[NLP Post-Process] Đã load '{filename}': +{count_new} từ mới "
                  f"(tổng: {len(self.word_freq)} từ, {len(self.syllable_set)} âm tiết)")
        except Exception as e:
            print(f"[NLP Post-Process] Lỗi đọc từ điển {path}: {e}")

    def _init_symspell(self):
        """Khởi tạo SymSpell cho spell correction nhanh."""
        try:
            from symspellpy import SymSpell, Verbosity
            self._symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            
            # Load từ điển vào SymSpell
            count = 0
            for word, freq in self.word_freq.items():
                self._symspell.create_dictionary_entry(word, freq)
                count += 1
            
            print(f"[NLP Post-Process] SymSpell đã sẵn sàng với {count} entries.")
        except ImportError:
            print("[NLP Post-Process] Không tìm thấy symspellpy. "
                  "Cài đặt: pip install symspellpy")
            print("[NLP Post-Process] Sẽ dùng fallback edit-distance correction.")
            self._symspell = None

    def _edit_distance_1(self, word: str) -> set:
        """Tạo tất cả các từ có edit distance = 1 với word."""
        letters = 'aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệghiklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvxyỳỷỹýỵ'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = {L + R[1:] for L, R in splits if R}
        transposes = {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1}
        replaces = {L + c + R[1:] for L, R in splits if R for c in letters}
        inserts = {L + c + R for L, R in splits for c in letters}
        
        return deletes | transposes | replaces | inserts

    def correct_syllable(self, syllable: str) -> str:
        """
        Sửa lỗi cho một âm tiết tiếng Việt.
        
        Returns:
            Âm tiết đã sửa, hoặc giữ nguyên nếu không tìm thấy gợi ý.
        """
        if not syllable:
            return syllable
            
        syllable_lower = syllable.lower()
        
        # Nếu âm tiết đã đúng trong từ điển → giữ nguyên
        if syllable_lower in self.syllable_set:
            return syllable
        
        # Thử SymSpell trước (nhanh hơn)
        if self._symspell:
            try:
                from symspellpy import Verbosity
                suggestions = self._symspell.lookup(
                    syllable_lower, 
                    Verbosity.CLOSEST, 
                    max_edit_distance=2
                )
                if suggestions:
                    best = suggestions[0]
                    corrected = best.term
                    # Giữ lại casing gốc
                    if syllable[0].isupper():
                        corrected = corrected[0].upper() + corrected[1:]
                    return corrected
            except Exception:
                pass
        
        # Fallback: Brute-force edit distance 1
        if self.syllable_set:
            candidates_1 = self._edit_distance_1(syllable_lower)
            valid_candidates = candidates_1 & self.syllable_set
            
            if valid_candidates:
                # Chọn từ có tần suất cao nhất
                best = max(valid_candidates, 
                          key=lambda w: self.word_freq.get(w, 0))
                if syllable[0].isupper():
                    best = best[0].upper() + best[1:]
                return best
        
        # Không tìm được gợi ý → giữ nguyên
        return syllable

    def correct_text(self, text: str) -> str:
        """Sửa lỗi chính tả cho toàn bộ đoạn text."""
        if not self.syllable_set:
            return text  # Không có từ điển thì trả về nguyên
            
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Tách dấu câu ra khỏi từ
            prefix_punct = ''
            suffix_punct = ''
            
            while word and not word[0].isalnum():
                prefix_punct += word[0]
                word = word[1:]
            while word and not word[-1].isalnum():
                suffix_punct = word[-1] + suffix_punct
                word = word[:-1]
            
            if word:
                corrected = self.correct_syllable(word)
                corrected_words.append(prefix_punct + corrected + suffix_punct)
            else:
                corrected_words.append(prefix_punct + suffix_punct)
        
        return ' '.join(corrected_words)


# ============================================================
# 4. WORD SEGMENTATION (Tách từ tiếng Việt)
# ============================================================

class VietnameseWordSegmenter:
    """
    Tách từ tiếng Việt sử dụng underthesea.
    Tiếng Việt có nhiều từ ghép (2-3 âm tiết), tách từ đúng 
    giúp cải thiện độ chính xác của spell correction.
    """

    def __init__(self):
        self._available = False
        try:
            from underthesea import word_tokenize
            self._word_tokenize = word_tokenize
            self._available = True
            print("[NLP Post-Process] underthesea word_tokenize đã sẵn sàng.")
        except ImportError:
            print("[NLP Post-Process] Không tìm thấy underthesea. "
                  "Cài đặt: pip install underthesea")
            print("[NLP Post-Process] Tách từ sẽ dùng phương pháp đơn giản (split by space).")

    def segment(self, text: str) -> str:
        """
        Tách từ tiếng Việt.
        
        Returns:
            Text đã tách từ (các từ ghép được nối bằng dấu gạch dưới).
        """
        if not self._available or not text.strip():
            return text
        
        try:
            # format="text" trả về string bình thường, từ ghép nối bằng _
            result = self._word_tokenize(text, format="text")
            return result
        except Exception as e:
            print(f"[NLP Post-Process] Lỗi tách từ: {e}")
            return text

    def segment_to_list(self, text: str) -> List[str]:
        """Tách từ và trả về danh sách các từ."""
        if not self._available or not text.strip():
            return text.split()
        
        try:
            result = self._word_tokenize(text)
            return result
        except Exception:
            return text.split()


# ============================================================
# 5. CONFIDENCE-BASED FILTERING (Lọc theo độ tin cậy)
# ============================================================

class ConfidenceFilter:
    """
    Lọc kết quả OCR dựa trên độ tin cậy và các heuristic.
    Loại bỏ các từ "rác" mà mô hình predict sai hoàn toàn.
    """

    # Pattern cho các chuỗi "rác" phổ biến từ OCR
    GARBAGE_PATTERNS = [
        re.compile(r'^[^aàảãáạăằẳẵắặâầẩẫấậeèẻẽéẹêềểễếệiìỉĩíịoòỏõóọôồổỗốộơờởỡớợuùủũúụưừửữứựyỳỷỹýỵAÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬEÈẺẼÉẸÊỀỂỄẾỆIÌỈĨÍỊOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢUÙỦŨÚỤƯỪỬỮỨỰYỲỶỸÝỴ0-9]{4,}$'),  # 4+ phụ âm liên tiếp không có nguyên âm
        re.compile(r'^(.)\1{3,}$'),  # Ký tự lặp 4+ lần: "aaaa", "llll"
        re.compile(r'^[^\w\s]+$'),    # Chỉ toàn ký tự đặc biệt
    ]

    @staticmethod
    def is_valid_vietnamese_syllable(syllable: str) -> bool:
        """Kiểm tra xem một chuỗi có phải âm tiết tiếng Việt hợp lệ không."""
        if not syllable:
            return False
        
        syllable_lower = syllable.lower()
        
        # Phải chứa ít nhất 1 nguyên âm (trừ số)
        vowels = set('aàảãáạăằẳẵắặâầẩẫấậeèẻẽéẹêềểễếệiìỉĩíịoòỏõóọôồổỗốộơờởỡớợuùủũúụưừửữứựyỳỷỹýỵ')
        
        if syllable_lower.isdigit():
            return True  # Số luôn hợp lệ
        
        has_vowel = any(c in vowels for c in syllable_lower)
        
        # Độ dài hợp lý (âm tiết tiếng Việt thường 1-7 ký tự)
        reasonable_length = 1 <= len(syllable_lower) <= 8
        
        return has_vowel and reasonable_length

    @classmethod    
    def filter_garbage(cls, text: str) -> str:
        """Loại bỏ các chuỗi rác khỏi kết quả OCR."""
        words = text.split()
        valid_words = []
        
        for word in words:
            # Kiểm tra garbage patterns
            is_garbage = False
            for pattern in cls.GARBAGE_PATTERNS:
                if pattern.match(word):
                    is_garbage = True
                    break
            
            if not is_garbage and cls.is_valid_vietnamese_syllable(word):
                valid_words.append(word)
            elif word.strip() and len(word) <= 2:
                # Giữ lại từ ngắn (có thể là chữ viết tắt hoặc số)
                valid_words.append(word)
        
        return ' '.join(valid_words)


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

class NLPPostProcessor:
    """
    Pipeline NLP Post-Processing chính.
    Kết hợp tất cả các bước xử lý theo thứ tự.
    """

    def __init__(self, dictionary_paths: Optional[List[str]] = None, 
                 enable_spell_correction: bool = True,
                 enable_word_segmentation: bool = True,
                 enable_confidence_filter: bool = True):
        """
        Khởi tạo NLP Post-Processor.
        
        Args:
            dictionary_paths: Danh sách đường dẫn tới các file từ điển tiếng Việt
            enable_spell_correction: Bật sửa lỗi chính tả
            enable_word_segmentation: Bật tách từ (cần underthesea)
            enable_confidence_filter: Bật lọc chuỗi rác
        """
        print("\n" + "=" * 50)
        print("KHỞI TẠO NLP POST-PROCESSOR")
        print("=" * 50)
        
        # 1. Text Normalizer (luôn bật)
        self.normalizer = VietnameseTextNormalizer()
        
        # 2. Diacritics Handler (luôn bật) 
        self.diacritics_handler = VietnameseDiacriticsHandler()
        
        # 3. Spell Corrector
        self.enable_spell_correction = enable_spell_correction
        self.spell_corrector = None
        if enable_spell_correction:
            self.spell_corrector = VietnameseSpellCorrector(dictionary_paths)
        
        # 4. Word Segmenter
        self.enable_word_segmentation = enable_word_segmentation
        self.word_segmenter = None
        if enable_word_segmentation:
            self.word_segmenter = VietnameseWordSegmenter()
        
        # 5. Confidence Filter
        self.enable_confidence_filter = enable_confidence_filter
        self.confidence_filter = ConfidenceFilter() if enable_confidence_filter else None
        
        print("=" * 50)
        print("NLP POST-PROCESSOR SẴN SÀNG!")
        print("=" * 50 + "\n")

    def process(self, raw_text: str, verbose: bool = False) -> str:
        """
        Chạy toàn bộ pipeline NLP trên text thô từ OCR.
        
        Args:
            raw_text: Text thô từ mô hình CRNN
            verbose: In chi tiết từng bước xử lý
            
        Returns:
            Text đã được chuẩn hóa và sửa lỗi
        """
        if not raw_text or not raw_text.strip():
            return raw_text

        text = raw_text
        
        if verbose:
            print(f"\n--- NLP POST-PROCESSING ---")
            print(f"[Input]       : '{text}'")

        # Bước 1: Chuẩn hóa Unicode & text
        text = self.normalizer.normalize(text)
        if verbose:
            print(f"[Normalized]  : '{text}'")

        # Bước 2: Sửa dấu tiếng Việt
        text = self.diacritics_handler.fix_common_diacritic_errors(text)
        if verbose:
            print(f"[Diacritics]  : '{text}'")

        # Bước 3: Lọc chuỗi rác
        if self.enable_confidence_filter and self.confidence_filter:
            text = self.confidence_filter.filter_garbage(text)
            if verbose:
                print(f"[Filtered]    : '{text}'")

        # Bước 4: Sửa lỗi chính tả
        if self.enable_spell_correction and self.spell_corrector:
            text = self.spell_corrector.correct_text(text)
            if verbose:
                print(f"[Spell Fixed] : '{text}'")

        # Bước 5: Tách từ (tùy chọn)
        if self.enable_word_segmentation and self.word_segmenter:
            text = self.word_segmenter.segment(text)
            if verbose:
                print(f"[Segmented]   : '{text}'")
        
        # Chuẩn hóa lần cuối
        text = self.normalizer.normalize_whitespace(text)
        
        if verbose:
            print(f"[Output]      : '{text}'")
            print(f"--- END ---\n")

        return text

    def process_batch(self, texts: List[str], verbose: bool = False) -> List[str]:
        """Xử lý một batch nhiều text cùng lúc."""
        return [self.process(text, verbose=verbose) for text in texts]


# ============================================================
# TIỆN ÍCH: Tạo instance nhanh
# ============================================================

def create_postprocessor(project_root: str = ".") -> NLPPostProcessor:
    """
    Factory function để tạo NLPPostProcessor với cấu hình mặc định.
    Tự động tìm và load các file từ điển trong data/dataset/.
    
    Args:
        project_root: Thư mục gốc của project
        
    Returns:
        NLPPostProcessor instance
    """
    # Sử dụng 2 file từ điển có sẵn trong dataset
    dict_paths = [
        os.path.join(project_root, "data", "dataset", "general_dict.txt"),   # ~22K từ vựng tiếng Việt chuẩn
        os.path.join(project_root, "data", "dataset", "vn_dictionary.txt"),  # ~30K từ thực tế từ dataset OCR
    ]
    
    # Chỉ giữ các file tồn tại
    existing_paths = [p for p in dict_paths if os.path.exists(p)]
    
    if not existing_paths:
        print(f"[NLP Post-Process] Cảnh báo: Không tìm thấy từ điển trong {project_root}/data/dataset/")
    
    return NLPPostProcessor(
        dictionary_paths=existing_paths,
        enable_spell_correction=True,
        enable_word_segmentation=True,
        enable_confidence_filter=True,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST NLP POST-PROCESSING CHO OCR TIẾNG VIỆT")
    print("=" * 60)
    
    # Tạo post-processor
    processor = create_postprocessor(".")
    
    # Các mẫu test mô phỏng lỗi OCR
    test_cases = [
        "Việệt  Nam",                          # Lỗi ký tự lặp + khoảng trắng thừa
        "ngườii  Việtt  Nam",                   # Lỗi ký tự lặp
        "trường   học   Bách    Khoa",          # Khoảng trắng thừa
        "Hà   Nộii",                            # Lỗi dấu + khoảng trắng
        "xiin  chàoo",                          # Lỗi chính tả
        "bbbkkk  lllmmm",                       # Chuỗi rác
        "Thành phố Hồ Chí Minh",               # Text đúng (không cần sửa)
        "123 Nguyễn Văn Cừ",                    # Text có số
    ]
    
    print("\nKết quả xử lý:")
    print("-" * 60)
    for text in test_cases:
        result = processor.process(text, verbose=True)
        print(f"  '{text}'")
        print(f"  → '{result}'")
        print()
