import torch
import torch.nn as nn
import unicodedata

class ToneMarkerConverter:
    """
    Chuyển đổi văn bản tiếng Việt sang bộ mã hóa 105 ký tự (Tone-marker encoding).
    Bộ ký tự bao gồm:
    - 95 ký tự ASCII (space to ~)
    - 9 ký tự dấu thanh/bổ trợ: ˋ, ˊ, ﹒, ˀ, ˜, ˇ, ˆ, ˒, ‑
    """
    
    # Bộ 105 ký tự chuẩn từ paper VinText
    CTLABELS = [
        " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
        "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
        "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~",
        "ˋ", "ˊ", "﹒", "ˀ", "˜", "ˇ", "ˆ", "˒", "‑"
    ]
    
    # Quy tắc giải mã và mã hóa
    TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
    SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
    TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D‑", "d‑"]
    
    # Bảng nguyên âm để đặt dấu thanh
    DICTIONARY = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"
    
    def __init__(self, max_len=25):
        self.alphabet = self.CTLABELS
        self.dict = {char: i for i, char in enumerate(self.alphabet)}
        self.max_len = max_len
        self.groups = self._make_groups()
        
    def _make_groups(self):
        groups = []
        i = 0
        while i < len(self.DICTIONARY) - 5:
            group = [c for c in self.DICTIONARY[i : i + 6]]
            i += 6
            groups.append(group)
        return groups

    def parse_tone(self, word):
        """Chuyển chữ có dấu sang: gốc + dấu thanh ở cuối (ví dụ: Rồng -> Rongˋ)."""
        res = ""
        tone = ""
        for char in word:
            found_in_dict = False
            for group in self.groups:
                if char in group:
                    if tone == "":
                        tone = self.TONES[group.index(char)]
                    res += group[0]
                    found_in_dict = True
                    break
            if not found_in_dict:
                res += char
        res += tone
        return res

    def full_parse(self, word):
        """Chuyển chữ Việt sang định dạng Tone-marker encoding (vd: Rồng -> Roˆngˋ)."""
        word = self.parse_tone(word)
        res = ""
        for char in word:
            if char in self.SOURCES:
                res += self.TARGETS[self.SOURCES.index(char)]
            else:
                res += char
        return res

    def correct_tone_position(self, word):
        """Quy tắc chính tả đặt dấu thanh vào đúng vị trí nguyên âm chính."""
        # word ở đây là chuỗi đã bỏ dấu thanh ở cuối
        if len(word) < 1:
            return ""
            
        first_ord_char = ""
        second_order_char = ""
        for char in word:
            for group in self.groups:
                if char in group:
                    second_order_char = first_ord_char
                    first_ord_char = group[0]
                    
        if word[-1] == first_ord_char and second_order_char != "":
            # Trường hợp đặc biệt qu, gi
            pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
            for pair in pair_chars:
                if pair in word and second_order_char in ["u", "U", "i", "I"]:
                    return first_ord_char
            return second_order_char
        return first_ord_char

    def encode(self, text_list):
        """Mã hóa batch văn bản thành tensor indices."""
        # Chuyển đổi sang 105 ký tự định dạng encode
        encoded_texts = [self.full_parse(t) for t in text_list]
        
        batch_size = len(encoded_texts)
        # Tensor [Batch, MaxLen] điền ID của space hoặc padding
        # Ở đây dùng index của space là 0 nếu theo CTLABELS
        # Tuy nhiên Attention model thường cần SOS/EOS
        # Trong paper VinText, 0 là GO token
        
        indices = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        for i, text in enumerate(encoded_texts):
            for j, char in enumerate(text[:self.max_len]):
                if char in self.dict:
                    indices[i, j] = self.dict[char]
                else:
                    indices[i, j] = self.dict.get(' ', 0)
        return indices

    def decode(self, indices):
        """Giải mã tensor indices ngược lại thành văn bản (NFC)."""
        texts = []
        for i in range(indices.size(0)):
            char_list = []
            for idx in indices[i]:
                idx = idx.item()
                if idx < len(self.alphabet):
                    char_list.append(self.alphabet[idx])
                else:
                    break # EOS or invalid
            
            raw_text = "".join(char_list).strip()
            # Thực hiện decode dấu thanh
            decoded_text = self._decode_tone_marker(raw_text)
            texts.append(decoded_text)
        return texts

    def _decode_tone_marker(self, recognition):
        """Logic decode từ 105 ký tự về chữ Việt có dấu chuẩn."""
        # 1. Khôi phục ký tự bổ trợ (aˇ -> ă, aˆ -> â, ...)
        for char in self.TARGETS:
            recognition = recognition.replace(char, self.SOURCES[self.TARGETS.index(char)])
            
        if len(recognition) < 1:
            return recognition
            
        # 2. Đưa dấu thanh về đúng vị trí
        if recognition[-1] in self.TONES:
            tone = recognition[-1]
            if len(recognition) < 2:
                return recognition[:-1] # chỉ có dấu thanh?
                
            base_word = recognition[:-1]
            replace_char = self.correct_tone_position(base_word)
            
            if tone != "":
                for group in self.groups:
                    if replace_char in group:
                        # Thay thế ký tự không dấu bằng ký tự có dấu thanh tương ứng
                        recognition = base_word.replace(replace_char, group[self.TONES.index(tone)], 1)
                        break
            else:
                recognition = base_word
                
        return recognition

class AttentionLabelConverter(ToneMarkerConverter):
    """Mở rộng ToneMarkerConverter cho Attention model (với SOS token)."""
    def __init__(self, max_len=25):
        super().__init__(max_len)
        # SOS token thường là 0 (space trong CTLABELS)
        # Nếu muốn SOS riêng, cần n_class = 105
        
    def encode(self, text_list):
        # Attention model trong paper: target bắt đầu bằng 0 (space/GO)
        # Nhãn thực tế được dịch sang phải 1 đơn vị
        encoded_indices = super().encode(text_list)
        return encoded_indices
