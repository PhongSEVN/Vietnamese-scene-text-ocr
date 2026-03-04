import torch
import torch.nn as nn
from torch.nn import functional as F
import random

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec) # [T*b, nOut]
        output = output.view(T, b, -1)
        return output

class Attention(nn.Module):
    """
    GRU-based Attention Decoder.
    Dựa trên paper VinText (dict-guided scene text recognition).
    """
    def __init__(self, hidden_size, output_size, max_len=25):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = 0.1
        self.max_len = max_len

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # Attention weights layer
        self.vat = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        """
        input: current token (batch_size)
        hidden: 1 x batch_size x hidden_size
        encoder_outputs: T x batch_size x hidden_size
        """
        embedded = self.embedding(input) # [B, H]
        embedded = self.dropout(embedded)

        batch_size = encoder_outputs.shape[1]

        # Calculate attention weights
        # hidden + encoder_outputs (broadcasting)
        alpha = hidden + encoder_outputs # [T, B, H]
        alpha = alpha.view(-1, alpha.shape[-1]) # [T*B, H]
        attn_weights = self.vat(torch.tanh(alpha)) # [T*B, 1]
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0)) # [B, 1, T]
        attn_weights = F.softmax(attn_weights, dim=2)

        # Apply attention
        attn_applied = torch.matmul(attn_weights, encoder_outputs.permute((1, 0, 2))) # [B, 1, H]

        if embedded.dim() == 1: # case batch 1
            embedded = embedded.unsqueeze(0)
            
        # Combine input and attention context
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1) # [B, 2*H]
        output = self.attn_combine(output).unsqueeze(0) # [1, B, H]

        output = F.relu(output)
        output, hidden = self.gru(output, hidden) # [1, B, H]

        output = F.log_softmax(self.out(output[0]), dim=1) # [B, output_size]
        return output, hidden, attn_weights

    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

class AttentionOCR(nn.Module):
    """
    Mô hình OCR nhận diện chữ với Attention Mechanism (Decoder).
    Gồm: CNN Backbone -> RNN Encoder -> Attention Decoder.
    """
    def __init__(self, img_h, nc, n_class, nh, n_rnn=2):
        super(AttentionOCR, self).__init__()
        
        # CNN Backbone tương tự CRNN nhưng phù hợp cho Attention
        # Input: [Batch,nc, 32, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 16x64
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 8x32
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 4x33
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 2x34
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True) # 1x33
        )

        # RNN Encoder (Feature extraction sequence)
        self.encoder = BidirectionalLSTM(512, nh, nh)
        
        # Attention Decoder
        self.n_class = n_class # 105 chars
        self.attention_decoder = Attention(nh, n_class)
        self.teach_prob = 0.5 #Xác xuất sử dụng Teacher Forcing trong huấn luyện

    def forward(self, input, targets=None):
        # 1. Feature Extraction (CNN)
        conv = self.cnn(input) # [B, 512, 1, W]
        b, c, h, w = conv.size()
        conv = conv.squeeze(2) # [B, 512, W]
        conv = conv.permute(2, 0, 1) # [W, B, 512]

        # 2. Sequence Encoding (RNN)
        encoder_outputs = self.encoder(conv) # [T, B, H]

        device = input.device
        batch_size = encoder_outputs.size(1)

        # 3. Attention Decoding
        # Nếu có targets, thực hiện decode để tính loss (trả về log_probs)
        if targets is not None:
            # target_variable: [B, Max_Len]
            # Thêm SOS token (0) vào đầu
            _sos = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
            target_variable = torch.cat((_sos, targets), dim=1)
            
            decoder_input = target_variable[:, 0]
            decoder_hidden = self.attention_decoder.initHidden(batch_size, device)
            
            outputs = []
            # decode theo độ dài nhãn
            for di in range(1, target_variable.shape[1]):
                decoder_output, decoder_hidden, _ = self.attention_decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                outputs.append(decoder_output.unsqueeze(0))
                
                # Teacher Forcing chỉ áp dụng khi training
                if self.training and random.random() < self.teach_prob:
                    decoder_input = target_variable[:, di]
                else:
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()
            
            # Trả về chuỗi log probabilities [T, B, n_class]
            return torch.cat(outputs, dim=0)
        else:
            # Inference mode: Không có targets, tự decode (trả về indices)
            max_len = self.attention_decoder.max_len
            decoder_input = torch.zeros(batch_size, device=device, dtype=torch.long)
            decoder_hidden = self.attention_decoder.initHidden(batch_size, device)
            
            decodes = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
            for di in range(max_len):
                decoder_output, decoder_hidden, _ = self.attention_decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze()
                decodes[:, di] = decoder_input
                
            return decodes # [B, MaxLen]
