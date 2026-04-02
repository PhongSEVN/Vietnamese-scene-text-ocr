import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VGGBackbone(nn.Module):
    def __init__(self, d_model=256, pretrained=True, dropout=0.5, ss=None, ks=None):
        super().__init__()
        if ss is None:
            ss = [(2,2),(2,2),(2,1),(2,1),(1,1)]
        if ks is None:
            ks = [(2,2),(2,2),(2,1),(2,1),(1,1)]

        weights = "DEFAULT" if pretrained else None
        cnn = models.vgg19_bn(weights=weights)

        pool_idx = 0
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, nn.MaxPool2d):
                cnn.features[i] = nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, d_model, 1)

    def forward(self, x):
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv


class EfficientNetBackbone(nn.Module):
    def __init__(self, d_model=256, pretrained=True, dropout=0.5):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        cnn = efficientnet_b0(weights=weights)

        # Patch the first Conv2d stride: (2,2) -> (1,1) to preserve spatial resolution.
        # For input (B, 3, 32, 128), this gives features output (B, 1280, 2, 8)
        # instead of the default (B, 1280, 1, 4).
        cnn.features[0][0].stride = (1, 1)

        self.features = cnn.features  # output: (B, 1280, H', W')
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(1280, d_model, 1)

    def forward(self, x):
        conv = self.features(x)          # (B, 1280, H', W')
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)  # (B, d_model, H', W')
        conv = conv.transpose(-1, -2)    # (B, d_model, W', H')
        conv = conv.flatten(2)           # (B, d_model, W'*H')
        conv = conv.permute(-1, 0, 1)   # (seq_len, B, d_model)
        return conv


class LanguageTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_len=1024)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, trans_dropout, batch_first=False
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_key_padding_mask=None, src_key_padding_mask=None):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(src.device)
        src = self.pos_enc(src * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask.bool() if tgt_key_padding_mask is not None else None,
            src_key_padding_mask=src_key_padding_mask,
        )
        output = output.transpose(0, 1)
        return self.fc(output)

    def gen_nopeek_mask(self, length):
        return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)

    def forward_encoder(self, src, src_key_padding_mask=None):
        src = self.pos_enc(src * math.sqrt(self.d_model))
        return self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)

    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        return self.fc(output), memory


class TransformerOCR(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, max_seq_length=128,
                 pos_dropout=0.1, trans_dropout=0.1,
                 cnn_pretrained=True, cnn_dropout=0.5, ss=None, ks=None,
                 backbone='vgg'):
        super().__init__()
        if backbone == 'efficientnet':
            self.cnn = EfficientNetBackbone(d_model, cnn_pretrained, cnn_dropout)
        else:
            self.cnn = VGGBackbone(d_model, cnn_pretrained, cnn_dropout, ss, ks)
        self.transformer = LanguageTransformer(
            vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, max_seq_length, pos_dropout, trans_dropout
        )
        self.vocab_size = vocab_size

    def forward(self, img, tgt_input, tgt_key_padding_mask=None, src_key_padding_mask=None):
        src = self.cnn(img)
        return self.transformer(src, tgt_input, tgt_key_padding_mask, src_key_padding_mask)

    @torch.no_grad()
    def greedy_decode(self, img, max_len=128, sos_token=1, eos_token=2, src_key_padding_mask=None):
        import numpy as np
        self.eval()
        device = img.device
        batch_size = img.size(0)

        src = self.cnn(img)
        memory = self.transformer.forward_encoder(src, src_key_padding_mask)

        translated_sentence = [[sos_token] * batch_size]
        char_probs = [[1.0] * batch_size]

        for _ in range(max_len):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = self.transformer.forward_decoder(tgt_inp, memory)
            output = F.softmax(output, dim=-1).cpu()

            values, indices = torch.topk(output, 1)
            next_tokens = indices[:, -1, 0].tolist()
            next_probs = values[:, -1, 0].tolist()

            char_probs.append(next_probs)
            translated_sentence.append(next_tokens)

            if all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
                break
            del output

        translated_sentence = np.asarray(translated_sentence).T
        char_probs = np.asarray(char_probs).T
        char_probs_masked = np.multiply(char_probs, translated_sentence > 3)
        valid_counts = (char_probs_masked > 0).sum(-1)
        avg_probs = np.sum(char_probs_masked, axis=-1) / np.maximum(valid_counts, 1)

        return translated_sentence, avg_probs


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 128)
    model = EfficientNetBackbone(d_model=256)
    out = model(x)
    print("EfficientNet output:", out.shape)
    # Expected: (seq_len, 2, 256)
