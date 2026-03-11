import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
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


class LanguageTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, trans_dropout, batch_first=False
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_key_padding_mask=None):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(src.device)
        src = self.pos_enc(src * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        if tgt_key_padding_mask is not None:
            output = self.transformer(src, tgt, tgt_mask=tgt_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask.float())
        else:
            output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        output = output.transpose(0, 1)
        return self.fc(output)

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

    def forward_encoder(self, src):
        src = self.pos_enc(src * math.sqrt(self.d_model))
        return self.transformer.encoder(src)

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
                 cnn_pretrained=True, cnn_dropout=0.5, ss=None, ks=None):
        super().__init__()
        self.cnn = VGGBackbone(d_model, cnn_pretrained, cnn_dropout, ss, ks)
        self.transformer = LanguageTransformer(
            vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, max_seq_length, pos_dropout, trans_dropout
        )
        self.vocab_size = vocab_size

    def forward(self, img, tgt_input, tgt_key_padding_mask=None):
        src = self.cnn(img)
        return self.transformer(src, tgt_input, tgt_key_padding_mask)

    @torch.no_grad()
    def greedy_decode(self, img, max_len=128, sos_token=1, eos_token=2):
        import numpy as np
        self.eval()
        device = img.device
        batch_size = img.size(0)

        src = self.cnn(img)
        memory = self.transformer.forward_encoder(src)

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
