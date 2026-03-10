import torch.nn as nn
from models.base import BidirectionalLSTM


class CRNN(nn.Module):
    def __init__(self, img_h, nc, n_class, nh, n_rnn=2):
        super(CRNN, self).__init__()

        # Input: [Batch, 1, 32, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),           # 16x64
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),           # 8x32
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 4x33
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 2x34
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)          # 1x33
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, n_class)
        )

    def forward(self, input):
        conv = self.cnn(input)          # [B, 512, 1, W]
        b, c, h, w = conv.size()
        assert h == 1, "The height of conv features must be 1"
        conv = conv.squeeze(2)          # [B, 512, W]
        conv = conv.permute(2, 0, 1)    # [T, B, 512]
        output = self.rnn(conv)
        return output
