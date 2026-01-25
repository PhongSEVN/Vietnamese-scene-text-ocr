"""
model.py - CRNN Model for Vietnamese Scene Text Recognition

Architecture: CNN (feature extraction) + RNN (sequence modeling) + CTC (loss)

Backbone options:
- ResNet-based
- VGG-based (classic CRNN)
- MobileNetV2 (lightweight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM layer for sequence modeling.
    Maps input features to output features in both directions.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            bidirectional=True, 
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, input_size]
            
        Returns:
            Output tensor [B, T, output_size]
        """
        # LSTM output: [B, T, hidden*2]
        lstm_out, _ = self.lstm(x)
        
        # Linear projection
        output = self.linear(lstm_out)
        
        return output


class VGGFeatureExtractor(nn.Module):
    """
    VGG-style CNN feature extractor.
    Classic architecture for CRNN.
    
    Input: [B, 3, 32, W]
    Output: [B, 512, 1, W/4]
    """
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: [B, 3, 32, W] -> [B, 64, 16, W/2]
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: [B, 64, 16, W/2] -> [B, 128, 8, W/4]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: [B, 128, 8, W/4] -> [B, 256, 4, W/4]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Height only
            
            # Block 4: [B, 256, 4, W/4] -> [B, 512, 2, W/4]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Height only
            
            # Block 5: [B, 512, 2, W/4] -> [B, 512, 1, W/4]
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # 2x1 -> 1x1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.output_channels = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor.
    Uses residual connections for better gradient flow.
    
    Input: [B, 3, 32, W]
    Output: [B, 512, 1, W/4]
    """
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 128, stride=(2, 2))    # 32 -> 16
        self.layer2 = self._make_layer(128, 256, stride=(2, 2))   # 16 -> 8
        self.layer3 = self._make_layer(256, 512, stride=(2, 1))   # 8 -> 4
        self.layer4 = self._make_layer(512, 512, stride=(2, 1))   # 4 -> 2
        
        # Final pool to reduce height to 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.output_channels = 512
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    stride: Tuple[int, int]) -> nn.Sequential:
        """Create a residual layer."""
        downsample = None
        if stride != (1, 1) or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride, downsample),
            ResidualBlock(out_channels, out_channels, (1, 1), None),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block with 2 conv layers."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: Tuple[int, int], downsample: nn.Module = None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CRNN(nn.Module):
    """
    CRNN: Convolutional Recurrent Neural Network for Scene Text Recognition
    
    Architecture:
    1. CNN: Extract visual features
    2. Map-to-Sequence: Convert 2D features to 1D sequence
    3. RNN: Model sequential dependencies
    4. Output: Character probability distribution
    
    Input: [B, 3, H, W] image
    Output: [T, B, num_classes] log probabilities for CTC
    """
    
    def __init__(self,
                 num_classes: int,
                 img_height: int = 32,
                 img_width: int = 128,
                 hidden_size: int = 256,
                 backbone: str = 'vgg'):
        """
        Initialize CRNN model.
        
        Args:
            num_classes: Number of output classes (charset size + blank)
            img_height: Input image height
            img_width: Input image width
            hidden_size: LSTM hidden size
            backbone: 'vgg' or 'resnet'
        """
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        
        # Feature extractor (CNN)
        if backbone == 'resnet':
            self.cnn = ResNetFeatureExtractor(input_channels=3)
        else:
            self.cnn = VGGFeatureExtractor(input_channels=3)
        
        cnn_output_channels = self.cnn.output_channels
        
        # Sequence modeling (RNN)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(cnn_output_channels, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Log probabilities [T, B, num_classes] for CTC loss
        """
        # CNN: [B, C, H, W] -> [B, channels, 1, W']
        features = self.cnn(x)
        
        # Squeeze height dimension: [B, channels, 1, W'] -> [B, channels, W']
        features = features.squeeze(2)
        
        # Transpose for RNN: [B, channels, W'] -> [B, W', channels]
        features = features.permute(0, 2, 1)
        
        # RNN: [B, T, hidden] -> [B, T, num_classes]
        output = self.rnn(features)
        
        # Transpose for CTC: [B, T, classes] -> [T, B, classes]
        output = output.permute(1, 0, 2)
        
        # Log softmax for CTC
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def get_output_length(self, input_width: int) -> int:
        """
        Calculate output sequence length given input width.
        Due to pooling operations, the sequence length changes.
        """
        # VGG: width / 4 after all pooling
        # ResNet: similar
        return input_width // 4


class CTCLoss(nn.Module):
    """
    Wrapper for CTC Loss with proper handling of blank index.
    """
    
    def __init__(self, blank: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
    
    def forward(self, 
                log_probs: torch.Tensor,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate CTC loss.
        
        Args:
            log_probs: [T, B, C] log probabilities
            targets: [B, S] target sequences (padded)
            input_lengths: [B] length of each input sequence
            target_lengths: [B] length of each target sequence
            
        Returns:
            CTC loss value
        """
        # Flatten targets for CTC
        targets_flat = []
        for i, length in enumerate(target_lengths):
            targets_flat.extend(targets[i, :length].tolist())
        targets_flat = torch.IntTensor(targets_flat)
        
        if log_probs.is_cuda:
            targets_flat = targets_flat.cuda()
        
        return self.ctc_loss(log_probs, targets_flat, 
                            input_lengths.int(), target_lengths.int())


# Test model
if __name__ == "__main__":
    # Test CRNN
    batch_size = 4
    img_height = 32
    img_width = 128
    num_classes = 200  # Charset size
    
    model = CRNN(
        num_classes=num_classes,
        img_height=img_height,
        img_width=img_width,
        hidden_size=256,
        backbone='vgg'
    )
    
    print(f"Model: CRNN")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, 3, img_height, img_width)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # [T, B, num_classes]
    print(f"Sequence length: {output.shape[0]}")
    print(f"Expected seq length: {model.get_output_length(img_width)}")
    
    # Test CTC loss
    criterion = CTCLoss(blank=0)
    
    # Dummy targets
    target_lengths = torch.IntTensor([5, 4, 6, 3])
    targets = torch.randint(1, num_classes, (batch_size, max(target_lengths)))
    input_lengths = torch.IntTensor([output.shape[0]] * batch_size)
    
    loss = criterion(output, targets, input_lengths, target_lengths)
    print(f"\nCTC Loss: {loss.item():.4f}")
