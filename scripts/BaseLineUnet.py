import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class SimpleEncoder(nn.Module):
    """A very simple encoder with two convolutional layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = SimpleConvBlock(3, 64)  # Assuming input is RGB
        self.conv2 = SimpleConvBlock(64, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        return x

class SimpleDecoder(nn.Module):
    """A very simple decoder."""
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = SimpleConvBlock(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.final_conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
