import torch
import torch.nn as nn

from network.encoders import ConvBlock, LeakyConvBlock

# Ignored in the paper but included in the codebase
class BasicDecoder(nn.Module):
    def __init__(self, data_depth):
        super(BasicDecoder, self).__init__()
        self.initial = ConvBlock(3, 32) # (B, 3, W, H) -> (B, 32, W, H)
        self.block1  = ConvBlock(32, 32) # (B, 32, W, H) -> (B, 32, W, H)
        self.block2  = ConvBlock(32, 32) # (B, 32, W, H) -> (B, 32, W, H)
        self.final   = ConvBlock(32, data_depth, lastBlock=True) # (B, 32, W, H) -> (B, 3, W, H)

    def forward(self, image):
        x = self.block2(self.block1(self.initial(image)))
        return self.final(x)


class DenseDecoder(nn.Module):
    def __init__(self, data_depth):
        super(DenseDecoder, self).__init__()
        self.initial = ConvBlock(3, 32) # (B, 3, W, H) -> (B, 32, W, H)
        self.block1  = ConvBlock(32, 32) # (B, 32, W, H) -> (B, 32, W, H)
        self.block2  = ConvBlock(32*2, 32) # (B, 32*2, W, H) -> (B, 32, W, H)
        self.final   = ConvBlock(32*3, data_depth, lastBlock=True) # (B, 32*3, W, H) -> (B, 3, W, H)

    def forward(self, image):
        x1 = self.initial(image)
        x2 = self.block1(x1)
        x3 = self.block2(torch.cat([x1, x2], dim=1))
        x4 = self.final(torch.cat([x1, x2, x3], dim=1))
        return x4


class LeakyBasicDecoder(nn.Module):
    def __init__(self, data_depth):
        super(LeakyBasicDecoder, self).__init__()
        self.initial = LeakyConvBlock(3, 32) # (B, 3, W, H) -> (B, 32, W, H)
        self.block1  = LeakyConvBlock(32, 32) # (B, 32, W, H) -> (B, 32, W, H)
        self.block2  = LeakyConvBlock(32, 32) # (B, 32, W, H) -> (B, 32, W, H)
        self.final   = LeakyConvBlock(32, data_depth, lastBlock=True) # (B, 32, W, H) -> (B, 3, W, H)

    def forward(self, image):
        x = self.block2(self.block1(self.initial(image)))
        return self.final(x)


class LeakyDenseDecoder(nn.Module):
    def __init__(self, data_depth):
        super(LeakyDenseDecoder, self).__init__()
        self.initial = LeakyConvBlock(3, 32) # (B, 3, W, H) -> (B, 32, W, H)
        self.block1  = LeakyConvBlock(32, 32) # (B, 32, W, H) -> (B, 32, W, H)
        self.block2  = LeakyConvBlock(32*2, 32) # (B, 32*2, W, H) -> (B, 32, W, H)
        self.final   = LeakyConvBlock(32*3, data_depth, lastBlock=True) # (B, 32*3, W, H) -> (B, 3, W, H)

    def forward(self, image):
        x1 = self.initial(image)
        x2 = self.block1(x1)
        x3 = self.block2(torch.cat([x1, x2], dim=1))
        x4 = self.final(torch.cat([x1, x2, x3], dim=1))
        return x4
