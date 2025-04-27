import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, lastBlock=False):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]

        # Paper specifies that the last block should not have batch norm or activation
        # They also specify LeakyReLu before BatchNorm, which is a bit unusual
        if not lastBlock:
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BasicEncoder(nn.Module):
    def __init__(self, data_depth):
        super(BasicEncoder, self).__init__()
        self.initial = ConvBlock(3, 32) # (B, 3, W, H) -> (B, 32, W, H)
        self.block1  = ConvBlock(32 + data_depth, 32) # (B, 32 + D, W, H) -> (B, 32, W, H)
        self.block2  = ConvBlock(32, 32) # (B, 32, W, H) -> (B, 32, W, H)
        self.final   = ConvBlock(32, 3, lastBlock=True) # (B, 32, W, H) -> (B, 3, W, H)

    def forward(self, image, data):
        x = torch.cat([self.initial(image), data], dim=1)
        x = self.final(self.block2(self.block1(x)))
        return F.tanh(x) # Within the codebase, but not in the paper


class ResidualEncoder(nn.Module):
    def __init__(self, data_depth):
        super(ResidualEncoder, self).__init__()
        self.initial = ConvBlock(3, 32) # (B, 3, W, H) -> (B, 32, W, H)
        self.block1  = ConvBlock(32 + data_depth, 32) # (B, 32 + D, W, H) -> (B, 32, W, H)
        self.block2  = ConvBlock(32, 32) # (B, 32, W, H) -> (B, 32, W, H)
        self.final   = ConvBlock(32, 3, lastBlock=True) # (B, 32 + D, W, H) -> (B, 3, W, H)

    def forward(self, image, data):
        x = torch.cat([self.initial(image), data], dim=1)
        x = self.final(self.block2(self.block1(x)))
        return x + image


class DenseEncoder(nn.Module):
    def __init__(self, data_depth):
        super(DenseEncoder, self).__init__()
        self.initial = ConvBlock(3, 32) # (B, 3, W, H) -> (B, 32, W, H)
        self.block1  = ConvBlock(32 + data_depth, 32)  # (B, 32 + D, W, H) -> (B, 32, W, H)
        self.block2  = ConvBlock(32*2 + data_depth, 32)  # (B, 32*2 + D, W, H) -> (B, 32, W, H)
        self.final   = ConvBlock(32*3 + data_depth, 3, lastBlock=True)  # (B, 32*3 + D, W, H) -> (B, 3, W, H)

    def forward(self, image, data):
        x1 = torch.cat([self.initial(image), data], dim=1) # image + data
        x2 = self.block1(x1)
        x3 = self.block2(torch.cat([x1, x2], dim=1)) # (image + data) + block1
        x4 = self.final(torch.cat([x1, x2, x3], dim=1)) # (image + data) + block1 + block2
        return x4 + image
