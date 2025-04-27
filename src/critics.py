import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders import ConvBlock


class BasicCritic(nn.Module):
    def __init__(self):
        super(BasicCritic, self).__init__()
        self.initial = ConvBlock(3, 32)
        self.block1  = ConvBlock(32, 32)
        self.block2  = ConvBlock(32, 32)
        self.final   = ConvBlock(32, 1, lastBlock=True)

    def forward(self, x):
        x = self.final(self.block2(self.block1(self.initial(x))))
        return F.adaptive_avg_pool2d(x, (1, 1))
