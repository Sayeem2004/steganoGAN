import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO : think data depth is wrong in some places

# --------------------
# Convolutional Block
# stride 1, padding 'same', then leaky relu activation then batch norm
# if block is last one in network, don't activate nor batch norm
# --------------------

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, 
                 lastBlock=False):
        super(ConvBlock, self).__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        
        if not lastBlock:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


# --------------------
# Basic Encoder (b)
# --------------------
class BasicEncoder(nn.Module):
    def __init__(self, data_depth):
        super(BasicEncoder, self).__init__()
        self.initial = ConvBlock(3, 32)
        self.block1 = ConvBlock(32 + data_depth, 32)
        self.block2 = ConvBlock(32, 32)
        self.final = nn.Conv2d(32, 3, kernel_size=1)  # output back to image format (3 channels)
    
    def forward(self, image, data):
        # Concatenate input image and data along channel dimension
        x = self.initial(image)
        x = torch.cat([x, data], dim=1) # (batch_size, 32 + D, W, H)
        x = self.block1(x)
        x = self.block2(x)
        x = self.final(x)  # (batch_size, 3, W, H)
        return x

# --------------------
# Dense Encoder (d)
# --------------------

class DenseEncoder(nn.Module):
    def __init__(self, data_depth):
        super(DenseEncoder, self).__init__()
        self.initial = ConvBlock(3, 32)
        
        # Densely connected blocks
        self.block1 = ConvBlock(32 + data_depth, 32)  # input + initial output
        self.block2 = ConvBlock(32*2 + data_depth, 32)  # input + outputs from initial, block1
        self.block3 = ConvBlock(32*3 + data_depth, 32)  # input + outputs from initial, block1, block2
        
        # Final layer
        self.final = nn.Conv2d(32*4 + (3 + data_depth), 3, kernel_size=1)  # output back to 3 channels

    def forward(self, image, data):
        x = torch.cat([image, data], dim=1)  # (batch, 3 + data_depth, W, H)
        
        x0 = self.initial(x)
        x1 = self.block1(torch.cat([x, x0], dim=1))
        x2 = self.block2(torch.cat([x, x0, x1], dim=1))
        x3 = self.block3(torch.cat([x, x0, x1, x2], dim=1))
        
        output = self.final(torch.cat([x, x0, x1, x2, x3], dim=1))
        return output

# --------------------
# Decoder
# --------------------
class Decoder(nn.Module):
    def __init__(self, data_depth):
        super(Decoder, self).__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 32)
        self.block3 = ConvBlock(32*2, 32)
        self.final = nn.Conv2d(32*3, data_depth, kernel_size=1)  # output the hidden data
        
    def forward(self, x):
        a = self.block1(x)
        b = self.block2(a)
        c = self.block3(torch.cat([a,b], dim=1))
        x = self.final(torch.cat([a,b,c], dim=1))  # (batch_size, D, W, H)
        return x

# --------------------
# Critic
#needs adaptive mean pooling
# --------------------
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 32)
        self.block3 = ConvBlock(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)  # output single score per patch
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final(x)  # (batch_size, 1, W, H)
        return x.mean(dim=[1,2,3])  # mean over spatial dimensions -> single score per image

# --------------------
# Full SteganoGAN Model
# --------------------
class SteganoGAN(nn.Module):
    def __init__(self, data_depth=1):
        super(SteganoGAN, self).__init__()
        self.encoder = BasicEncoder(data_depth)
        self.decoder = Decoder(data_depth)
        self.critic = Critic()
        
    def forward(self, image, data):
        encoded_image = self.encoder(image, data)
        recovered_data = self.decoder(encoded_image)
        critic_score = self.critic(encoded_image)
        return encoded_image, recovered_data, critic_score
