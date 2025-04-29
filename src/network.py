import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from src.decoders import BasicDecoder, DenseDecoder
from src.critics import BasicCritic


class BasicSteganoGAN(nn.Module):
    def __init__(self, data_depth=1, device=None):
        super(BasicSteganoGAN, self).__init__()
        self.data_depth = data_depth
        self.encoder = BasicEncoder(data_depth)
        self.decoder = DenseDecoder(data_depth)
        self.critic = BasicCritic()
        self.name = "BasicSteganoGAN"

        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)
            self.critic.to(device)

    # Convert text to (B, D, H, W)
    def convert_text(self, text):
        # TODO: Implement the conversion of text to (B, D, H, W)
        pass

    # Convert (B, D, H, W) to text
    def unconvert_text(self, text):
        # TODO: Implement the conversion of (B, D, H, W) to text
        pass

    def random_data(self, images):
        data = []
        for image in images:
            data.append(torch.rand((self.data_depth, image.shape[-2], image.shape[-1]), device=image.device) > 0.5)
        return data

    # Assumes that the image and data are already converted
    def forward(self, images, data):
        generated = self.encoder(images, data)
        return generated, self.decoder(generated)

    # Assumes that the image and data are already converted
    def critic_score(self, images):
        return self.critic(images)

    # Does not assume that the data is already converted
    def encode(self, image, text):
        bits = self.convert_text(text)
        return self.encoder(image, bits)

    # Assumes that the data is already converted
    def decode(self, image):
        decoded = self.decoder(image)
        return self.unconvert_text(decoded)


class ResidualSteganoGAN(BasicSteganoGAN):
    def __init__(self, data_depth=1, device=None):
        super(ResidualSteganoGAN, self).__init__()
        self.data_depth = data_depth
        self.encoder = ResidualEncoder(data_depth)
        self.decoder = DenseDecoder(data_depth)
        self.critic = BasicCritic()
        self.name = "ResidualSteganoGAN"

        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)
            self.critic.to(device)


class DenseSteganoGAN(BasicSteganoGAN):
    def __init__(self, data_depth=1, device=None):
        super(DenseSteganoGAN, self).__init__()
        self.data_depth = data_depth
        self.encoder = DenseEncoder(data_depth)
        self.decoder = DenseDecoder(data_depth)
        self.critic = BasicCritic()
        self.name = "DenseSteganoGAN"

        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)
            self.critic.to(device)
