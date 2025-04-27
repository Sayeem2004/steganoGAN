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

        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)
            self.critic.to(device)

    # Normalize to [-1, 1]
    def convert_image(self, images):
        images = [image / 127.5 - 1.0 for image in images]
        images = [images.to(device=self.device) for images in images]
        return images

    # Denormalize to [0, 255]
    def unconvert_image(self, images):
        images = [(image + 1.0) * 127.5 for image in images]
        return images.clamp(0, 255).to(torch.uint8)

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
            data.append(torch.rand((self.data_depth, image.shape[1], image.shape[2]), device=image.device) > 0.5)
        return data

    # Assumes that the image and data are already converted
    def forward(self, images, data):
        generated = self.encoder(images, data)
        return generated, self.decoder(generated)

    # Assumes that the image and data are already converted
    def critic_score(self, images):
        x = self.critic(images)
        return x

    # Does not assume that the image and data are already converted
    def encode(self, image, text):
        scaled = self.convert_image(image)
        bits = self.convert_text(text)
        generated = self.encoder(scaled, bits)
        return self.unconvert_image(generated)

    # Assumes that the image and data are already converted
    def decode(self, image):
        decoded = self.decoder(image)
        return self.unconvert_image(decoded)


class ResidualSteganoGAN(BasicSteganoGAN):
    def __init__(self, data_depth=1, device=None):
        super(ResidualSteganoGAN, self).__init__()
        self.data_depth = data_depth
        self.encoder = ResidualEncoder(data_depth)
        self.decoder = DenseDecoder(data_depth)
        self.critic = BasicCritic()

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

        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)
            self.critic.to(device)
