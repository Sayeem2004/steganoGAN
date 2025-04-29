import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F

from src.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from src.decoders import BasicDecoder, DenseDecoder
from src.critics import BasicCritic
import src.reed as rsCrypto

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
    def convert_text(self, text, depth, width, height):
        bits = rsCrypto.text_to_bits(text) + [0] * 32

        payload = bits
        duplicateNum = 1
        while len(payload) < width * height * depth:
            duplicateNum += 1
            payload += bits

        payload = payload[:width * height * depth]
        payload = torch.FloatTensor(payload).view(1, depth, height, width)
        return payload
    
    def unconvert_text(self, bits):
        candidates = Counter()
        byte_array = rsCrypto.bits_to_bytearray(bits)
        candidateMessages = byte_array.split(b'\x00\x00\x00\x00')

        for candidate in candidateMessages:
            candidate = rsCrypto.bytearray_to_text(bytearray(candidate))
            if candidate: candidates[candidate] += 1

        if len(candidates) == 0: raise ValueError('Failed to find message.')
        candidate, _ = candidates.most_common(1)[0]
        return candidate

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
        _, H, W = image.shape
        bits = self.convert_text(text, self.data_depth, W, H)
        generated = self.encoder(image.unsqueeze(0), bits)
        return generated

    # Assumes that the data is already converted
    def decode(self, image):
        image = image.to(self.device)
        bits = self.decoder(image).view(-1) > 0
        bits = bits.data.int().cpu().numpy().tolist()
        return self.unconvert_text(bits)


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
