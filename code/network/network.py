import torch
import torch.nn as nn
from collections import Counter

import network.reed as rsCrypto
from network.critics import BasicCritic
from network.decoders import DenseDecoder, LeakyDenseDecoder
from network.encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from network.encoders import LeakyBasicEncoder, LeakyResidualEncoder, LeakyDenseEncoder


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
        # Adding zero padding to help find message
        bits = rsCrypto.text_to_bits(text) + [0] * 32

        payload = bits
        # Duplicating message until all possible bits are hidden message
        while len(payload) < width * height * depth:
            payload += bits

        # Cutoff bits that don't fit within block size
        payload = payload[:width * height * depth]
        payload = torch.FloatTensor(payload).view(1, depth, height, width)
        return payload

    def unconvert_text(self, bits):
        candidates = Counter()
        byte_array = rsCrypto.bits_to_bytearray(bits)
        # Every recovered 0 byte consider as a delimeter for message candidate
        candidateMessages = byte_array.split(b'\x00\x00\x00\x00')

        for candidate in candidateMessages:
            candidate = rsCrypto.bytearray_to_text(bytearray(candidate))
            if candidate == False: continue
            candidates[candidate] += 1

        if len(candidates) == 0: return None
        # Return the candidate that was decrypted most often
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
        if image.ndim == 4:  # If batched input (B, C, H, W)
            _, _, H, W = image.shape
        elif image.ndim == 3:  # If single image input (C, H, W)
            _, H, W = image.shape
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # _, H, W = image.shape
        bits = self.convert_text(text, self.data_depth, W, H)
        generated = self.encoder(image.unsqueeze(0), bits)
        return generated, bits

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


class LeakyBasicSteganoGAN(BasicSteganoGAN):
    def __init__(self, data_depth=1, device=None):
        super(LeakyBasicSteganoGAN, self).__init__()
        self.data_depth = data_depth
        self.encoder = LeakyBasicEncoder(data_depth)
        self.decoder = LeakyDenseDecoder(data_depth)
        self.critic = BasicCritic()
        self.name = "LeakyBasicSteganoGAN"

        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)
            self.critic.to(device)


class LeakyResidualSteganoGAN(BasicSteganoGAN):
    def __init__(self, data_depth=1, device=None):
        super(LeakyResidualSteganoGAN, self).__init__()
        self.data_depth = data_depth
        self.encoder = LeakyResidualEncoder(data_depth)
        self.decoder = LeakyDenseDecoder(data_depth)
        self.critic = BasicCritic()
        self.name = "LeakyResidualSteganoGAN"

        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)
            self.critic.to(device)


class LeakyDenseSteganoGAN(BasicSteganoGAN):
    def __init__(self, data_depth=1, device=None):
        super(LeakyDenseSteganoGAN, self).__init__()
        self.data_depth = data_depth
        self.encoder = LeakyDenseEncoder(data_depth)
        self.decoder = LeakyDenseDecoder(data_depth)
        self.critic = BasicCritic()
        self.name = "LeakyDenseSteganoGAN"

        if device is not None:
            self.device = device
            self.encoder.to(device)
            self.decoder.to(device)
            self.critic.to(device)
