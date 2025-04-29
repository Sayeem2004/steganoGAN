import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

from src.network import DenseSteganoGAN, BasicSteganoGAN, ResidualSteganoGAN


def load_model(model_type, data_depth, model_path, device):
    if model_type == "basic": model = BasicSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "residual": model = ResidualSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "dense": model = DenseSteganoGAN(data_depth=data_depth, device=device)
    else: raise ValueError(f"Unknown model type: {model_type}")

    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    return model


def run_model(model_type, data_depth, model_path, image_path, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, data_depth, model_path, device)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(Image.open(image_path).convert("RGB"))
    
    encoded = model.encode(image, text)
    decoded = model.decode(encoded)
    return encoded, decoded

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SteganoGAN models")
    parser.add_argument("--model_type", type=str, choices=["basic", "residual", "dense"], default="dense", help="Type of SteganoGAN model to evaluate.")
    parser.add_argument("--data_depth", type=int, default=1, help="Data depth of the model (bits per pixel).")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the .pth model file to load.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a custom dataset directory.")
    parser.add_argument("--text", type=str, default="Hello World!", help="Text to encode into image.")
    args = parser.parse_args()
   
    encoded, decoded = run_model(args.model_type, args.data_depth, args.model_path, args.image_path, args.text)