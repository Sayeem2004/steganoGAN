import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random, string

from metrics import calculate_psnr
from src.network import DenseSteganoGAN, BasicSteganoGAN, ResidualSteganoGAN
from src.network import LeakyDenseSteganoGAN, LeakyBasicSteganoGAN, LeakyResidualSteganoGAN


def load_model(model_type, data_depth, model_path, device):
    if model_type == "basic": model = BasicSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "residual": model = ResidualSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "dense": model = DenseSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "leaky_basic": model = LeakyBasicSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "leaky_residual": model = LeakyResidualSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "leaky_dense": model = LeakyDenseSteganoGAN(data_depth=data_depth, device=device)
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
    
    encoded, bits = model.encode(image, text)
    decoded = model.decode(encoded)
    return encoded, decoded

def evaluate_model_on_dataset(
    model_type, data_depth, model_path=None, dataset_path=None
):
    # Obtain the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, data_depth, model_path, device)
    transform = transforms.Compose([transforms.ToTensor()])

    # Load testing images from the directory
    image_files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.endswith((".png"))
    ]
    images = [transform(Image.open(img).convert("RGB")) for img in image_files]
    dataloader = DataLoader(images, batch_size=1, shuffle=False)
    return model, dataloader

def visualize_examples(
    model, dataloader, num_examples=3, save_path=None, model_type=None, data_depth=None, text="fail"
):
    examples = []
    model.eval()
    with torch.no_grad():
        for image in dataloader:
            text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=40))
            if len(examples) >= num_examples:
                break
            encoded_image, data = model.encode(image.squeeze(0), text)
            decoded_text = model.decode(encoded_image)

            examples.append({
                "cover": image.squeeze(0).cpu(),
                "stego": encoded_image.squeeze(0).cpu(),
                "psnr": calculate_psnr(image, encoded_image),
                "msg" : text,
                "acc": (decoded_text == text),
                "data": data.squeeze(0).cpu(),
                "decoded": decoded_text,
            })

    fig, axes = plt.subplots(num_examples, 5, figsize=(15, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    if model_type and data_depth:
        fig.suptitle(
            f"{model_type.capitalize()} SteganoGAN with Data Depth {data_depth}",
            fontsize=16,
        )

    for i, example in enumerate(examples):
        # 1. Cover Image
        cover_img = example["cover"].permute(1, 2, 0).numpy()  # C x H x W -> H x W x C
        axes[i, 0].imshow(cover_img)
        axes[i, 0].set_title("Cover Image")
        axes[i, 0].axis("off")

        # 2. Stego Image
        stego_img = example["stego"].permute(1, 2, 0).numpy()  # C x H x W -> H x W x C
        axes[i, 1].imshow(stego_img)
        axes[i, 1].set_title(f"Stego Image (PSNR: {example['psnr']:.2f})")
        axes[i, 1].axis("off")

        # 3. Distortion (new)
        distortion = torch.abs(example["stego"] - example["cover"])
        scaling_factor = 10
        distortion = torch.clamp(distortion * scaling_factor, 0, 1)
        distortion_img = distortion.permute(1, 2, 0).numpy()
        axes[i, 2].imshow(distortion_img)
        axes[i, 2].set_title("Distortion (|Stego - Cover|)")
        axes[i, 2].axis("off")
        # 4. Hidden Data
        data = example["data"][0].numpy()
        axes[i, 3].imshow(data, cmap="gray")
        axes[i, 3].set_title("Hidden Data")
        axes[i, 3].axis("off")

        # 5. Decoded Data
        # decoded = (example["decoded"][0] > 0).float().numpy()
        axes[i, 4].imshow(data, cmap="gray")
        axes[i, 4].set_title(f"Decoded {example['msg']} into {example['decoded']}(Acc: {example['acc']:.2f})")
        axes[i, 4].axis("off")

    plt.tight_layout(pad = 0.2, rect=[0.25,0.25,0.25,0.25])
    if save_path:
        if os.path.isdir(save_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{model_type}_depth{data_depth}_{timestamp}.png"
            full_path = os.path.join(save_path, filename)
        else:
            full_path = save_path
        plt.savefig(full_path)
        print(f"Visualization saved to {full_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SteganoGAN models")
    parser.add_argument("--model_type", type=str, choices=["basic", "residual", "dense", "leaky_basic", "leaky_residual", "leaky_dense"],
                        default="dense", help="Type of SteganoGAN model to evaluate.")
    parser.add_argument("--data_depth", type=int, default=1, help="Data depth of the model (bits per pixel).")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the .pth model file to load.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a custom dataset directory.")
    parser.add_argument("--text", type=str, default="Hello World!", help="Text to encode into image.")
    args = parser.parse_args()
   
    # encoded, decoded = run_model(args.model_type, args.data_depth, args.model_path, args.image_path, args.text)
    model, dataloader = evaluate_model_on_dataset(
        args.model_type, args.data_depth, args.model_path, args.image_path
    )
    visualize_examples(
            model,
            dataloader,
            num_examples=5,
            save_path=None,
            model_type=args.model_type,
            data_depth=args.data_depth,
        )

    encoded, decoded = run_model(args.model_type, args.data_depth, args.model_path, args.image_path, args.text)
