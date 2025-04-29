import os
import torch
import argparse
import numpy as np
from math import exp
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn.functional import conv2d
from torch.utils.data import DataLoader
from torchvision import transforms

from src.network import DenseSteganoGAN, BasicSteganoGAN, ResidualSteganoGAN


def calculate_rs_bpp(bit_accuracy, data_depth):
    ratio = max(0, 2 * bit_accuracy - 1)
    rs_bpp = ratio * data_depth
    return rs_bpp


def calculate_psnr(cover_image, stego_image):
    mse = torch.mean((cover_image - stego_image) ** 2)
    if mse < 1e-10: return float("inf")
    max_pixel_value = 1.0
    psnr = 20 * torch.log10(torch.tensor(max_pixel_value)) - 10 * torch.log10(mse)
    return psnr.item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(img1, img2, window_size=11):
    if img1.shape != img2.shape: raise ValueError("Input images must have the same dimensions")
    _, channel, _, _ = img1.shape
    window = create_window(window_size, channel)

    if img1.is_cuda: window = window.cuda(img1.get_device())
    window  = window.type_as(img1)
    padding = window_size // 2

    mu1 = conv2d(img1, window, padding=padding, groups=channel)
    mu2 = conv2d(img2, window, padding=padding, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12   = conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    dynamic_range = img1.max() - img1.min()
    C1 = (0.01 + dynamic_range) ** 2
    C2 = (0.03 + dynamic_range) ** 2

    ssim_numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map         = ssim_numerator / ssim_denominator
    return ssim_map.mean().item()


def evaluate_steganogan(model, dataloader):
    results = {"accuracy": [], "rs_bpp": [], "psnr": [], "ssim": []}
    model.eval()

    with torch.no_grad():
        for cover_image in tqdm(dataloader, desc="Network Metrics", leave=True):
            cover_image = model.convert_image([cover_image])[0]
            data        = model.random_data([cover_image])[0].unsqueeze(0)

            stego_image, decoded_data = model(cover_image, data)
            acc = (decoded_data > 0).float().eq(data).float().mean().item()

            results["accuracy"].append(acc)
            results["rs_bpp"].append(calculate_rs_bpp(acc, model.data_depth))
            results["psnr"].append(calculate_psnr(cover_image, stego_image))
            results["ssim"].append(calculate_ssim(cover_image, stego_image))

    avg_results = {
        "accuracy": np.mean(results["accuracy"]),
        "rs_bpp": np.mean(results["rs_bpp"]),
        "psnr": np.mean(results["psnr"]),
        "ssim": np.mean(results["ssim"]),
    }
    return avg_results


def load_model(model_type, data_depth, model_path, device):
    if model_type == "basic":      model = BasicSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "residual": model = ResidualSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "dense":    model = DenseSteganoGAN(data_depth=data_depth, device=device)
    else: raise ValueError(f"Unknown model type: {model_type}")

    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    return model


def evaluate_model_on_dataset(model_type, data_depth, model_path=None, dataset_path=None):
    # Obtain the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, data_depth, model_path, device)
    transform = transforms.Compose([transforms.ToTensor()])

    # Load testing images from the directory
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png'))]
    images      = [transform(Image.open(img).convert("RGB")) for img in image_files]
    dataloader  = DataLoader(images, batch_size=1, shuffle=False)

    # Evaluate the model on the dataset
    metrics = evaluate_steganogan(model, dataloader)
    print(f"Model Type: {model_type.capitalize()}")
    print(f"Data Depth: {data_depth}")
    print(f"Bit Accuracy: {metrics['accuracy']:.4f}")
    print(f"Reed-Solomon BPP: {metrics['rs_bpp']:.4f}")
    print(f"PSNR: {metrics['psnr']:.2f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    return metrics, model, dataloader


def visualize_examples(model, dataloader, num_examples=3, save_path=None):
    examples = []
    model.eval()

    with torch.no_grad():
        for image in dataloader:
            if len(examples) >= num_examples: break
            image = model.convert_image([image])[0]
            data  = model.random_data([image])[0].unsqueeze(0)
            print(image.shape, data.shape)

            encoded_image, decoded_data = model(image, data)
            normal_image  = model.unconvert_image(encoded_image)[0]

            examples.append({
                "cover": image.squeeze(0).cpu(),
                "stego": normal_image.squeeze(0).cpu(),
                "data": data.squeeze(0).cpu(),
                "decoded": decoded_data.squeeze(0).cpu()
            })

    _, axes = plt.subplots(num_examples, 4, figsize=(12, 3 * num_examples))
    if num_examples == 1: axes = axes.reshape(1, -1)

    for i, example in enumerate(examples):
        cover_img = example["cover"].permute(1, 2, 0).numpy() # C x H x W -> H x W x C
        axes[i, 0].imshow(np.clip(cover_img, 0, 1))
        axes[i, 0].set_title("Cover Image")
        axes[i, 0].axis("off")

        stego_img = example["stego"].permute(1, 2, 0).numpy() # C x H x W -> H x W x C
        psnr      = calculate_psnr(example["cover"].unsqueeze(0), example["stego"].unsqueeze(0))
        axes[i, 1].imshow(np.clip(stego_img, 0, 1))
        axes[i, 1].set_title(f"Stego Image (PSNR: {psnr:.2f})")
        axes[i, 1].axis("off")

        data = example["data"][0].numpy()
        axes[i, 2].imshow(data, cmap="gray")
        axes[i, 2].set_title("Hidden Data")
        axes[i, 2].axis("off")

        decoded = (example["decoded"][0] > 0).float().numpy()
        acc     = (decoded == example["data"][0].numpy()).mean()
        axes[i, 3].imshow(decoded, cmap="gray")
        axes[i, 3].set_title(f"Decoded Data (Acc: {acc:.2f})")
        axes[i, 3].axis("off")

    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    else: plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SteganoGAN models")
    parser.add_argument("--model_type", type=str, choices=["basic", "residual", "dense"], default="dense",
                        help="Type of SteganoGAN model to evaluate")
    parser.add_argument("--data_depth", type=int, default=1, help="Data depth of the model (bits per pixel)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the .pth model file to load")
    parser.add_argument("--dataset_path", type=str, default="Div2K_test_LR_unknown/X4/", help="Path to a custom dataset directory")

    parser.add_argument("--visualize", action="store_true", help="Visualize examples of steganography")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to visualize")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save visualization output")
    args = parser.parse_args()

    print(f"\nEvaluating {args.model_type.capitalize()} SteganoGAN model with data depth {args.data_depth}...")
    metrics, model, dataloader = evaluate_model_on_dataset(args.model_type, args.data_depth, args.model_path, args.dataset_path)

    if args.visualize:
        print(f"\nVisualizing {args.num_examples} examples...")
        visualize_examples(model, dataloader, num_examples=args.num_examples, save_path=args.save_path)
    print("\nEvaluation complete.")
