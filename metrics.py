import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import conv2d
from math import exp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import os

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
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    batch_size, channel, height, width = img1.shape

    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    padding = window_size // 2

    mu1 = conv2d(img1, window, padding=padding, groups=channel)
    mu2 = conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim_numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_numerator / ssim_denominator

    return ssim_map.mean().item()


def evaluate_steganogan(model, dataloader, device="cuda"):
    model.eval()
    results = {"accuracy": [], "rs_bpp": [], "psnr": [], "ssim": []}

    with torch.no_grad():
        for cover_images, _ in dataloader:
            batch_size = cover_images.size(0)
            # input image
            cover_images = cover_images.to(device)
            normalized_cover_images = model.convert_image([cover_images])[0]

            data = torch.bernoulli(
                torch.ones(
                    batch_size,
                    model.data_depth,
                    cover_images.size(2),
                    cover_images.size(3),
                )
                * 0.5
            ).to(device)

            # output image
            stego_images, decoded_data = model(normalized_cover_images, data)
            # unconvertinng image not sure if its right
            stego_images = ((stego_images + 1.0) * 127.5) / 255.0

            bit_accuracy = (
                (decoded_data > 0).float().eq(data).float().mean(dim=[1, 2, 3])
            )

            for i in range(batch_size):
                # correct
                acc = bit_accuracy[i].item()
                results["accuracy"].append(acc)
                # correct
                rs_bpp = calculate_rs_bpp(acc, model.data_depth)
                results["rs_bpp"].append(rs_bpp)
                # not sure if its correct since cover_images are not normalized?
                psnr = calculate_psnr(cover_images[i : i + 1], stego_images[i : i + 1])
                results["psnr"].append(psnr)
                # not sure if its correct since cover_images are not normalized?
                ssim = calculate_ssim(cover_images[i : i + 1], stego_images[i : i + 1])
                results["ssim"].append(ssim)

    avg_results = {
        "accuracy": np.mean(results["accuracy"]),
        "rs_bpp": np.mean(results["rs_bpp"]),
        "psnr": np.mean(results["psnr"]),
        "ssim": np.mean(results["ssim"]),
    }

    return avg_results


def load_model(model_type, data_depth, model_path, device):
    if model_type == "basic":
        model = BasicSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "residual":
        model = ResidualSteganoGAN(data_depth=data_depth, device=device)
    elif model_type == "dense":
        model = DenseSteganoGAN(data_depth=data_depth, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")

    return model


def evaluate_model_on_dataset(
    model_type, data_depth, model_path=None, dataset_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_type, data_depth, model_path, device)

    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_path and os.path.exists(dataset_path):
        from torchvision.datasets import ImageFolder

        print(f"Loading custom dataset from {dataset_path}")
        test_dataset = ImageFolder(root=dataset_path, transform=transform)
    else:
        print("Using CIFAR10 dataset")
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate the model
    metrics = evaluate_steganogan(model, test_dataloader, device=device)

    print(f"Model Type: {model_type.capitalize()}")
    print(f"Data Depth: {data_depth}")
    print(f"Bit Accuracy: {metrics['accuracy']:.4f}")
    print(f"Reed-Solomon BPP: {metrics['rs_bpp']:.4f}")
    print(f"PSNR: {metrics['psnr']:.2f}")
    print(f"SSIM: {metrics['ssim']:.4f}")

    return metrics


def visualize_examples(
    model, dataloader, num_examples=3, device="cuda", save_path=None
):
    model.eval()

    examples = []
    with torch.no_grad():
        for images, _ in dataloader:
            if len(examples) >= num_examples:
                break
            # input image
            images = images.to(device)
            normalized_images = model.convert_image([images])[0]

            batch_size = images.size(0)
            data = torch.bernoulli(
                torch.ones(
                    batch_size,
                    model.data_depth,
                    images.size(2),
                    images.size(3),
                )
                * 0.5
            ).to(device)
            # output image
            stego_images, decoded_data = model(normalized_images, data)
            # unconvertinng image not sure if its right
            stego_images = ((stego_images + 1.0) * 127.5) / 255.0

            for i in range(min(batch_size, num_examples - len(examples))):
                examples.append(
                    {
                        "cover": images[i].cpu(),
                        "stego": stego_images[i].cpu(),
                        "data": data[i].cpu(),
                        "decoded": decoded_data[i].cpu(),
                    }
                )

    fig, axes = plt.subplots(num_examples, 4, figsize=(12, 3 * num_examples))

    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i, example in enumerate(examples):
        cover_img = example["cover"].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(cover_img, 0, 1))
        axes[i, 0].set_title("Cover Image")
        axes[i, 0].axis("off")
        # not sure if its right
        stego_img = example["stego"].permute(1, 2, 0).numpy()
        axes[i, 1].imshow(np.clip(stego_img, 0, 1))
        psnr = calculate_psnr(
            example["cover"].unsqueeze(0), example["stego"].unsqueeze(0)
        )
        axes[i, 1].set_title(f"Stego Image (PSNR: {psnr:.2f})")
        axes[i, 1].axis("off")
        # correct
        data_img = example["data"][0].numpy()
        axes[i, 2].imshow(data_img, cmap="gray")
        axes[i, 2].set_title("Hidden Data")
        axes[i, 2].axis("off")
        # correct
        decoded = (example["decoded"][0] > 0).float().numpy()
        axes[i, 3].imshow(decoded, cmap="gray")
        acc = (decoded == example["data"][0].numpy()).mean()
        axes[i, 3].set_title(f"Decoded Data (Acc: {acc:.2f})")
        axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SteganoGAN models")

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["basic", "residual", "dense"],
        default="dense",
        help="Type of SteganoGAN model to evaluate",
    )

    parser.add_argument(
        "--data_depth",
        type=int,
        default=1,
        help="Data depth of the model (bits per pixel)",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the .pth model file to load",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a custom dataset directory (uses CIFAR10 if not provided)",
    )

    parser.add_argument(
        "--run_tests", action="store_true", help="Run unit tests for the metrics"
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Visualize examples of steganography"
    )

    parser.add_argument(
        "--num_examples", type=int, default=3, help="Number of examples to visualize"
    )

    parser.add_argument(
        "--viz_output",
        type=str,
        default=None,
        help="Path to save visualization output (shows on screen if not provided)",
    )

    args = parser.parse_args()

    print(
        f"\nEvaluating {args.model_type.capitalize()} SteganoGAN model with data depth {args.data_depth}..."
    )
    metrics = evaluate_model_on_dataset(
        args.model_type, args.data_depth, args.model_path, args.dataset_path
    )

    if args.visualize:
        transform = transforms.Compose([transforms.ToTensor()])

        if args.dataset_path and os.path.exists(args.dataset_path):
            from torchvision.datasets import ImageFolder

            print(f"Loading custom dataset from {args.dataset_path} for visualization")
            test_dataset = ImageFolder(root=args.dataset_path, transform=transform)
        else:
            print("Using CIFAR10 dataset for visualization")
            test_dataset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = load_model(args.model_type, args.data_depth, args.model_path, device)

        print(f"\nVisualizing {args.num_examples} examples...")
        visualize_examples(
            model,
            test_dataloader,
            num_examples=args.num_examples,
            device=device,
            save_path=args.viz_output,
        )

# Example usage: (download the testimg dataset for debugging)
# python metrics.py --model_type dense --data_depth 3 --model_path models/DenseSteganoGAN/3/epoch_32.pth --dataset_path testimg/ --visualize
