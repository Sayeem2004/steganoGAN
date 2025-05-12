import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random, string
from scipy import stats

from metrics import calculate_psnr
from src.network import DenseSteganoGAN, BasicSteganoGAN, ResidualSteganoGAN
from src.network import LeakyDenseSteganoGAN, LeakyBasicSteganoGAN, LeakyResidualSteganoGAN

#load model
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

#load model and data set
def load_model_and_dataset(
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

#visualize the distributions of accuracies of recovering messages using reedsolomon
def visualize_rsbpp(model, dataloader, save_path=None, model_type=None, data_depth=None, text="fail"):
    examples = []
    model.eval()
    with torch.no_grad():
        for image in dataloader:
            #text that is hidden in the image
            text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=40))
            encoded_image, data = model.encode(image.squeeze(0), text)
            decoded_image = (model.decoder(encoded_image).squeeze(0) > 0).float()
            #whether or not message was correctly decoded
            temp_acc = 1-torch.mean(torch.abs(data.squeeze(0)- decoded_image.squeeze(0))).item()
            examples.append({
                "msg" : text,
                "acc": temp_acc,
            })

    #list of 0,1s of whether the message in each image succesfully decoded
    accs = [d.get("acc") for d in examples if "acc" in d]
    mean = np.mean(accs)  # Mean (mu)
    std_dev = np.sqrt(np.var(accs))  # Standard deviation (sigma)
    x_axis = np.arange(0.655, 0.8, 0.001)

    pdf = stats.norm.pdf(x_axis, mean, std_dev)
    plt.plot(x_axis, pdf)
    plt.yticks([])
    std_lim = 1.96 # 95% CI
    low = mean-std_lim*std_dev
    high = mean+std_lim*std_dev
    padding = 0.5
    top = 30
    plt.fill_between(x_axis, pdf, where=(low < x_axis) & (x_axis < high))
    plt.text(low, top + padding, round(low, 2), ha='center', fontsize = 20)
    plt.text(high, top + padding, round(high, 2), ha='center', fontsize = 20)

    # Add labels and title
    plt.xlabel("Accuracy", fontsize = 20)
    plt.ylabel("Frequency", fontsize = 20)
    plt.title(f"Decode Accuracy (95% Interval, Mean={round(mean,3)}, Std Dev={round(std_dev,3)})")
    low_xs = [low, low]
    high_xs = [high, high]
    low_ys = [0, top]
    high_ys = [0, top]
    plt.plot(low_xs, low_ys, color = 'black')
    plt.plot(high_xs, high_ys, color = 'black')
    # Display the plot
    plt.show()
    if save_path:
        if os.path.isdir(save_path):
            filename = f"{model_type}_depth{data_depth}.png"
            full_path = os.path.join(save_path, filename)
        else:
            full_path = save_path
        plt.savefig(full_path)
        print(f"Visualization saved to {full_path}")
    else:
        plt.show()

#visualize decoded data/distortion for stego images
def visualize_minimized_StegoMessages(
    model, dataloader, num_examples=2, save_path=None, model_type=None, data_depth=None, text="fail"
):
    examples = []
    model.eval()
    with torch.no_grad():
        for image in dataloader:
            if len(examples) >= num_examples:
                break
            encoded_image, data = model.encode(image.squeeze(0), text)
            decoded = model.decoder(encoded_image)
            decoded_text = model.decode(encoded_image)
            decoded_text = "Failed" if decoded == None else decoded_text

            examples.append({
                "cover": image.squeeze(0).cpu(),
                "stego": encoded_image.squeeze(0).cpu(),
                "psnr": calculate_psnr(image, encoded_image),
                "msg" : text,
                "accuracy" : round((decoded > 0).float().eq(data).float().mean().item(), 3),
                "data": data.squeeze(0).cpu(),
                "decoded": decoded.squeeze(0).cpu(),
                "decoded_text": decoded_text,
            })
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 3 * num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    if model_type and data_depth:
        fig.suptitle(
            f"{model_type.capitalize()} SteganoGAN with Data Depth {data_depth}",
            fontsize=16,
        )

    for i, example in enumerate(examples):
        # 1. Cover Image
        size = 20
        cover_img = example["cover"].permute(1, 2, 0).numpy()  # C x H x W -> H x W x C
        axes[i, 0].imshow(cover_img)
        axes[i, 0].set_title("Cover Image", fontsize = size+10)
        axes[i, 0].axis("off")

        # 2. Stego Image
        stego_img = example["stego"].permute(1, 2, 0).numpy()  # C x H x W -> H x W x C
        axes[i, 1].imshow(stego_img)
        axes[i, 1].set_title(f"Stego Image ", fontsize = size+10)
        axes[i, 1].axis("off")

        # 3. Distortion (new)
        distortion = torch.abs(example["stego"] - example["cover"])
        scaling_factor = 10
        distortion = torch.clamp(distortion * scaling_factor, 0, 1)
        distortion_img = distortion.permute(1, 2, 0).numpy()
        axes[i, 2].imshow(distortion_img)
        axes[i, 2].set_title("Distortion (|Stego - Cover|)", fontsize = size)
        axes[i, 2].axis("off")
        result_text = f"'{example['decoded_text']}'"
        axes[i, 2].text(-0.715, -0.1, result_text, fontsize=26,va='top', transform=axes[i,2].transAxes, \
         ha='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout(pad = 0.2, rect=[0.25,0.25,0.25,0.25])
    if save_path:
        if os.path.isdir(save_path):
            filename = f"{model_type}_depth{data_depth}.png"
            full_path = os.path.join(save_path, filename)
        else:
            full_path = save_path
        plt.savefig(full_path)
        print(f"Visualization saved to {full_path}")
    else:
        plt.show()

#visualize distorition/hidden data/ decoded data for stego images
def visualize_StegoMessages(
    model, dataloader, num_examples=2, save_path=None, model_type=None, data_depth=None, text="fail"
):
    examples = []
    model.eval()
    with torch.no_grad():
        for image in dataloader:
            if len(examples) >= num_examples:
                break
            encoded_image, data = model.encode(image.squeeze(0), text)
            decoded = model.decoder(encoded_image)
            decoded_text = model.decode(encoded_image)
            decoded_text = "Failed" if decoded == None else decoded_text

            examples.append({
                "cover": image.squeeze(0).cpu(),
                "stego": encoded_image.squeeze(0).cpu(),
                "psnr": calculate_psnr(image, encoded_image),
                "msg" : text,
                "accuracy" : round((decoded > 0).float().eq(data).float().mean().item(), 3),
                "data": data.squeeze(0).cpu(),
                "decoded": decoded.squeeze(0).cpu(),
                "decoded_text": decoded_text,
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
        decoded = (example["decoded"][0] > 0).float().numpy()
        axes[i, 4].imshow(decoded, cmap="gray")
        result_text = f"'{example['decoded_text']}'"
        axes[i, 4].text(-0.1, -0.1, result_text, fontsize=26,va='top', transform=axes[i,4].transAxes, \
         ha='center', bbox=dict(facecolor='white', alpha=0.5))
        axes[i, 4].set_title(f"Decoded Message (Acc : {example['accuracy']})")
        axes[i, 4].axis("off")

    plt.tight_layout(pad = 0.2, rect=[0.25,0.25,0.25,0.25])
    if save_path:
        if os.path.isdir(save_path):
            filename = f"{model_type}_depth{data_depth}.png"
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
    parser.add_argument("--image_path", type=str, default="data/COCO_val_2017/", help="Path to a custom dataset directory.")
    parser.add_argument("--text", type=str, default="Hello World!", help="Text to encode into image.")
    parser.add_argument("--visualizer", type=str, choices=["rsbpp_accuracy", "message_decode_full", "message_decode_small"],
                        default="rsbpp_accuracy", help="Type of visualization to display")
    args = parser.parse_args()
   
    model, dataloader = load_model_and_dataset(
        args.model_type, args.data_depth, args.model_path, args.image_path
    )
    if args.visualizer == "rsbpp_accuracy":
        visualize_rsbpp(
                model,
                dataloader,
                save_path=None,
                model_type=args.model_type,
                data_depth=args.data_depth,
            )
    elif args.visualizer == "message_decode_full":
        visualize_StegoMessages(
                model,
                dataloader,
                num_examples=1,
                save_path=None,
                model_type=args.model_type,
                data_depth=args.data_depth,
                text=args.text
            )
    else:
        visualize_minimized_StegoMessages(
                model,
                dataloader,
                num_examples=1,
                save_path=None,
                model_type=args.model_type,
                data_depth=args.data_depth,
                text=args.text
            )

