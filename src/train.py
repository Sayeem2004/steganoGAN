import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms

from src.network import BasicSteganoGAN, ResidualSteganoGAN, DenseSteganoGAN
from PIL import Image


# Create a dataset class for the images and texts
class RenderDataset(torch.utils.data.Dataset):
    def __init__(self, images, texts):
        self.images = images
        self.texts = texts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]
        return image, text


# Train the encoder and decoder of the SteganoGAN
def update_forward(network, dataloader, optimizer):
    total_encode_mse      = 0
    total_decode_loss     = 0
    total_decode_acc      = 0
    total_generated_score = 0

    for images, data in tqdm(dataloader, desc="Network Training", leave=True):
        generated, decoded = network(images, data)
        encoder_mse        = F.mse_loss(generated, images)
        decoder_loss       = F.binary_cross_entropy_with_logits(decoded.float(), data.float())
        decoder_acc        = (decoded > 0).float().eq(data).float().mean()
        generated_score    = network.critic(generated)

        optimizer.zero_grad()
        (encoder_mse + decoder_loss + generated_score).backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.25)
        optimizer.step()

        total_encode_mse      += encoder_mse.item()
        total_decode_loss     += decoder_loss.item()
        total_decode_acc      += decoder_acc.item()
        total_generated_score += generated_score.item()

    total_encode_mse      /= len(dataloader)
    total_decode_loss     /= len(dataloader)
    total_decode_acc      /= len(dataloader)
    total_generated_score /= len(dataloader)

    print(f"Encode MSE: {total_encode_mse:.4f}, Decode Loss: {total_decode_loss:.4f}, "
          f"Decode Acc: {total_decode_acc:.4f}, Generated Score: {total_generated_score:.4f}")
    return total_encode_mse, total_decode_loss, total_decode_acc, total_generated_score


# Train the critic of the SteganoGAN
def update_critic(network, dataloader, optimizer):
    total_critic_loss = 0

    for images, data in tqdm(dataloader, desc="Critic Training", leave=True):
        generated       = network.forward(images, data)[0]
        image_score     = network.critic(images)
        generated_score = network.critic(generated)

        optimizer.zero_grad()
        (image_score - generated_score).backward(retain_graph=True)
        optimizer.step()

        for param in network.critic.parameters():
            param.data.clamp_(-0.1, 0.1)
        total_critic_loss += (image_score - generated_score).item()

    total_critic_loss /= len(dataloader)
    print(f"Critic Loss: {total_critic_loss:.4f}")
    return total_critic_loss


# Validate both the encoder, decoder, and critic of the SteganoGAN
def validate_both(network, dataloader):
    total_encode_mse      = 0
    total_decode_loss     = 0
    total_decode_acc      = 0
    total_generated_score = 0
    total_critic_loss     = 0

    total_ssim = 0
    total_psnr = 0
    total_bpp  = 0

    for images, data in tqdm(dataloader, desc="Both Validation", leave=True):
        generated, decoded = network(images, data)
        encoder_mse        = F.mse_loss(generated, images)
        decoder_loss       = F.binary_cross_entropy_with_logits(decoded, data)
        decoder_acc        = (decoded > 0).float().eq(data).float().mean()
        generated_score    = network.critic(generated)
        image_score        = network.critic(images)

        total_encode_mse      += encoder_mse.item()
        total_decode_loss     += decoder_loss.item()
        total_decode_acc      += decoder_acc.item()
        total_generated_score += generated_score.item()
        total_critic_loss     += (image_score - generated_score).item()

        # TODO: Calculate SSIM, PSNR, and BPP
        # total_ssim += ssim.item()
        # total_psnr += psnr.item()
        # total_bpp  += bpp.item()

    total_encode_mse      /= len(dataloader)
    total_decode_loss     /= len(dataloader)
    total_decode_acc      /= len(dataloader)
    total_generated_score /= len(dataloader)
    total_critic_loss     /= len(dataloader)

    # total_ssim /= len(dataloader)
    # total_psnr /= len(dataloader)
    # total_bpp  /= len(dataloader)

    print(f"Validation - Encode MSE: {total_encode_mse:.4f}, Decode Loss: {total_decode_loss:.4f}, "
          f"Decode Acc: {total_decode_acc:.4f}, Generated Score: {total_generated_score:.4f}, "
          f"Critic Loss: {total_critic_loss:.4f}")
    # print(f"SSIM: {total_ssim:.4f}, PSNR: {total_psnr:.4f}, BPP: {total_bpp:.4f}")
    return (total_encode_mse, total_decode_loss, total_decode_acc, total_generated_score,
            total_critic_loss, total_ssim, total_psnr, total_bpp)


def train_network(train_images, val_images, epochs=10):
    # Create the network and optimizer
    network             = DenseSteganoGAN(data_depth=1, device='cuda' if torch.cuda.is_available() else 'cpu')
    combined_parameters = list(network.encoder.parameters()) + list(network.decoder.parameters())
    forward_optimizer   = torch.optim.AdamW(combined_parameters, lr=0.0001)
    critic_optimizer    = torch.optim.AdamW(network.critic.parameters(), lr=0.0001)

    # Convert images and generate random data
    train_images  = network.convert_image(train_images)
    train_payload = network.random_data(train_images)
    val_images    = network.convert_image(val_images)
    val_payload   = network.random_data(val_images)

    # Create training and validation datasets
    val_data   = RenderDataset(val_images, val_payload)
    train_data = RenderDataset(train_images, train_payload)

    # Create data loaders
    val_loader   = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    # Training loop
    train_stats, eval_stats = [], []
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        network.train()
        encode_mse, decode_loss, decode_acc, generated_score = update_forward(network, train_loader, forward_optimizer)
        critic_loss = update_critic(network, train_loader, critic_optimizer)
        train_stats.append({
            "epoch": epoch + 1, "encode_mse": encode_mse, "decode_loss": decode_loss,
            "decode_acc": decode_acc, "generated_score": generated_score, "critic_loss": critic_loss
        })

        network.eval()
        encode_mse, decode_loss, decode_acc, generated_score, critic_loss, ssim, psnr, bpp = validate_both(network, val_loader)
        eval_stats.append({
            "epoch": epoch + 1, "encode_mse": encode_mse, "decode_loss": decode_loss, "decode_acc": decode_acc,
            "generated_score": generated_score, "critic_loss": critic_loss, "ssim": ssim, "psnr": psnr, "bpp": bpp
        })

        # Save the model
        torch.save(network.state_dict(), f"models/model_epoch_{epoch + 1}.pth")
        print(f"Model saved as model_epoch_{epoch + 1}.pth")

    return train_stats, eval_stats


if __name__ == "__main__":
    # Load training images from the directory
    train_image_dir   = "DIV2K_train_LR_bicubic/X4/"
    train_image_files = [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith(('.png'))]
    train_transform   = transforms.Compose([transforms.ToTensor()])
    train_images      = [train_transform(Image.open(img).convert("RGB")) for img in train_image_files]

    # Load validation images from the directory
    val_image_dir   = "DIV2K_valid_LR_bicubic/X4/"
    val_image_files = [os.path.join(val_image_dir, f) for f in os.listdir(val_image_dir) if f.endswith(('.png'))]
    val_transform   = transforms.Compose([transforms.ToTensor()])
    val_images      = [val_transform(Image.open(img).convert("RGB")) for img in val_image_files]

    # Train the network
    print("Starting training...")
    train_stats, eval_stats = train_network(train_images, val_images, epochs=10)
    print("Training completed.")

    # Save the training and evaluation losses
    torch.save(train_stats, "models/train_stats.pth")
    torch.save(eval_stats, "models/eval_stats.pth")
    print("Training and evaluation losses saved.")
