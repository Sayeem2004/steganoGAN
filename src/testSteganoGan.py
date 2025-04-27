import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

# Import your model
from SteganoGan import SteganoGAN  # Ensure 'SteganoGAN.py' exists in the same directory or is in the Python path

# --------------------
# Load Image from Filepath
# --------------------
def load_image(image_path, image_size=(128, 128)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Converts to (C, H, W) and normalizes [0,1]
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

# --------------------
# Save Tensor as Image
# --------------------
def save_tensor_as_image(tensor, save_path):
    tensor = tensor.squeeze(0).detach().cpu().clamp(0,1)  # (C, H, W)
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image.save(save_path)

# --------------------
# Main Testing Function
# --------------------
def main(image_path, save_encoded_path, save_recovered_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    data_depth = 1  # Number of secret data channels
    model = SteganoGAN(data_depth=data_depth).to(device)
    model.eval()  # Inference mode

    # Load image
    image = load_image(image_path).to(device)

    # Create dummy hidden data
    batch_size, _, W, H = image.shape
    hidden_data = torch.rand(batch_size, data_depth, W, H).to(device)  # Random secret data

    # Forward pass
    with torch.no_grad():
        encoded_image, recovered_data, critic_score = model(image, hidden_data)

    # Save encoded image
    save_tensor_as_image(encoded_image, save_encoded_path)

    # Save recovered data visualization (normalize for visibility)
    recovered_vis = (recovered_data - recovered_data.min()) / (recovered_data.max() - recovered_data.min())
    save_tensor_as_image(recovered_vis, save_recovered_path)

    print(f"Encoded image saved to: {save_encoded_path}")
    print(f"Recovered data visualization saved to: {save_recovered_path}")

# --------------------
# Example Usage
# --------------------
if __name__ == "__main__":
    image_path = "sampleImage.jpg"  # Path to your test image
    save_encoded_path = "encoded_output.png"
    save_recovered_path = "recovered_output.png"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found!")

    main(image_path, save_encoded_path, save_recovered_path)
