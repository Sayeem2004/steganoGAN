import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as mets
import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import metrics
from torch.utils.data import DataLoader


def get_model_and_dataloader(model_type, data_depth, model_path=None, dataset_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = metrics.load_model(model_type, data_depth, model_path, device)
    transform = transforms.Compose([transforms.ToTensor()])
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png') or f.endswith(('.jpg'))]
    images      = [transform(Image.open(img).convert("RGB")) for img in image_files]
    dataloader  = DataLoader(images, batch_size=1, shuffle=False)
    return model, dataloader
    

def save_images(model,dataloader,save_path="save_images",num_examples=3):
    """Saves images to a directory and returns the list of saved images.

    Args:
        model: The trained SteganoGAN model.
        dataloader: DataLoader containing the images to be processed.
        save_path: Path to save the images.
        num_examples: Number of images to process and save.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    examples = []
    model.eval()
    with torch.no_grad():
        for image in dataloader:
            if len(examples) >= num_examples: break
            data  = model.random_data([image])[0].unsqueeze(0)
            encoded_image, _  = model(image, data)

            inp_np = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            inp_pil = Image.fromarray((np.clip(inp_np, 0, 1) * 255).astype(np.uint8))
            inp_pil.save(os.path.join(save_path, f"input_image{len(examples)+1}.png"))

            s_img = encoded_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            s_img_pil = Image.fromarray((np.clip(s_img, 0, 1) * 255).astype(np.uint8))
            s_img_pil.save(os.path.join(save_path, f"save_images{len(examples)+1}.png"))


            examples.append(s_img_pil)
    return examples

def visualize_roc(csv_path="trad_steganalysis"):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV file is empty.")
        return

    y_true = df['File name'].apply(lambda x: 1 if 'save_images' in str(x) else 0).tolist()
    y_pred = df['Chi Square'].astype(float).tolist()

    fpr, tpr, thresholds = mets.roc_curve(y_true, y_pred)
    roc_auc = mets.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stegexpose visualization")
    parser.add_argument("--csv_path", type=str, default="trad_steganalysis", help="Path to a custom dataset directory")
    parser.add_argument("--dataset_path", type=str, default="data/COCO_val_2017", help="Path to a custom dataset directory")
    parser.add_argument("--save_path", type=str, default="save_and_div", help="Path to a custom dataset directory")
    parser.add_argument("--model_type", type=str, choices=["basic", "residual", "dense"], default="dense",
                      help="Type of SteganoGAN model to evaluate")
    parser.add_argument("--data_depth", type=int, default=1, help="Data depth of the model (bits per pixel)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the .pth model file to load")
    parser.add_argument("--visualize", action="store_true", help="Visualize examples of steganography")
    parser.add_argument("--save", action="store_true", help="Save input and encoded images to folder")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to visualize")
    
    args = parser.parse_args()

   
    if args.save:
      csv_path = args.csv_path
      model, dataloader = get_model_and_dataloader(model_type=args.model_type,
                                                data_depth=args.data_depth,
                                                model_path=args.model_path,
                                                dataset_path=args.dataset_path)
      save_images(model=model, dataloader=dataloader, save_path=args.save_path,num_examples=args.num_examples)
      
    if args.visualize:
        visualize_roc(args.csv_path)

    print("\nEvaluation complete.")
