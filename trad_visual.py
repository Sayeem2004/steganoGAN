import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
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
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png'))]
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
            image = model.convert_image([image])[0]
            data  = model.random_data([image])[0].unsqueeze(0)
            encoded_image, _ = model(image, data)
            normal_image  = model.unconvert_image(encoded_image)[0]

            inp_img = model.unconvert_image(image)[0]
            inp_np = inp_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
            inp_pil = Image.fromarray((np.clip(inp_np, 0, 1) * 255).astype(np.uint8))
            inp_pil.save(os.path.join(save_path, f"input_image{len(examples)+1}.png"))

            s_img = normal_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            s_img_pil = Image.fromarray((np.clip(s_img, 0, 1) * 255).astype(np.uint8))
            s_img_pil.save(os.path.join(save_path, f"save_images{len(examples)+1}.png"))


            examples.append(s_img_pil)
    return examples

# create roc curve of true positive and false positive
def visualize_csv(model,dataloader,csv_path="save_images",num_examples=3):
    """csv of format:

    File name,Above stego threshold?,Secret message size in bytes (ignore for clean files),Primary Sets,Chi Square,Sample Pairs,RS analysis,Fusion (mean)
    save_images2.png,true,71824,NaN,1.9725536709611061E-7,0.4789900629378289,1.0,0.49299675339773197
    save_images3.png,true,37257,NaN,2.157525520910131E-9,0.46220604860185016,1.0,0.4874020169197919
    save_images1.png,true,64315,NaN,4.914422820285362E-10,null,1.0,0.5000000002457211
    """
    "test files are true false, save_image files in csv are true positive"

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV file is empty.")
        return
    # files named with save_images*.png are the true positive files, all others are true negative
    y_true = df['File name'].apply(lambda x: 1 if 'save_images' in str(x) else 0).tolist()
    y_pred = df['Above stego threshold?'].apply(lambda x: 1 if str(x).lower() == 'true' else 0).tolist()
    # create roc curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
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
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stegexpose visualization")
    parser.add_argument("--csv_path", type=str, default="steganalysis1.csv", help="Path to a custom dataset directory")
    parser.add_argument("--dataset_path", type=str, default="Div2K_test_LR_unknown/X4/", help="Path to a custom dataset directory")
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
      save_images(model=model, dataloader=dataloader, save_path=args.save_path)
      
    if args.visualize:
        visualize_csv(model, dataloader, csv_path)

    print("\nEvaluation complete.")
