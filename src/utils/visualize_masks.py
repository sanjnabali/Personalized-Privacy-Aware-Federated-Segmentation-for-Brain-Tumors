import torch
import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LEDGER_PATH = os.path.join(PROJECT_ROOT, "experiments", "blockchain_ledger.json")



from src.utils.config_loader import load_config
from models.segmentation_model import get_model
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord

def visualize_results():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Global Model
    model_path = config['paths']['global_model']
    if not os.path.exists(model_path):
        print("Global model not found! Run the orchestrator first.")
        return

    print(f"Loading Global Model from: {model_path}")
    # Load 'segmentation' mode (3 classes)
    model = get_model(mode="segmentation").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # 2. Pick a random patient from Client 1 split
    split_path = os.path.join(config['paths']['splits_dir'], "client1_split.json")
    with open(split_path, "r") as f:
        data_list = json.load(f)
    
    # Let's take the first patient for consistency
    sample_data = data_list[0] 
    print(f"Visualizing Patient: {sample_data['image'][0]}")

    # 3. Preprocess Single Image
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ])
    
    data_item = transforms(sample_data)
    image = data_item["image"].unsqueeze(0).to(device) # Add batch dim
    label = data_item["label"].unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        output = model(image)
        # Apply Softmax to get probabilities, then Argmax to get class index (0, 1, 2, 3)
        # Shape: [1, 3, 96, 96, 96] -> [1, 96, 96, 96]
        prediction = torch.argmax(output, dim=1).detach().cpu().numpy()

    # 5. Extract Middle Slice for Visualization
    # Image shape: [1, 4, D, H, W] -> We take channel 0 (Flair) and middle depth
    depth_slice = image.shape[2] // 2 
    
    img_slice = image[0, 0, :, :, depth_slice].cpu().numpy() # Flair
    lbl_slice = label[0, 0, :, :, depth_slice].cpu().numpy() # Ground Truth
    pred_slice = prediction[0, :, :, depth_slice]            # AI Prediction

    # 6. Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("MRI Input (Flair)")
    plt.imshow(img_slice, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth (Doctor)")
    plt.imshow(lbl_slice, cmap="jet", alpha=0.7) # Jet colormap makes tumors pop
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Swin-UNETR Prediction")
    plt.imshow(pred_slice, cmap="jet", alpha=0.7)
    plt.axis('off')

    save_path = os.path.join(project_root, "results", "paper_figures", "segmentation_result.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Snapshot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_results()