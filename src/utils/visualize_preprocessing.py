import os
import glob
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import random

# --- CONFIGURATION ---
# Adjust these to match your exact folder structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "brats_training")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "processed")

def visualize_random_subject():
    print(f"🔍 Searching for processed files in: {PROCESSED_DATA_DIR}")
    
    # 1. Find a random processed subject
    pt_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.pt"))
    if not pt_files:
        print("❌ No .pt files found. Run preprocessing first.")
        return

    selected_file = random.choice(pt_files)
    subject_id = os.path.basename(selected_file).replace(".pt", "")
    print(f"✨ Visualizing Subject: {subject_id}")

    # 2. Load Processed Tensor (The 'After')
    # Note: weights_only=False is required for the dictionary structure
    try:
        data_dict = torch.load(selected_file, weights_only=False)
        img_tensor = data_dict["image"]  # Shape: (C, H, W, D)
        lbl_tensor = data_dict["label"] if "label" in data_dict else None
        
        print(f"   Processed Shape: {img_tensor.shape}")
        print(f"   Intensity Range: {img_tensor.min():.2f} to {img_tensor.max():.2f} (Z-Score)")
    except Exception as e:
        print(f"❌ Error loading .pt file: {e}")
        return

    # 3. Load Original Raw NIfTI (The 'Before')
    # We try to find the FLAIR image for comparison
    raw_subject_dir = os.path.join(RAW_DATA_DIR, subject_id)
    flair_search = glob.glob(os.path.join(raw_subject_dir, "*flair.nii*"))
    
    if flair_search:
        raw_img_path = flair_search[0]
        raw_nii = nib.load(raw_img_path)
        raw_data = raw_nii.get_fdata()
        print(f"   Original Shape: {raw_data.shape}")
    else:
        print(f"⚠️ Raw NIfTI file not found in {raw_subject_dir}")
        raw_data = None

    # 4. Plotting Side-by-Side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # A. Original Image
    if raw_data is not None:
        # Get middle slice
        mid_slice = raw_data.shape[2] // 2
        axes[0].imshow(np.rot90(raw_data[:, :, mid_slice]), cmap="gray")
        axes[0].set_title(f"Original Raw FLAIR\n{raw_data.shape}")
    else:
        axes[0].text(0.5, 0.5, "Raw File Not Found", ha='center')
        axes[0].set_title("Original Raw")

    # B. Preprocessed Tensor (Channel 0 = FLAIR usually)
    # Tensor is (C, H, W, D), we take middle slice of depth
    d_idx = img_tensor.shape[3] // 2
    
    # We use .cpu() and .numpy() to plot
    processed_slice = img_tensor[0, :, :, d_idx].cpu().numpy()
    
    axes[1].imshow(np.rot90(processed_slice), cmap="gray")
    axes[1].set_title(f"Preprocessed Input (Tensor)\n{img_tensor.shape}\n(Cropped & Normalized)")

    # C. Label (Segmentation Mask)
    if lbl_tensor is not None:
        lbl_slice = lbl_tensor[0, :, :, d_idx].cpu().numpy()
        # Use a colorful map for labels (0=Bg, 1=NCR, 2=ED, 4=ET)
        axes[2].imshow(np.rot90(lbl_slice), cmap="jet", interpolation="nearest")
        axes[2].set_title("Ground Truth Label")
    else:
        axes[2].set_title("No Label")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_random_subject()