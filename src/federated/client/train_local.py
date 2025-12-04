import torch
import torch.nn as nn
import os
import sys
import json
from monai.losses import DiceCELoss

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------

from src.utils.config_loader import load_config
from models.segmentation_model import get_model
from src.data.loaders.monai_loader import get_dataloader
from src.ipfs.upload_model import upload_file

def train_client(client_id, global_weights_path, round_num):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[Client {client_id}] Training (Round {round_num})...")
    
    # 1. Load Data
    split_path = os.path.join(config['paths']['splits_dir'], f"client{client_id}_split.json")
    with open(split_path, "r") as f:
        data_list = json.load(f)
    loader = get_dataloader(data_list, batch_size=config['model']['batch_size'], mode="train")
    
    # 2. Init Model
    model = get_model(pretrained_weights=global_weights_path, mode="segmentation").to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['federated']['local_epochs'])
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
    
    # 3. Train Loop
    model.train()
    epochs = config['federated']['local_epochs']
    final_epoch_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0
        step = 0
        for batch in loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        
        scheduler.step()
        final_epoch_loss = epoch_loss / step
        print(f"   Epoch {epoch+1}/{epochs} - Loss: {final_epoch_loss:.4f}")

    # 4. Save Latest Model (Overwrites previous file to save space)
    save_dir = os.path.join(project_root, "saved_models", f"client{client_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    # We save as 'latest' for the aggregation
    local_path = os.path.join(save_dir, "model_latest.pt")
    torch.save(model.state_dict(), local_path)
    
    # Upload 'latest' for the next round
    cid = upload_file(local_path)
    if not cid: cid = f"ERR_CID_CLIENT{client_id}"

    # Return the LOSS so the server can decide if this is the "Best Round"
    return cid, local_path, final_epoch_loss