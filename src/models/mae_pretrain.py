import os
import sys
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse

# --- PATH FIX (To run from root) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.mae_encoder import MAE_Swin
from src.models.loss_functions import MAELoss
from src.data.loaders.monai_loader import get_loader

# --- CONFIGURATION ---
CONFIG = {
    # FIX: Point to 'Data/processed' where run_preprocessing.py saved the files
    "data_dir": "Data/processed", 
    "save_dir": "saved_models/maes",
    "epochs": 20,              
    "batch_size": 1,           
    "accum_steps": 4,          
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "patch_size": (96, 96, 96), 
    "mask_ratio": 0.75,        
    "use_amp": True            
}

def run_pretraining():
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting MAE Pre-training on {device}")
    print(f"   Config: AMP={CONFIG['use_amp']}, Mask={CONFIG['mask_ratio']}, Accum={CONFIG['accum_steps']}")
    
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # 2. Load Data
    print(f"Loading Dataset from: {os.path.abspath(CONFIG['data_dir'])}")
    try:
        train_loader = get_loader(CONFIG["data_dir"], batch_size=CONFIG["batch_size"], mode="train")
    except FileNotFoundError as e:
        print(f"DATA ERROR: {e}")
        return

    # 3. Initialize Model, Optimizer, Scheduler, Scaler
    model = MAE_Swin(img_size=CONFIG["patch_size"], mask_ratio=CONFIG["mask_ratio"]).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = GradScaler(enabled=CONFIG["use_amp"])
    loss_fn = MAELoss()
    
    # 4. Training Loop
    model.train()
    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        optimizer.zero_grad() 
        
        for i, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            
            # A. Mixed Precision Forward Pass
            with autocast(enabled=CONFIG["use_amp"]):
                recon, target, mask = model(images)
                loss = loss_fn(recon, target, mask)
                loss = loss / CONFIG["accum_steps"]
            
            # B. Backward Pass with Scaler
            scaler.scale(loss).backward()
            
            # C. Weight Update
            if (i + 1) % CONFIG["accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * CONFIG["accum_steps"] 
            progress_bar.set_postfix({"MAE Loss": loss.item() * CONFIG["accum_steps"]})
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
        
        # 5. Save Checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(CONFIG["save_dir"], f"mae_epoch_{epoch+1}.pt")
            model.save_encoder(save_path)
            print(f"Saved Encoder Checkpoint: {save_path}")

    final_path = os.path.join(CONFIG["save_dir"], "mae_final.pt")
    model.save_encoder(final_path)
    print(f"MAE Pre-training Complete. Encoder ready for Phase 4 Transfer.")

if __name__ == "__main__":
    run_pretraining()