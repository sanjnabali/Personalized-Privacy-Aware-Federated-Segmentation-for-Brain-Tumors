import torch
import torch.nn as nn
from monai.losses import DiceCELoss

class BraTSLoss(nn.Module):
    """
    Compound loss for Segmentation: Dice + Cross Entropy.
    Handles class imbalance via Soft Dice.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,  # Convert target labels to One-Hot
            softmax=True,      # Apply Softmax to predictions
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )

    def forward(self, preds, targets):
        return self.loss_fn(preds, targets)

class MAELoss(nn.Module):
    """
    Loss for Self-Supervised Learning.
    MSE Loss computed ONLY on masked regions.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, recon, target, mask):
        # mask: 1 = visible, 0 = masked
        # We want loss on MASKED regions (where mask == 0)
        loss_map = self.mse(recon, target)
        
        # Invert mask (1 where we masked, 0 where we kept)
        mask_region = 1 - mask
        
        # Compute mean only on masked pixels
        loss = (loss_map * mask_region).sum() / (mask_region.sum() + 1e-6)
        return loss