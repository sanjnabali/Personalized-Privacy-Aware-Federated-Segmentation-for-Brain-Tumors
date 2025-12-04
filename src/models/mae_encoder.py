import torch
import torch.nn as nn
import inspect
import monai
from monai.networks.nets import SwinUNETR

class MAE_Swin(nn.Module):
    """
    Self-Supervised Masked Autoencoder (MAE) Wrapper.
    
    Includes Regularization Techniques:
    - Dropout
    - Stochastic Depth (DropPath)
    """
    def __init__(self, img_size=(96, 96, 96), in_channels=4, feature_size=48, mask_ratio=0.75, dropout=0.0, drop_path_rate=0.2):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        
        # DEBUG: Print version to ensure we aren't using an ancient build
        print(f"   [Debug] MONAI Version: {monai.__version__}")

        # COMPATIBILITY FIX: Check if 'SwinUNETR' expects 'drop_path_rate' (old) or 'dropout_path_rate' (new)
        # This prevents crashing on slightly different 1.x versions.
        sig = inspect.signature(SwinUNETR.__init__)
        if 'dropout_path_rate' in sig.parameters:
            drop_path_kw = {'dropout_path_rate': drop_path_rate}
        elif 'drop_path_rate' in sig.parameters:
            drop_path_kw = {'drop_path_rate': drop_path_rate}
        else:
            drop_path_kw = {}

        try:
            # We use SwinUNETR as the backbone.
            self.backbone = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=in_channels, 
                feature_size=feature_size,
                use_checkpoint=True,
                spatial_dims=3,
                # --- REGULARIZATION ---
                drop_rate=dropout,             # Standard Dropout (Head)
                attn_drop_rate=0.0,            # Attention Dropout
                **drop_path_kw                 # Automatically uses the correct argument name
            )
        except TypeError as e:
            print(f"\n❌ CRITICAL MONAI ERROR: {e}")
            print(f"   Your installed SwinUNETR expects these arguments: {list(sig.parameters.keys())}")
            print("   Please run: 'pip install --upgrade monai'\n")
            raise e

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W, D)
        Returns:
            recon: Reconstructed image
            mask: Binary mask used (1 = kept, 0 = masked)
        """
        B, C, H, W, D = x.shape
        
        # 1. Generate Random Mask (Block-wise)
        patch_size = 16 
        mask_h, mask_w, mask_d = H // patch_size, W // patch_size, D // patch_size
        
        # Create low-res mask
        noise = torch.rand(B, 1, mask_h, mask_w, mask_d, device=x.device)
        mask_low_res = (noise > self.mask_ratio).float() # 1 = Keep, 0 = Mask
        
        # Upsample mask to full resolution
        mask = torch.nn.functional.interpolate(mask_low_res, size=(H, W, D), mode='nearest')
        
        # 2. Apply Mask
        x_masked = x * mask
        
        # 3. Create the reconstruction target
        # The loss should only be calculated on the masked patches.
        # We zero out the visible patches in the target.
        x_target = x * (1 - mask)
        
        # 4. Forward Pass (Reconstruction)
        recon = self.backbone(x_masked)

        # 5. Only consider the reconstruction of the masked patches for the loss
        recon_masked_only = recon * (1 - mask)
        
        return recon_masked_only, x_target, mask

    def save_encoder(self, path):
        """Saves weights to be loaded by the segmentation model"""
        torch.save(self.backbone.state_dict(), path)