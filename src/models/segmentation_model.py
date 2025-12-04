import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class BraTSSwinUNETR(nn.Module):
    """
    The main segmentation model.
    Architecture: Swin Transformer Encoder + U-Net Decoder (Swin-UNETR).
    
    Ref: Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors", 2022.
    """
    def __init__(self, img_size=(96, 96, 96), in_channels=4, out_channels=3, feature_size=48):
        super().__init__()
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,  # 48 is the 'Tiny' config, good for 8GB-12GB VRAM
            use_checkpoint=True,        # Saves memory by recomputing gradients
            spatial_dims=3
        )

    def forward(self, x):
        return self.model(x)

    def load_mae_weights(self, mae_path):
        """
        Transfers weights from the Self-Supervised MAE Encoder to this Segmentation model.
        This is the key step for high accuracy on small data.
        """
        print(f"Loading MAE Pretrained Weights from {mae_path}...")
        try:
            # Load the pretrained state dict
            # FIX: Added weights_only=False to prevent pickling errors on newer PyTorch versions
            # This is safe because we trust our own locally saved weights.
            pretrained_state = torch.load(mae_path, weights_only=False)
            
            # The MAE model likely has prefixes like 'swin_viT.' or 'encoder.'
            # We map them to the SwinUNETR 'swinViT' submodule.
            model_state = self.model.state_dict()
            
            transferred_layers = 0
            for k, v in pretrained_state.items():
                # Clean up prefixes if necessary (logic depends on how we save MAE)
                # Assuming direct mapping for Swin backbone
                if k in model_state and v.shape == model_state[k].shape:
                    model_state[k] = v
                    transferred_layers += 1
            
            self.model.load_state_dict(model_state, strict=False)
            print(f"Transferred {transferred_layers} layers from MAE Encoder.")
            
        except Exception as e:
            print(f"Failed to load MAE weights: {e}")