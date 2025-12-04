import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

class BraTSMetrics:
    def __init__(self, device):
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        self.device = device

    def reset(self):
        self.dice_metric.reset()
        self.hausdorff_metric.reset()

    def update(self, preds, targets):
        """
        Args:
            preds: (B, C, H, W, D) Logits or Softmax
            targets: (B, 1, H, W, D) Integer labels
        """
        # Convert targets to One-Hot for Metric Calculation
        # Labels: 0, 1 (NCR), 2 (ED), 4 (ET) -> We need to map to 3 channels
        
        # Post-processing for Metric:
        # 1. Apply Sigmoid/Softmax
        # 2. Threshold > 0.5
        # 3. Compute
        
        # Note: In a full pipeline, we map 0,1,2,4 to 3 binary channels (WT, TC, ET)
        # For simplicity here, we assume preds/targets are already aligned in channel space
        self.dice_metric(y_pred=preds, y=targets)
        # Hausdorff is expensive, use sparingly
        # self.hausdorff_metric(y_pred=preds, y=targets)

    def compute(self):
        return self.dice_metric.aggregate().item()