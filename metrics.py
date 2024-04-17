import numpy as np

class MetricsCalculator:
    def __init__(self, target, output):
        self.target = target
        self.output = output

    def dice_coef(self):
        # Flatten the tensors
        output_flat = self.output.flatten()
        target_flat = self.target.flatten()

        # Calculate intersection and union
        intersection = 2 * ((output_flat * target_flat).sum())
        union = output_flat.sum() + target_flat.sum()

        # Calculate Dice coefficient, handle divide by zero
        dice = (intersection / union) if union > 0 else 0.0

        return dice

    def calculate_tpr(self):
        # Ensure that target and output have the same shape
        assert self.target.shape == self.output.shape, "Shape mismatch between target and output"

        # Count True Positives (TP) and False Negatives (FN)
        tp = np.sum((self.target == 1) & (self.output == 1))
        fn = np.sum((self.target == 1) & (self.output == 0))

        # Calculate TPR
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0

        return tpr

    def iou_coef(self):
        # Flatten the tensors
        output_flat = self.output.flatten()
        target_flat = self.target.flatten()

        # Calculate intersection and union
        intersection = (output_flat * target_flat).sum()
        union = output_flat.sum() + target_flat.sum() - intersection

        # Calculate IoU, handle divide by zero
        iou = (intersection / union) if union > 0 else 0.0

        return iou
