import numpy as np
import torch
import warnings
from torch import nn
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage.measurements import center_of_mass


class HausdorffDTLoss(nn.Module):
    """Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, include_background=True, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha
        self.include_background = include_background

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] == 1

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, N, x, y, z) or (b, N, x, y)
        target: (b, N, x, y, z) or (b, N, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        n_pred_ch = pred.shape[1]
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:,...]
                pred = pred[:, 1:,...]
        
        loss = 0
        for i in torch.arange(pred.shape[1]):
            pred1, target1 = torch.unsqueeze(pred[:,i,...], 1), torch.unsqueeze(target[:,i,...], 1)
            pred_dt = torch.from_numpy(self.distance_field(pred1.cpu().numpy())).float().to(pred.device)
            target_dt = torch.from_numpy(self.distance_field(target1.cpu().numpy())).float().to(pred.device)

            pred_error = (pred1 - target1) ** 2
            distance = pred_dt ** self.alpha + target_dt ** self.alpha

            dt_field = pred_error * distance
            loss += dt_field.mean()
        loss = loss/pred.shape[1]
        
        return loss


class CenterOfMassLoss(nn.Module):
    """Center of mass loss based on euclidean distance"""

    def __init__(self, include_background=True, **kwargs):
        super(CenterOfMassLoss, self).__init__()
        self.include_background = include_background

    @torch.no_grad()
    def centerOfMass(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros((img.shape[0], 2))
        for batch in range(len(img)):
            fg_mask = img[batch] == 1

            if fg_mask.any():
                field[batch] = np.array(center_of_mass(fg_mask))
            else:
                field[batch] = np.array([np.nan,np.nan])

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, N, x, y, z) or (b, N, x, y)
        target: (b, N, x, y, z) or (b, N, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        n_pred_ch = pred.shape[1]
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:,...]
                pred = pred[:, 1:,...]
        
        loss = 0
        for i in torch.arange(pred.shape[1]):
            pred1, target1 = pred[:,i,...], target[:,i,...]

            pred_dt = torch.from_numpy(self.centerOfMass(pred1.cpu().numpy())).float().to(pred1.device)
            target_dt = torch.from_numpy(self.centerOfMass(target1.cpu().numpy())).float().to(pred1.device)
            loss += torch.mean(torch.sqrt(torch.nansum((pred_dt - target_dt) ** 2, axis=1)))
        loss = loss/pred.shape[1]
        return loss

