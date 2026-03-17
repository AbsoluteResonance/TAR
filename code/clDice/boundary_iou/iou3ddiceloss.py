import os
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_signed_distance_map_3d(mask):
    mask = cp.asarray(mask)
    boundary = cp.zeros_like(mask, dtype=cp.uint8)
    
    # Detect 3D boundaries
    boundary[:-1, :, :] |= (mask[:-1, :, :] != mask[1:, :, :])
    boundary[1:, :, :] |= (mask[:-1, :, :] != mask[1:, :, :])
    boundary[:, :-1, :] |= (mask[:, :-1, :] != mask[:, 1:, :])
    boundary[:, 1:, :] |= (mask[:, :-1, :] != mask[:, 1:, :])
    boundary[:, :, :-1] |= (mask[:, :, :-1] != mask[:, :, 1:])
    boundary[:, :, 1:] |= (mask[:, :, :-1] != mask[:, :, 1:])
    
    dist_to_boundary = distance_transform_edt(~boundary)
    signed_dist = cp.where(mask, -dist_to_boundary, dist_to_boundary).astype(cp.float32)
    
    return cp.asnumpy(signed_dist)

class BoundaryDiceLoss(nn.Module):
    def __init__(self, d=2):
        super(BoundaryDiceLoss, self).__init__()
        self.d = d

    def forward(self, output, target):
        pred_probs = F.softmax(output, dim=1)[:, 1, :, :, :].unsqueeze(1)
        batch_size = target.size(0)
        total_loss = 0.0

        for b in range(batch_size):
            gt_mask = target[b, 0].cpu().numpy().astype(np.uint8)
            gt_dist = compute_signed_distance_map_3d(gt_mask)
            gt_boundary_region = np.abs(gt_dist) <= self.d

            pred_mask = (pred_probs[b, 0] > 0.5).cpu().numpy().astype(np.uint8)
            pred_dist = compute_signed_distance_map_3d(pred_mask)
            pred_boundary_region = np.abs(pred_dist) <= self.d

            combined_region = torch.from_numpy(gt_boundary_region | pred_boundary_region).to(target.device)
            if combined_region.sum() == 0:
                continue

            pred_region = pred_probs[b, 0][combined_region].view(1, -1)
            target_region = target[b, 0][combined_region].view(1, -1)

            intersection = torch.sum(pred_region * target_region)
            cardinality = torch.sum(pred_region) + torch.sum(target_region)
            dice = (2. * intersection + 1e-5) / (cardinality + 1e-5)
            total_loss += (1 - dice)

        return total_loss / batch_size