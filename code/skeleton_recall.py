import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import skeletonize, dilation

from cldice.clDice.cldice_loss.pytorch.soft_skeleton import SoftSkeletonize as clDice_SoftSkeletonize
import time

class SoftSkeletonRecallLoss(nn.Module):
    def __init__(self, batch_dice: bool = False, smooth: float = 1.):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(SoftSkeletonRecallLoss, self).__init__()

        self.batch_dice = batch_dice
        self.smooth = smooth
        self.clDice_soft_skeletonize = clDice_SoftSkeletonize(num_iter=10)

    def forward(self, x, y, loss_mask=None):
        # make everything shape (b, c)
        axes = list(range(2, len(x.shape)))
    
        loss_mask = skeleton_transform(y) if loss_mask is None else None
        #loss_mask = self.clDice_soft_skeletonize(y) if loss_mask is None else None

        with torch.no_grad():
            y_onehot = y.float()
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))

        rec = rec.mean()
        return 1.-rec

def skeleton_transform(mask):
    seg_all = mask[:, 0, ...].cpu().numpy()
    # Add tubed skeleton GT
    bin_seg = (seg_all > 0)
    seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
    
    # Skeletonize
    for b in range(bin_seg.shape[0]):
        skel = skeletonize(bin_seg[b])
        skel = (skel > 0).astype(np.int16)
        skel = dilation(dilation(skel))
        skel *= seg_all[b].astype(np.int16)
        seg_all_skel[b] = skel

    return torch.from_numpy(seg_all_skel).unsqueeze(1).to(mask.device)

class cldiceSkeletonLoss(nn.Module):
    def __init__(self, smooth = 1., alpha=1., beta=1.):
        super(cldiceSkeletonLoss, self).__init__()
        self.smooth = smooth
        self.soft_skeletonize = clDice_SoftSkeletonize(num_iter=10)
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        y_true = y_true.to(torch.float32)
        skel_pred = self.soft_skeletonize(y_pred)

        #skel_true = self.soft_skeletonize(y_true)
        skel_true = skeleton_transform(y_true)

        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        # make everything shape (b, c)
        axes = list(range(2, len(y_pred.shape)))
        with torch.no_grad():
            y_onehot = y_true.float()
            sum_gt = y_onehot.sum(axes) if skel_true is None else (y_onehot * skel_true).sum(axes)
        inter_rec = (y_pred * y_onehot).sum(axes) if skel_true is None else (y_pred * y_onehot * skel_true).sum(axes)
        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))
        rec = 1. - rec.mean()

        return self.alpha * cl_dice +  self.beta * rec
