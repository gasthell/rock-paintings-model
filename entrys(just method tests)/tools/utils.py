import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy.optimize import linear_sum_assignment as LinearSumAssignment

# Ref: https://github.com/NVlabs/ODISE/blob/e97b06c424c575fec9fc5368dd4b3e050d91abc4/odise/modeling/meta_arch/odise.py#L923
class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest-exact')

        with torch.no_grad():
            mask = mask.detach()
            mask = (mask > 0).to(mask.dtype)
            denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x

def _fast_hist(label_true, label_pred, n_class):
    # Adapted from https://github.com/janghyuncho/PiCIE/blob/c3aa029283eed7c156bbd23c237c151b19d6a4ad/utils.py#L99
    pred_n_class = np.maximum(n_class,label_pred.max()+1)
    mask = (label_true >= 0) & (label_true < n_class) # Exclude unlabelled data.
    hist = np.bincount(pred_n_class * label_true[mask] + label_pred[mask],\
                           minlength=n_class * pred_n_class).reshape(n_class, pred_n_class)
    return hist

def hungarian_matching(pred, label, n_class):
    # X,Y: b x 512 x 512
    batch_size = pred.shape[0]
    tp = np.zeros(n_class)
    fp = np.zeros(n_class)
    fn = np.zeros(n_class)
    all = 0
    for i in range(batch_size):
        hist = _fast_hist(label[i].flatten(), pred[i].flatten(), n_class)
        row_ind, col_ind = LinearSumAssignment(hist, maximize=True)
        all += hist.sum()
        fn += (np.sum(hist, 1) - hist[row_ind, col_ind])
        tp += hist[row_ind, col_ind]
        hist = hist[:, col_ind] # re-order hist to align labels to calculate FP
        fp += (np.sum(hist, 0) - np.diag(hist))
    return tp, fp, fn, all, hist, col_ind
