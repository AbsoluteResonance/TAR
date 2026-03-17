import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

def compute_boundary_iou_3d(gt_mask, pred_mask, boundary_width=None):
    """
    Compute 3D Boundary IoU between a ground truth mask and a predicted mask.

    Args:
        gt_mask (numpy.ndarray): 3D binary ground truth mask (0: background, 1: foreground).
        pred_mask (numpy.ndarray): 3D binary predicted mask (0: background, 1: foreground).
        boundary_width (int, optional): Width of the boundary region in voxels. 
                                      If None, it is set to 2% of the diagonal length of the volume.

    Returns:
        float: 3D Boundary IoU score.
    """

    boundary_iou = []

    for b in range(gt_mask.shape[0]):
        gt_mask = cp.array(gt_mask[b,0,...], dtype=bool)
        pred_mask = cp.array(pred_mask[b,0,...], dtype=bool)

        h, w, d = gt_mask.shape
        if boundary_width is None:
            # Default boundary width: 2% of the diagonal length of the 3D volume
            boundary_width = int(0.02 * cp.sqrt(h ** 2 + w ** 2 + d ** 2))

        # Compute boundary regions for GT and prediction
        gt_boundary = _find_boundary_voxels_3d(gt_mask, boundary_width)
        pred_boundary = _find_boundary_voxels_3d(pred_mask, boundary_width)

        # Compute intersection and union
        intersection = cp.logical_and(gt_boundary, pred_boundary).sum()
        union = cp.logical_or(gt_boundary, pred_boundary).sum()

        # Avoid division by zero
        boundary_iou.append(intersection / union if union > 0 else 0.0)

    return cp.mean(cp.asarray(boundary_iou))

def _find_boundary_voxels_3d(mask, boundary_width):
    """
    Find boundary voxels by expanding the mask boundary by `boundary_width` voxels.

    Args:
        mask (numpy.ndarray): 3D binary mask (0: background, 1: foreground).
        boundary_width (int): Width of the boundary region.

    Returns:
        numpy.ndarray: Binary mask of the boundary region.
    """
    # Compute the distance transform from the mask boundary
    mask_boundary = _get_mask_boundary_3d(mask)
    dist_transform = distance_transform_edt(~mask_boundary)

    # The boundary region is all voxels within `boundary_width` voxels of the boundary
    boundary_region = (dist_transform <= boundary_width) & mask

    return boundary_region

def _get_mask_boundary_3d(mask):
    """
    Extract the boundary voxels of a 3D binary mask.

    Args:
        mask (numpy.ndarray): 3D binary mask (0: background, 1: foreground).

    Returns:
        numpy.ndarray: Binary mask of the boundary voxels.
    """
    # Use binary erosion to find the inner boundary (26-connected neighborhood in 3D)
    struct = generate_binary_structure(3, 3)  # 3D full connectivity
    eroded = binary_erosion(mask, structure=struct)
    boundary = mask & ~eroded

    return boundary

# Example usage
if __name__ == "__main__":
    # Example 3D binary masks (replace with your actual masks)
    gt_mask = cp.zeros((50, 50, 50), dtype=bool)
    gt_mask[10:40, 10:40, 10:40] = True  # A cube ground truth mask

    pred_mask = cp.zeros((50, 50, 50), dtype=bool)
    pred_mask[11:41, 11:41, 11:41] = True  # A slightly larger predicted mask

    # Compute 3D Boundary IoU
    boundary_iou = compute_boundary_iou_3d(gt_mask, pred_mask)
    print(f"3D Boundary IoU: {boundary_iou:.4f}")