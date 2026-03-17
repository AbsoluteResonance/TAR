import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

def compute_boundary_iou(gt_mask, pred_mask, boundary_width=None):
    """
    Compute Boundary IoU between a ground truth mask and a predicted mask.

    Args:
        gt_mask (numpy.ndarray): Binary ground truth mask (0: background, 1: foreground).
        pred_mask (numpy.ndarray): Binary predicted mask (0: background, 1: foreground).
        boundary_width (int, optional): Width of the boundary region in pixels. 
                                      If None, it is set to 2% of the diagonal length of the mask.

    Returns:
        float: Boundary IoU score.
    """
    assert gt_mask.shape == pred_mask.shape, "Masks must have the same shape."
    assert gt_mask.dtype == bool and pred_mask.dtype == bool, "Masks must be binary (boolean)."

    h, w = gt_mask.shape
    if boundary_width is None:
        # Default boundary width: 2% of the diagonal length
        boundary_width = int(0.02 * np.sqrt(h ** 2 + w ** 2))

    # Compute boundary regions for GT and prediction
    gt_boundary = _find_boundary_pixels(gt_mask, boundary_width)
    pred_boundary = _find_boundary_pixels(pred_mask, boundary_width)

    # Compute intersection and union
    intersection = np.logical_and(gt_boundary, pred_boundary).sum()
    union = np.logical_or(gt_boundary, pred_boundary).sum()

    # Avoid division by zero
    boundary_iou = intersection / union if union > 0 else 0.0

    return boundary_iou

def _find_boundary_pixels(mask, boundary_width):
    """
    Find boundary pixels by expanding the mask boundary by `boundary_width` pixels.

    Args:
        mask (numpy.ndarray): Binary mask (0: background, 1: foreground).
        boundary_width (int): Width of the boundary region.

    Returns:
        numpy.ndarray: Binary mask of the boundary region.
    """
    # Compute the distance transform from the mask boundary
    mask_boundary = _get_mask_boundary(mask)
    dist_transform = distance_transform_edt(~mask_boundary)

    # The boundary region is all pixels within `boundary_width` pixels of the boundary
    boundary_region = (dist_transform <= boundary_width) & mask

    return boundary_region

def _get_mask_boundary(mask):
    """
    Extract the boundary pixels of a binary mask.

    Args:
        mask (numpy.ndarray): Binary mask (0: background, 1: foreground).

    Returns:
        numpy.ndarray: Binary mask of the boundary pixels.
    """
    # Use binary erosion to find the inner boundary
    struct = generate_binary_structure(2, 2)  # 8-connected neighborhood
    eroded = binary_erosion(mask, structure=struct)
    boundary = mask & ~eroded

    return boundary

# Example usage
if __name__ == "__main__":
    # Example binary masks (replace with your actual masks)
    gt_mask = np.zeros((100, 100), dtype=bool)
    gt_mask[20:80, 20:80] = True  # A square ground truth mask

    pred_mask = np.zeros((100, 100), dtype=bool)
    pred_mask[20:80, 20:80] = True  # A slightly larger predicted mask

    # Compute Boundary IoU
    boundary_iou = compute_boundary_iou(gt_mask, pred_mask)
    print(f"Boundary IoU: {boundary_iou:.4f}")