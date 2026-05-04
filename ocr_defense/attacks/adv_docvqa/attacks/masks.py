import torch
from typing import Tuple


def mask_include_all(image: torch.Tensor):
    """
    Creates a mask for the image, including all pixels.
    """
    mask = torch.ones_like(image, dtype=torch.bool)
    return mask


def mask_exclude_white(image: torch.Tensor, threshold: int = 245) -> torch.Tensor:
    """
    Exclude near-white areas from perturbations.

    Expected input shape is HxWxC (uint8-like range 0..255).
    Pixels brighter than threshold across all channels are masked out.
    """
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        # Fallback: if shape is unexpected, allow perturbing everything.
        return torch.ones_like(image, dtype=torch.bool)

    # Per-pixel keep-mask: perturb only non-white pixels.
    keep = (image[..., :3] < threshold).any(dim=-1, keepdim=True)
    return keep.expand_as(image).to(dtype=torch.bool)


def mask_bottom_right_corner(image: torch.Tensor, ratio=0.15) -> torch.Tensor:
    mask = torch.zeros_like(image, dtype=torch.bool)

    # the images in this version have C*W*H shape
    H, W, _ = image.shape
    size = int(min(W,H)*ratio)
    x_start, y_start = W - size, H - size
    mask[y_start:, x_start:, :] = 1  # Exclude the bottom-right corner

    return mask