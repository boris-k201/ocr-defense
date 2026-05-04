from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


class DonutImageProcessor:
    """
    Compatibility wrapper for legacy attack code.
    Old code expects a callable processor that returns {"pixel_values": tensor}.
    """

    def __init__(self, processor: Any) -> None:
        self._processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        use_fast: bool = True,
        local_files_only: bool = False,
    ) -> "DonutImageProcessor":
        processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=use_fast,
            local_files_only=local_files_only,
        )
        return cls(processor)

    def __call__(self, image_like) -> dict[str, torch.Tensor]:
        if isinstance(image_like, torch.Tensor):
            t = image_like.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                # CHW -> HWC
                t = t.permute(1, 2, 0)
            arr = t.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr[..., :3] if arr.ndim == 3 else arr).convert("RGB")
        elif isinstance(image_like, Image.Image):
            pil = image_like.convert("RGB")
        else:
            arr = np.asarray(image_like)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr[..., :3] if arr.ndim == 3 else arr).convert("RGB")

        out = self._processor(images=pil, return_tensors="pt")
        return {"pixel_values": out["pixel_values"]}


__all__ = ["DonutImageProcessor"]
