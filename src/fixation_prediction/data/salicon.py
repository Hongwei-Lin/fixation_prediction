"""SALICON dataset class for PyTorch."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

import numpy as np
import torch
from PIL import Image

from fixation_prediction.data.utils import (
    list_salicon_image_ids,
    load_salicon_fixations,
)


class SaliconDataset(Dataset):
    """PyTorch Dataset for SALICON saliency data.

    Returns one sample at a time. Each sample contains the image, the
    ground-truth saliency heatmap, the discrete fixation points, and the
    image ID. See __getitem__ for the exact format.

    Only supports the 'train' and 'val' splits. The original SALICON 'test'
    split has no public labels, so it is not supported here.
    """

    VALID_SPLITS = ("train", "val")

    def __init__(
        self,
        root: Path | str,
        split: str,
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Path to the SALICON dataset root (containing images/,
                fixations/, maps/ subdirectories).
            split: One of 'train' or 'val'.
            image_size: Target (height, width) after resizing.

        Raises:
            ValueError: If split is not 'train' or 'val'.
            FileNotFoundError: If the dataset directories cannot be found.
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self.VALID_SPLITS}, got '{split}'"
            )

        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.images_dir = self.root / "images" / split
        self.maps_dir = self.root / "maps" / split
        self.fixations_dir = self.root / "fixations" / split

        for d in (self.images_dir, self.maps_dir, self.fixations_dir):
            if not d.exists():
                raise FileNotFoundError(f"Directory not found: {d}")

        self.image_ids = list_salicon_image_ids(self.root / "images", split)

    def __len__(self) -> int:
        return len(self.image_ids)
    

    def __getitem__(self, idx: int) -> dict:
        """Load and return one sample.

        Returns:
            Dictionary with keys:
                "image": float tensor, shape (3, H, W), pixel values in [0, 1].
                "saliency": float tensor, shape (1, H, W), sums to 1.
                "fixations": int tensor, shape (N, 2), columns are (x, y) in
                    the resized image's coordinate space.
                "image_id": string identifier of the sample.

            H and W are the configured image_size.
        """
        img_id = self.image_ids[idx]
        target_h, target_w = self.image_size

        # Load image
        image_path = self.images_dir / f"{img_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size  # PIL uses (width, height)
        image = image.resize((target_w, target_h), Image.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        # PIL/NumPy is (H, W, 3); PyTorch wants (3, H, W)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        # Load saliency map
        map_path = self.maps_dir / f"{img_id}.png"
        saliency = Image.open(map_path).convert("L")  # grayscale
        saliency = saliency.resize((target_w, target_h), Image.BILINEAR)
        saliency_array = np.asarray(saliency, dtype=np.float32)
        total = saliency_array.sum()
        if total > 0:
            saliency_array /= total
        saliency_tensor = torch.from_numpy(saliency_array).unsqueeze(0)

        # Load and rescale fixations
        mat_path = self.fixations_dir / f"{img_id}.mat"
        fixations, _ = load_salicon_fixations(mat_path)
        if fixations.shape[0] > 0:
            scale_x = target_w / original_w
            scale_y = target_h / original_h
            fixations_scaled = fixations.astype(np.float32).copy()
            fixations_scaled[:, 0] *= scale_x
            fixations_scaled[:, 1] *= scale_y
            fixations_scaled = fixations_scaled.astype(np.int32)
            # Clip to valid range after rounding
            fixations_scaled[:, 0] = np.clip(fixations_scaled[:, 0], 0, target_w - 1)
            fixations_scaled[:, 1] = np.clip(fixations_scaled[:, 1], 0, target_h - 1)
        else:
            fixations_scaled = np.zeros((0, 2), dtype=np.int32)
        fixations_tensor = torch.from_numpy(fixations_scaled)

        return {
            "image": image_tensor,
            "saliency": saliency_tensor,
            "fixations": fixations_tensor,
            "image_id": img_id,
        }
    

def salicon_collate(
    batch: list[dict],
) -> dict:
    """Collate function for SaliconDataset.

    Default PyTorch collation stacks tensors with torch.stack, which requires
    identical shapes. Fixation tensors have variable length (N, 2) per sample,
    so they cannot be stacked. This function stacks the fixed-shape tensors
    normally and keeps fixations as a list of per-sample tensors.

    Args:
        batch: List of sample dicts as returned by SaliconDataset.__getitem__.

    Returns:
        Dict with keys:
            "image":     stacked tensor, shape (B, 3, H, W).
            "saliency":  stacked tensor, shape (B, 1, H, W).
            "fixations": list of B tensors, each shape (N_i, 2).
            "image_id":  list of B strings.
    """
    return {
        "image": torch.stack([s["image"] for s in batch]),
        "saliency": torch.stack([s["saliency"] for s in batch]),
        "fixations": [s["fixations"] for s in batch],
        "image_id": [s["image_id"] for s in batch],
    }