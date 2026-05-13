"""Utility functions for loading SALICON data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def list_salicon_image_ids(images_dir: Path | str, split: str) -> list[str]:
    """List all SALICON image IDs in a given split.

    Args:
        images_dir: Path to the images/ folder.
        split: One of 'train', 'val', 'test'.

    Returns:
        Sorted list of image IDs (filenames without extension), e.g.
        ['COCO_train2014_000000000009', 'COCO_train2014_000000000025', ...].

    Raises:
        FileNotFoundError: If the split subdirectory does not exist.
    """
    split_dir = Path(images_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    image_files = sorted(split_dir.glob("*.jpg"))
    return [f.stem for f in image_files]

def load_salicon_fixations(mat_path: Path | str) -> tuple[np.ndarray, tuple[int, int]]:
    """Load fixation points from a SALICON .mat file.

    A SALICON .mat file contains fixation data from ~50 mouse-tracking workers
    for a single image. Each worker has their own array of (x, y) fixation
    points. This function aggregates fixations across all workers into one
    array.

    Coordinate convention: column 0 is x (horizontal pixel), column 1 is y
    (vertical pixel). NumPy array indexing is [row=y, col=x] — apply the swap
    at the point where coordinates are used as array indices, not here.

    Args:
        mat_path: Path to the .mat file.

    Returns:
        Tuple (fixations, image_size):
            fixations: shape (N, 2), int32, columns are (x, y) in pixel coords.
                       Empty array (shape (0, 2)) if the image has no fixations.
            image_size: (height, width) of the original image.

    Raises:
        FileNotFoundError: If mat_path does not exist.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Fixation file not found: {mat_path}")

    mat = loadmat(str(mat_path))

    resolution = mat["resolution"][0]
    image_size = (int(resolution[0]), int(resolution[1]))

    gaze = mat["gaze"]
    all_fixations: list[np.ndarray] = []
    for i in range(gaze.shape[0]):
        worker_fix = gaze[i, 0]["fixations"]
        if worker_fix.size > 0:
            all_fixations.append(worker_fix)

    if not all_fixations:
        return np.zeros((0, 2), dtype=np.int32), image_size

    fixations = np.vstack(all_fixations).astype(np.int32)
    return fixations, image_size


def fixations_to_density_map(
    fixations: np.ndarray,
    image_size: tuple[int, int],
    sigma: float = 19.0,
) -> np.ndarray:
    """Convert discrete fixation points to a continuous Gaussian density map.

    Places a 2D Gaussian at each fixation point and sums them. The result is
    normalized to sum to 1 so it can be interpreted as a probability
    distribution over pixels.

    For SALICON, pre-computed maps exist in the maps/ folder, so this function
    is mainly used for datasets without pre-computed heatmaps (CXR gaze data,
    mammography).

    Args:
        fixations: shape (N, 2), columns are (x, y) in pixel coordinates.
                   Empty (N=0) is allowed and returns a zero map.
        image_size: (height, width) of the output map.
        sigma: Standard deviation of the Gaussian, in pixels. SALICON's
               default of ~19 pixels corresponds to roughly 1 degree of
               visual angle at typical viewing distance.

    Returns:
        Density map of shape (height, width), float32, normalized so the sum
        equals 1.0. Returns an all-zero map if fixations is empty.
    """
    from scipy.ndimage import gaussian_filter

    H, W = image_size
    density = np.zeros((H, W), dtype=np.float32)

    if fixations.shape[0] == 0:
        return density

    # SALICON fixations are (x, y); array indexing is [row=y, col=x].
    for x, y in fixations:
        if 0 <= y < H and 0 <= x < W:
            density[y, x] += 1.0

    density = gaussian_filter(density, sigma=sigma)

    total = density.sum()
    if total > 0:
        density /= total

    return density

