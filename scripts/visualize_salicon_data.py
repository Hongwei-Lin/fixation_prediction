"""Visualize SALICON dataset samples for sanity-checking the data pipeline.

Reads the data config via Hydra, builds a SaliconDataset, and saves
visualization figures to disk. Each figure has three panels: the image,
the image with saliency heatmap overlay, and the image with fixation points.

Usage:
    python scripts/visualize_data.py
    python scripts/visualize_data.py visualize.num_samples=10
    python scripts/visualize_data.py data.image_size='[384,384]'
"""
from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from fixation_prediction.data.salicon import SaliconDataset


def visualize_sample(sample: dict, output_path: Path) -> None:
    """Save a three-panel visualization of one sample."""
    image = sample["image"].permute(1, 2, 0).numpy()
    saliency = sample["saliency"].squeeze(0).numpy()
    fixations = sample["fixations"].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(saliency, cmap="hot", alpha=0.5)
    axes[1].set_title("Image + saliency overlay")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].scatter(fixations[:, 0], fixations[:, 1], s=2, c="lime", alpha=0.6)
    axes[2].set_title(f"Image + {len(fixations)} fixation points")
    axes[2].axis("off")

    fig.suptitle(f"{sample['image_id']}", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Visualize a few samples from the configured dataset."""
    # Build dataset from config
    dataset = SaliconDataset(
        root=cfg.data.root,
        split=cfg.data.train_split,
        image_size=tuple(cfg.data.image_size),
    )
    print(f"Dataset: {len(dataset)} samples in '{cfg.data.train_split}' split")

    # Choose sample indices to visualize
    num_samples = cfg.get("visualize", {}).get("num_samples", 4)
    rng = np.random.default_rng(seed=cfg.training.seed)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)
    print(f"Visualizing samples at indices: {indices.tolist()}")

    # Hydra changes the working directory to outputs/<run>/, so we save there
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    for idx in indices:
        sample = dataset[int(idx)]
        out_path = output_dir / f"sample_{int(idx):05d}_{sample['image_id']}.png"
        visualize_sample(sample, out_path)
        print(f"  Saved {out_path}")

    print(f"\nDone. Outputs in: {Path.cwd() / output_dir}")


if __name__ == "__main__":
    main()