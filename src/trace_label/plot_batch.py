from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .batch import CDF_THRESHOLDS


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    output_dir = Path(args.output_path).expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(f"input file not found: {input_path}")

    samples = load_batch_samples(input_path)
    plot_histograms(samples=samples, output_dir=output_dir)
    print(f"saved {len(CDF_THRESHOLDS)} histogram PNG files to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one PNG histogram for each CDF column from batch.py output")
    parser.add_argument("-i", "--input-file", required=True, help="Path to the .npy file produced by batch.py")
    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Directory where the histogram PNG files will be written",
    )
    return parser.parse_args()


def load_batch_samples(input_path: Path) -> np.ndarray:
    samples = np.load(input_path)
    if samples.ndim != 2:
        raise SystemExit(f"expected a 2D numpy array, got shape {samples.shape}")
    if samples.shape[1] != len(CDF_THRESHOLDS):
        raise SystemExit(
            f"expected {len(CDF_THRESHOLDS)} columns for {CDF_THRESHOLDS.tolist()}, got shape {samples.shape}",
        )
    return np.asarray(samples, dtype=np.float32)


def plot_histograms(samples: np.ndarray, output_dir: Path) -> None:
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, 1.0, 51)

    for column_index, threshold in enumerate(CDF_THRESHOLDS):
        values = samples[:, column_index]
        fig, ax = plt.subplots(figsize=(8, 5))

        if values.size > 0:
            mean = float(np.mean(values))
            median = float(np.median(values))
            ax.hist(values, bins=bins, color="#1d4e89", edgecolor="white", linewidth=0.5)
            ax.axvline(mean, color="#c1121f", linestyle="--", linewidth=1.5, label=f"mean={mean:.3f}")
            ax.axvline(median, color="#2a9d8f", linestyle="-.", linewidth=1.5, label=f"median={median:.3f}")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "no traces", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"Distribution of F({threshold})")
        ax.set_xlabel(f"F({threshold})")
        ax.set_ylabel("Count")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"F{threshold}.png", dpi=200)
        plt.close(fig)


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


if __name__ == "__main__":
    main()
