from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from trace_label.plot_batch import load_batch_samples, main


class FakeAxes:
    def __init__(self) -> None:
        self.transAxes = object()

    def hist(self, values, bins, color, edgecolor, linewidth) -> None:
        return None

    def axvline(self, value, color, linestyle, linewidth, label) -> None:
        return None

    def legend(self) -> None:
        return None

    def text(self, x, y, label, ha, va, transform) -> None:
        return None

    def set_title(self, title) -> None:
        return None

    def set_xlabel(self, label) -> None:
        return None

    def set_ylabel(self, label) -> None:
        return None

    def set_xlim(self, left, right) -> None:
        return None

    def grid(self, enabled, alpha) -> None:
        return None


class FakeFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, path: Path, dpi: int) -> None:
        Path(path).write_bytes(b"fake-png")


def install_fake_matplotlib(monkeypatch) -> None:
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.use = lambda backend: None

    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.subplots = lambda figsize: (FakeFigure(), FakeAxes())
    fake_pyplot.close = lambda fig: None

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)


def test_load_batch_samples_rejects_wrong_shape(tmp_path) -> None:
    input_path = tmp_path / "bad.npy"
    np.save(input_path, np.ones((4, 9), dtype=np.float32))

    try:
        load_batch_samples(input_path)
    except SystemExit as exc:
        assert "expected 10 columns" in str(exc)
    else:
        raise AssertionError("expected load_batch_samples to reject the input shape")


def test_plot_batch_main_writes_one_png_per_histogram(tmp_path, monkeypatch) -> None:
    install_fake_matplotlib(monkeypatch)

    input_path = tmp_path / "cdf.npy"
    output_dir = tmp_path / "plots"
    np.save(input_path, np.full((6, 10), 0.5, dtype=np.float32))

    monkeypatch.setattr(sys, "argv", ["plot-batch", "-i", str(input_path), "-o", str(output_dir)])
    main()

    expected_files = [output_dir / f"F{threshold}.png" for threshold in (10, 20, 30, 40, 50, 60, 100, 150, 200, 250)]
    assert all(path.is_file() for path in expected_files)
