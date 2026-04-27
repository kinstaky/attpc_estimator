from __future__ import annotations

import sys

from .histogram import main as histogram_main


def main() -> None:
    histogram_main(["cdf", *sys.argv[1:]])
