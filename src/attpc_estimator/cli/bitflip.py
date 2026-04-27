from __future__ import annotations

import sys

from .histogram import main as histogram_main


def main() -> None:
    histogram_main(["bitflip", *sys.argv[1:]])
