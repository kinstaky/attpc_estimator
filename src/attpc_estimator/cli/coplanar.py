from __future__ import annotations

import sys

from .histogram import main as histogram_main


def main() -> None:
    histogram_main(["coplanar", *sys.argv[1:]])
