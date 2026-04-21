from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True, slots=True)
class PadInfo:
    pad_id: int
    x: float
    y: float
    scale: float
    direction: int
    cobo: int
    asad: int
    aget: int
    channel: int
    cy: float
    cx: float


class PadLookup:
    def __init__(self, pads: list[PadInfo]) -> None:
        self._pads = list(pads)
        self._by_hardware = {
            (pad.cobo, pad.asad, pad.aget, pad.channel): pad for pad in self._pads
        }
        self._by_pad_id = {pad.pad_id: pad for pad in self._pads}

    def get_by_hardware(
        self,
        *,
        cobo: int,
        asad: int,
        aget: int,
        channel: int,
    ) -> PadInfo | None:
        return self._by_hardware.get((int(cobo), int(asad), int(aget), int(channel)))

    def get_by_pad_id(self, pad_id: int) -> PadInfo | None:
        return self._by_pad_id.get(int(pad_id))

    def as_list(self) -> list[PadInfo]:
        return list(self._pads)


def default_pads_csv_path() -> Path:
    return Path(__file__).with_name("pads.csv")


def load_pad_lookup(path: Path | None = None) -> PadLookup:
    csv_path = default_pads_csv_path() if path is None else Path(path)
    pads: list[PadInfo] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pads.append(
                PadInfo(
                    pad_id=int(row["pad"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    scale=float(row["scale"]),
                    direction=int(row["direction"]),
                    cobo=int(row["cobo"]),
                    asad=int(row["asad"]),
                    aget=int(row["aget"]),
                    channel=int(row["channel"]),
                    cy=float(row["cy"]),
                    cx=float(row["cx"]),
                )
            )
    return PadLookup(pads)
