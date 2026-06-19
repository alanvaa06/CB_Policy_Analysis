# src/cbp/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 5, 22)
    n0: int = 20
    target_series: tuple[str, ...] = ("DGS2", "EFFR")
    event_window: tuple[int, int] = (-1, 1)
    data_dir: Path = Path("data")
    fred_api_key: str | None = None
