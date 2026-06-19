# src/cbp/io/store.py
from pathlib import Path
import pandas as pd

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
