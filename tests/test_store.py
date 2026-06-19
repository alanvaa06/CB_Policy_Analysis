# tests/test_store.py
import pandas as pd
from cbp.io.store import write_parquet, read_parquet

def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    p = tmp_path / "x.parquet"
    write_parquet(df, p)
    out = read_parquet(p)
    pd.testing.assert_frame_equal(df, out)
