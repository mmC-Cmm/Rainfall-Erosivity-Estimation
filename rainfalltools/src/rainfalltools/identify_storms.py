# identify_storms.py
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names to be case-insensitive across files."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def identify_storms(input_dir: str | Path,
                    output_dir: str | Path,
                    rain_col: str = "rain",
                    time_col: str | None = "time",
                    recursive: bool = True) -> pd.DataFrame:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(input_dir.glob(pattern))

    summaries = []

    for src in files:
        if not src.is_file():
            continue

        try:
            df = pd.read_csv(src)
        except Exception as e:
            print(f"⚠️  Failed to read {src}: {e}")
            continue

        # normalize column names
        df = _normalize_columns(df)
        rcol = rain_col.lower() if rain_col else "rain"
        tcol = time_col.lower() if time_col else None

        if rcol not in df.columns:
            print(f"⚠️  {src.name}: missing '{rcol}' — skipped.")
            continue

        # Convert to numeric and replace negative values with 0
        df[rcol] = pd.to_numeric(df[rcol], errors="coerce")
        df.loc[df[rcol] < 0, rcol] = 0

        if df.empty:
            print(f"— No valid rainfall data in {src.name}.")
            continue

        # Optional sort by time if present
        if tcol and tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
            df = df.sort_values(by=tcol, kind="mergesort")

        rain = df[rcol]

        # Identify increases
        inc_mask = rain.diff().gt(0).fillna(False)
        prev_mask = inc_mask.shift(-1, fill_value=False)
        storm_mask = inc_mask | prev_mask

        storm_df = df.loc[storm_mask].copy()

        if storm_df.empty:
            continue

        # Mirror input relative path
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            storm_df.to_csv(dst, index=False)
        except Exception as e:
            print(f"⚠️  Failed to write {dst}: {e}")
            continue

        summaries.append({
            "Input_File": str(src),
            "Output_File": str(dst),
            "Rows_Input": int(len(df)),
            "Rows_Kept": int(len(storm_df)),
            "Pct_Kept": round(100.0 * len(storm_df) / max(len(df), 1), 2)
        })

    return pd.DataFrame(summaries)
