# identify_storms.py
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Iterable, Optional, Union, List
import pandas as pd
import glob

__all__ = ["identify_storms", "collect_csvs"]

def collect_csvs(root: Union[str, Path], recursive: bool = True) -> List[str]:
    """Return a list of CSV files under root."""
    root = str(root)
    pattern = f"{root}/**/*.csv" if recursive else f"{root}/*.csv"
    return sorted(glob.glob(pattern, recursive=recursive))

def identify_storms(
    files: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    *,
    rain_col: str = "rain",
    time_col: Optional[str] = "time",
) -> pd.DataFrame:
    """
    Detect storm rows (around positive increments in cumulative rainfall)
    for a list of CSV files. Writes filtered CSVs and returns a summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for src in files:
        src = Path(src)
        if not src.is_file() or src.suffix.lower() != ".csv":
            summaries.append({"input_file": str(src), "status": "skip_non_csv"})
            continue

        try:
            df = pd.read_csv(src)
        except Exception as e:
            summaries.append({"input_file": str(src), "status": f"read_error:{e}"})
            continue

        if rain_col not in df.columns:
            summaries.append({
                "input_file": str(src), "status": f"missing_column:{rain_col}",
                "rows_input": len(df), "rows_kept": 0, "pct_kept": 0.0
            })
            continue

        # Clean rain; negatives → 0
        df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0)
        df.loc[df[rain_col] < 0, rain_col] = 0

        # Optional time sort (stable sort)
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).sort_values(time_col, kind="mergesort").reset_index(drop=True)

        # Positive increments
        inc = df[rain_col].diff().gt(0).fillna(False)

        # Keep the increase row AND the row BEFORE the increase
        keep_mask = inc | inc.shift(1, fill_value=False)

        out_df = df.loc[keep_mask].copy()

        # Output uses the same filename under output_dir
        dst = output_dir / src.name
        try:
            out_df.to_csv(dst, index=False)
            status = "ok" if not out_df.empty else "empty_after_filter"
        except Exception as e:
            status = f"write_error:{e}"

        summaries.append({
            "input_file": str(src),
            "output_file": str(dst),
            "rows_input": len(df),
            "rows_kept": len(out_df),
            "pct_kept": round(100.0 * len(out_df) / max(len(df), 1), 2),
            "status": status,
        })

    return pd.DataFrame(summaries)
