# identify_storms.py
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import glob
import re

__all__ = ["identify_storms", "collect_csvs"]

def collect_csvs(root: Union[str, Path], recursive: bool = True) -> List[str]:
    root = str(root)
    pattern = f"{root}/**/*.csv" if recursive else f"{root}/*.csv"
    return sorted(glob.glob(pattern, recursive=recursive))

# matches STID_YYYYMM or STID_YYYY_MM
NAME_RE = re.compile(r"(?P<stid>[^/_]+)_(?P<ym>\d{4})[_-]?(?P<mm>\d{2})?")

def identify_storms(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    recursive: bool = True,
    rain_col: str = "rain",
    time_col: Optional[str] = "time",
) -> pd.DataFrame:
    """
    Detect storm rows (around positive increments in cumulative rainfall)
    for all CSV files inside input_dir. Writes filtered CSVs to:
        <output_dir>/<STID>/<YEAR>/<STID>_<YEARMONTH>.csv
    and returns a summary DataFrame.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(p) for p in collect_csvs(input_dir, recursive=recursive)]
    summaries = []

    for src in files:
        # --- parse STID / YEAR / YEARMONTH from the path/name ---
        # Preferred: folder structure .../<STID>/<YEAR>/<name>.csv
        stid = src.parent.parent.name if src.parent.parent != src.anchor else None
        year = src.parent.name if src.parent != src.anchor else None

        # Fallback: parse from filename like STID_YYYYMM or STID_YYYY_MM
        m = NAME_RE.match(src.stem)
        if m:
            stid = stid or m.group("stid")
            ym = m.group("ym") + (m.group("mm") or "")
        else:
            # last resort: use pieces of name
            ym = "".join([t for t in re.findall(r"\d+", src.stem) if len(t) in (6, 8)])[:6]

        # derive year from YEARMONTH if needed
        if not year and ym and len(ym) >= 6:
            year = ym[:4]

        if not (stid and year and ym and len(ym) >= 6):
            summaries.append({"input_file": str(src), "status": "name_parse_failed"})
            continue

        # --- read & process ---
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

        df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0)
        df.loc[df[rain_col] < 0, rain_col] = 0

        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = (
                df.dropna(subset=[time_col])
                  .sort_values(time_col, kind="mergesort")
                  .reset_index(drop=True)
            )

        inc = df[rain_col].diff().gt(0).fillna(False)
        keep_mask = inc | inc.shift(1, fill_value=False)
        out_df = df.loc[keep_mask].copy()

        # --- build output path: <output>/<STID>/<YEAR>/<STID>_<YEARMONTH>.csv ---
        dst_dir = output_dir / stid / year
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{stid}_{ym[:6]}.csv"

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
