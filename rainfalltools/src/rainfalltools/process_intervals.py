# process_intervals.py
# -*- coding: utf-8 -*-
"""
Process 5-min rainfall files (cumulative, mm) and save by STID/year/month:
  <output>/<STID>/<YEAR>/<STID>_<YYYY><MM>.csv

Assumes columns (case-insensitive): stid (optional), time, rain
- Negative rain value is set to 0
- Adds interval duration (min), interval depth (mm), intensity (mm/hr)

CLI:
  python process_intervals.py --input storms_identification --output storm_interval_information

Import:
  from process_intervals import process_directory
  process_directory("storms_identification", "storm_interval_information")
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# manually lowercase names
OUTPUT_COLS = [
    "stid",
    "time",
    "cumulative rain depth (mm)",
    "duration of interval (min)",
    "rain depth in interval (mm)",
    "rainfall intensity (mm/hr)",
]

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make column matching case-insensitive without renaming everything."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]  # strip spaces only
    return out

def _process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a processed copy with interval metrics (mm-only)."""
    df = _norm_cols(df)
    cols_lower = {c.lower(): c for c in df.columns}
    if "rain" not in cols_lower or "time" not in cols_lower:
        raise KeyError("Input must contain 'rain' and 'time' columns (any case).")

    rain_col = cols_lower["rain"]
    time_col = cols_lower["time"]

    # clean rain
    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce")
    df.loc[df[rain_col] < 0, rain_col] = 0

    # time
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col, kind="mergesort").reset_index(drop=True)

    # cumulative (mm)
    df["cumulative rain depth (mm)"] = df[rain_col]

    # duration (min)
    df["duration of interval (min)"] = df[time_col].diff().dt.total_seconds().div(60).fillna(0)

    # interval depth (mm), clip tiny negatives
    df["rain depth in interval (mm)"] = df["cumulative rain depth (mm)"].diff().fillna(0).clip(lower=0)

    # intensity (mm/hr)
    dur_hr = df["duration of interval (min)"] / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inten = df["rain depth in interval (mm)"] / dur_hr.replace(0, np.nan)
    df["rainfall intensity (mm/hr)"] = inten.fillna(0)

    # arrange columns (keep stid if present)
    desired = [c for c in OUTPUT_COLS if c in df.columns]
    others = [c for c in df.columns if c not in desired]
    return df[desired + others]

def _infer_stid_for_group(g: pd.DataFrame, fallback: str) -> str:
    """Get STID from column if present, else use fallback (e.g., filename stem or folder)."""
    cols_lower = {c.lower(): c for c in g.columns}
    if "stid" in cols_lower and g[cols_lower["stid"]].notna().any():
        return str(g[cols_lower["stid"]].dropna().astype(str).iloc[0])
    return fallback

def process_directory(input_dir: str | Path, output_dir: str | Path, recursive: bool = True) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"No CSV files found in {in_dir}")
        return

    for src in files:
        if not (src.is_file() and src.suffix.lower() == ".csv"):
            continue

        try:
            raw = pd.read_csv(src)
        except Exception as e:
            print(f"⚠️  Failed to read {src}: {e}")
            continue

        try:
            df_proc = _process_df(raw)
        except Exception as e:
            print(f"⚠️  Skipped {src.name}: {e}")
            continue

        # derive year & month
        df_proc["__year__"] = pd.to_datetime(df_proc["time"], errors="coerce").dt.year
        df_proc["__month__"] = pd.to_datetime(df_proc["time"], errors="coerce").dt.month

        # split by STID (if present) then by year+month
        cols_lower = {c.lower(): c for c in df_proc.columns}
        if "stid" in cols_lower:
            stid_groups = df_proc.groupby(cols_lower["stid"], dropna=False)
        else:
            # no stid column: treat entire file as one station; fallback id from filename stem
            fallback_stid = src.stem.split("_")[0]
            stid_groups = [(fallback_stid, df_proc)]

        for stid_key, df_st in stid_groups:
            stid_val = _infer_stid_for_group(df_st, str(stid_key))

            for (yr, mo), g in df_st.groupby(["__year__", "__month__"]):
                if pd.isna(yr) or pd.isna(mo):
                    continue
                year_dir = out_dir / stid_val / f"{int(yr):04d}"
                year_dir.mkdir(parents=True, exist_ok=True)

                out_name = f"{stid_val}_{int(yr):04d}{int(mo):02d}.csv"
                out_path = year_dir / out_name

                # drop helper cols and write
                g = g.drop(columns=["__year__", "__month__"])
                g.to_csv(out_path, index=False)

    print("Done!")

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert cumulative mm rain to interval depths & intensities and save by STID/year.")
    p.add_argument("--input", "-i", required=True, help="Input directory containing CSV files")
    p.add_argument("--output", "-o", required=True, help="Output directory for processed CSV files")
    p.add_argument("--nonrecursive", action="store_true", help="Process only top-level CSVs (no subfolders)")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    process_directory(args.input, args.output, recursive=not args.nonrecursive)
