# max_30_min_rainfall.py
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
from typing import Union, List, Optional, Tuple
import pandas as pd
import numpy as np

# Required columns (exact names)
REQUIRED_MIN = [
    "time",
    "duration of interval (min)",
    "rain depth in interval (mm)",
    "stid",
]

# Expect filenames like STID_YYYY_MM_EVENT.csv (EVENT optional)
NAME_RE = re.compile(r"^(?P<stid>[^_]+)_(?P<year>\d{4})_(?P<month>\d{2})(?:_(?P<event>\d+))?\.csv$")

# Rolling windows (minutes)
ROLL_WINDOWS = [5, 10, 15, 20, 30]


# ---------- Helpers ----------
def _collect_paths(path: str, recursive: bool) -> List[str]:
    if os.path.isfile(path):
        return [path]
    pattern = os.path.join(path, "**", "*.csv") if recursive else os.path.join(path, "*.csv")
    return sorted(glob.glob(pattern, recursive=recursive))

def _parse_name_parts(path: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None, None, None, None
    stid = m.group("stid")
    year = int(m.group("year"))
    month = int(m.group("month"))
    event = int(m.group("event")) if m.group("event") is not None else None
    return stid, year, month, event

def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    # numeric arrays
    df["_rain_mm"] = pd.to_numeric(df["rain depth in interval (mm)"], errors="coerce").fillna(0.0).values
    df["_dur_min"] = pd.to_numeric(df["duration of interval (min)"], errors="coerce").fillna(0.0).values
    return df

def _rolling_sum_with_min_duration(df: pd.DataFrame, minutes: int) -> np.ndarray:
    """
    Duration-aware rolling sum:
    At each row i, sum 'rain depth in interval (mm)' over rows whose times are within
    [time[i]-minutes, time[i]], but only return a value if the sum of 'duration of interval (min)'
    in that window is >= minutes; else 0.0.
    """
    # Convert times to integer nanoseconds for robust searchsorted
    times_ns = df["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    rain = df["_rain_mm"]
    dur  = df["_dur_min"]
    n = len(df)

    # Prefix sums
    rain_ps = np.concatenate(([0.0], np.cumsum(rain)))
    dur_ps  = np.concatenate(([0.0], np.cumsum(dur)))

    out = np.zeros(n, dtype=float)
    window_ns = np.int64(minutes) * 60 * 1_000_000_000

    for i in range(n):
        end_ns = times_ns[i]
        start_ns = end_ns - window_ns
        # left bound: first index with times_ns >= start_ns
        left = np.searchsorted(times_ns, start_ns, side="left")
        right = i + 1  # prefix sums are [0..n], so right = i+1
        dur_sum = dur_ps[right] - dur_ps[left]
        if dur_sum >= minutes:
            out[i] = rain_ps[right] - rain_ps[left]
        else:
            out[i] = 0.0
    return out

def _add_rolling_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for m in ROLL_WINDOWS:
        df[f"{m}-min rainfall (mm)"] = _rolling_sum_with_min_duration(df, m)
    return df

def _pick_output_path(output_dir: str, fpath: str, df: pd.DataFrame) -> str:
    """
    Save to: <output_dir>/<stid>/<year>/<stid>_<year>_<month>_<event>.csv when possible.
    Fallbacks:
      - If event missing, keep original basename under <stid>/<year>/
      - If parsing fails, mirror basename under <output_dir>/<stid>/<year>/
    """
    stid_f, year_f, month_f, event_f = _parse_name_parts(fpath)

    stid_col = None
    if "stid" in df.columns and df["stid"].notna().any():
        stid_col = str(df["stid"].dropna().iloc[0]).strip()

    year_col = month_col = None
    t0 = pd.to_datetime(df["time"], errors="coerce").dropna()
    if len(t0):
        year_col  = int(t0.dt.year.iloc[0])
        month_col = int(t0.dt.month.iloc[0])

    stid  = stid_f or stid_col or "unknown"
    year  = year_f  or year_col
    month = month_f or month_col
    event = event_f

    if stid and year is not None and month is not None and event is not None:
        out_dir = os.path.join(output_dir, stid, f"{year:04d}")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{stid}_{year:04d}_{month:02d}_{event}.csv")

    if stid and year is not None:
        out_dir = os.path.join(output_dir, stid, f"{year:04d}")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, os.path.basename(fpath))

    out_dir = os.path.join(output_dir, stid or "unknown", str(year or "unknown_year"))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, os.path.basename(fpath))

def _ordered_output_columns(df: pd.DataFrame) -> List[str]:
    base = [
        "time",
        "cumulative rain depth (mm)",
        "duration of interval (min)",
        "rain depth in interval (mm)",
        "rainfall intensity (mm/hr)",
        "stid",
        "rain",
        "unit energy (MJ/ha*mm)",
        "energy in interval (MJ/ha)",
    ]
    rolls = [f"{m}-min rainfall (mm)" for m in ROLL_WINDOWS]
    return [c for c in base if c in df.columns] + [c for c in rolls if c in df.columns]


# ---------- Public API ----------
def process_max_rolling(
    input_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    recursive: bool = True,
) -> List[str]:
    """
    Compute duration-aware rolling rainfall for 5, 10, 15, 20, 30 minutes and
    write augmented CSVs to:
        <output_dir>/<stid>/<year>/<stid>_<year>_<month>_<event>.csv  (when parseable)
    Returns a list of output file paths written.
    """
    files = _collect_paths(str(input_path), recursive=recursive)
    written = []

    for fpath in files:
        try:
            df_raw = pd.read_csv(fpath)
        except Exception as e:
            print(f"read_error: {fpath}: {e}")
            continue

        # Validate required columns
        missing = [c for c in REQUIRED_MIN if c not in df_raw.columns]
        if missing:
            print(f"missing columns in {fpath}: {missing}")
            continue

        # Prep & compute
        df = _prep_df(df_raw)
        df = _add_rolling_columns(df)

        # Arrange columns similar to your example and save
        out_cols = _ordered_output_columns(df)
        out_df = df[out_cols].copy()

        out_path = _pick_output_path(str(output_dir), fpath, df)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_csv(out_path, index=False)
        written.append(out_path)

    return written


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute 5/10/15/20/30-min rolling rainfall (duration-aware).")
    ap.add_argument("--input", required=True, help="CSV file or directory of CSVs.")
    ap.add_argument("--output", required=True, help="Base output directory (e.g., max_intensity).")
    ap.add_argument("--no-recursive", action="store_true", help="If input is a folder, do not search subfolders.")
    args = ap.parse_args()

    outputs = process_max_rolling(
        input_path=args.input,
        output_dir=args.output,
        recursive=not args.no_recursive,
    )
    print(f"Wrote {len(outputs)} file(s).")
