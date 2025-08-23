# rainfall_erosivity.py
# -*- coding: utf-8 -*-
import os
import re
import glob
import argparse
from typing import Union, List, Optional, Tuple, Dict
import pandas as pd

# Expect names like STID_YYYY_MM[_EVENT].csv (EVENT optional)
NAME_RE = re.compile(
    r"^(?P<stid>[^_]+)_(?P<year>\d{4})_(?P<month>\d{2})(?:_(?P<event>\d+))?\.csv$"
)

# Required columns in each single-storm CSV
REQUIRED_COLS = [
    "time",
    "stid",
    "duration of interval (min)",
    "30-min rainfall (mm)",
    "rain",
    "energy in interval (MJ/ha)",
]

# Output column order for the per-event rows
OUTPUT_COL_ORDER = [
    "stid",
    "year",
    "month",
    "day",
    "max 30-min rainfall (mm)",
    "max 30-min intensity (mm/hr)",
    "total energy (MJ/ha)",
    "rainfall erosivity ((MJ mm)/(ha hr))",
    "storm file",
]

def _collect_csvs(input_path: Union[str, os.PathLike], recursive: bool) -> List[str]:
    """Collect CSV paths from a file or directory."""
    if os.path.isfile(input_path):
        return [str(input_path)]
    pattern = (
        os.path.join(str(input_path), "**", "*.csv")
        if recursive
        else os.path.join(str(input_path), "*.csv")
    )
    return sorted(glob.glob(pattern, recursive=recursive))

def _parse_name_parts(path: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """Parse STID, YEAR, MONTH, EVENT from filename."""
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None, None, None, None
    return (
        m.group("stid"),
        int(m.group("year")),
        int(m.group("month")),
        int(m.group("event")) if m.group("event") else None,
    )

def _summarize_event(df: pd.DataFrame, storm_file: str) -> Dict[str, object]:
    """Compute per-storm EI30 stats from a single-storm interval table."""
    t = pd.to_datetime(df["time"], errors="coerce").dropna()
    year = int(t.dt.year.iloc[0]) if len(t) else 9999
    month = int(t.dt.month.iloc[0]) if len(t) else 99
    day = int(t.dt.day.iloc[0]) if len(t) else 99

    total_duration = float(pd.to_numeric(df["duration of interval (min)"], errors="coerce").fillna(0).sum())
    max30_from_col = float(pd.to_numeric(df["30-min rainfall (mm)"], errors="coerce").fillna(0).max())
    max_cum_rain   = float(pd.to_numeric(df["rain"], errors="coerce").fillna(0).max())
    total_energy   = float(pd.to_numeric(df["energy in interval (MJ/ha)"], errors="coerce").fillna(0).sum())

    # If the storm spans at least 30 minutes of intervals, prefer the rolling-30 column;
    # else fall back to the maximum cumulative depth as a conservative proxy.
    if total_duration >= 30.0:
        max30_rainfall = max30_from_col
        I30 = 2.0 * max30_rainfall  # mm/hr
    else:
        max30_rainfall = max_cum_rain
        I30 = 2.0 * max_cum_rain

    erosivity = I30 * total_energy  # (MJ mm)/(ha hr)

    return {
        "year": year,
        "month": month,
        "day": day,
        "max 30-min rainfall (mm)": max30_rainfall,
        "max 30-min intensity (mm/hr)": I30,
        "total energy (MJ/ha)": total_energy,
        "rainfall erosivity ((MJ mm)/(ha hr))": erosivity,
        "storm file": os.path.basename(storm_file),
    }

def process_rainfall_erosivity(
    input_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    recursive: bool = True,
    write_combined: bool = True,
) -> pd.DataFrame:
    """
    Read single-storm CSVs, compute EI30 per event, and append to monthly files:
        <output_dir>/<STID>/<YEAR>/<STID>_<YEAR>_<MONTH>.csv
    Also returns and (optionally) writes a combined summary.

    Parameters
    ----------
    input_dir : str | Path
        Directory of single-storm CSV files (or a single CSV).
    output_dir : str | Path
        Root folder for per-month outputs.
    recursive : bool
        Recurse into subfolders under input_dir.
    write_combined : bool
        If True, write 'all_events_erosivity.csv' in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = _collect_csvs(input_dir, recursive)
    all_rows: List[Dict[str, object]] = []

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            all_rows.append({"storm file": os.path.basename(fpath), "error": f"read_error: {e}"})
            continue

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            all_rows.append({"storm file": os.path.basename(fpath), "error": f"missing columns: {missing}"})
            continue

        # Clean & order
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        if df.empty:
            all_rows.append({"storm file": os.path.basename(fpath), "error": "no valid timestamps"})
            continue

        # Compute event stats
        stats = _summarize_event(df, fpath)

        # Determine station and (fallback) date parts from filename and/or data
        stid_file, year_file, month_file, _ = _parse_name_parts(fpath)
        stid_col = df["stid"].dropna().astype(str).iloc[0].strip() if df["stid"].notna().any() else None

        stid = stid_file or stid_col or "unknown"
        year_final = year_file or stats["year"]
        month_final = month_file or stats["month"]

        # Output path: <output>/<STID>/<YEAR>/<STID>_<YEAR>_<MONTH>.csv
        st_dir = os.path.join(output_dir, stid, f"{year_final:04d}")
        os.makedirs(st_dir, exist_ok=True)
        out_name = f"{stid}_{year_final:04d}_{month_final:02d}.csv"
        out_path = os.path.join(st_dir, out_name)

        # Prepare one-row DataFrame in the desired column order
        row = {"stid": stid, **stats}
        out_df = pd.DataFrame([row])
        # Keep only known columns, in order
        cols = [c for c in OUTPUT_COL_ORDER if c in out_df.columns]
        out_df = out_df[cols]

        # Append to the monthly file (create header if new)
        write_header = not os.path.exists(out_path)
        try:
            out_df.to_csv(out_path, mode="a", header=write_header, index=False)
        except Exception as e:
            all_rows.append({**row, "output_file": out_path, "error": f"write_error: {e}"})
            continue

        all_rows.append({**row, "output_file": out_path})

    summary_df = pd.DataFrame(all_rows)
    if write_combined and not summary_df.empty:
        combined_path = os.path.join(output_dir, "all_events_erosivity.csv")
        summary_df.to_csv(combined_path, index=False)
    return summary_df

# -------- Optional CLI --------
def _cli():
    ap = argparse.ArgumentParser(description="Compute EI30 rainfall erosivity from per-storm interval CSVs.")
    ap.add_argument("-i", "--input", required=True, help="Input directory or single CSV.")
    ap.add_argument("-o", "--output", required=True, help="Output directory.")
    ap.add_argument("--nonrecursive", action="store_true", help="If input is a directory, do not search subfolders.")
    ap.add_argument("--no-combined", action="store_true", help="Do not write combined summary CSV.")
    args = ap.parse_args()

    df = process_rainfall_erosivity(
        input_dir=args.input,
        output_dir=args.output,
        recursive=not args.nonrecursive,
        write_combined=not args.no_combined,
    )
    print(f"✅ Done. Rows: {len(df)}  Errors: {df.get('error').notna().sum() if 'error' in df.columns else 0}")

if __name__ == "__main__":
    _cli()
