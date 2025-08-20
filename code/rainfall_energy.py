# rainfall_erosivity.py
# -*- coding: utf-8 -*-
import os
import re
import glob
import argparse
from typing import Union, List, Optional, Tuple
import pandas as pd

# Filename pattern: STID_YYYY_MM_EVENT.csv or STID_YYYY_MM.csv
NAME_RE = re.compile(r"^(?P<stid>[^_]+)_(?P<year>\d{4})_(?P<month>\d{2})(?:_(?P<event>\d+))?\.csv$")

# Exact column names expected (no lowercasing)
REQUIRED_COLS = [
    "time",
    "stid",
    "duration of interval (min)",
    "30-min rainfall (mm)",
    "rain",
    "energy in interval (MJ/ha)",
]

OUTPUT_COL_ORDER = [
    "stid",
    "year",
    "month",
    "day",
    "event",
    "total duration (min)",
    "max 30-min rainfall (mm)",
    "max 30-min intensity (mm/h)",
    "total energy (MJ/ha)",
    "rainfall erosivity ((MJ-mm)/(ha-hr))",
    "storm file",
]

def _collect_csvs(input_path: Union[str, os.PathLike], recursive: bool) -> List[str]:
    if os.path.isfile(input_path):
        return [str(input_path)]
    pattern = os.path.join(str(input_path), "**", "*.csv") if recursive else os.path.join(str(input_path), "*.csv")
    return sorted(glob.glob(pattern, recursive=recursive))

def _parse_name_parts(path: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None, None, None, None
    return (
        m.group("stid"),
        int(m.group("year")),
        int(m.group("month")),
        int(m.group("event")) if m.group("event") else None,
    )

def _summarize_event(df: pd.DataFrame, storm_file: str) -> dict:
    # Ensure proper types
    t = pd.to_datetime(df["time"], errors="coerce").dropna()
    year = int(t.dt.year.iloc[0]) if len(t) else 9999
    month = int(t.dt.month.iloc[0]) if len(t) else 99
    day = int(t.dt.day.iloc[0]) if len(t) else 99

    total_duration = float(pd.to_numeric(df["duration of interval (min)"], errors="coerce").fillna(0).sum())
    max30 = float(pd.to_numeric(df["30-min rainfall (mm)"], errors="coerce").fillna(0).max())
    max_rain = float(pd.to_numeric(df["rain"], errors="coerce").fillna(0).max())
    total_energy = float(pd.to_numeric(df["energy in interval (MJ/ha)"], errors="coerce").fillna(0).sum())

    # I30 rule
    if total_duration >= 30.0:
        I30 = max30 * (60.0 / 30.0)  # = 2 * max30
    else:
        I30 = 2.0 * max_rain

    erosivity = I30 * total_energy  # (mm/h) * (MJ/ha) = (MJ-mm)/(ha-hr)

    return {
        "year": year,
        "month": month,
        "day": day,
        "total duration (min)": total_duration,
        "max 30-min rainfall (mm)": max30,
        "max 30-min intensity (mm/h)": I30,
        "total energy (MJ/ha)": total_energy,
        "rainfall erosivity ((MJ-mm)/(ha-hr))": erosivity,
        "storm file": os.path.basename(storm_file),
    }

def process_erosivity_from_rollings(
    input_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    recursive: bool = True,
    write_combined: bool = True,
) -> pd.DataFrame:
    """
    For each augmented storm CSV (already includes '30-min rainfall (mm)'):
      - max R30
      - I30 per your rule
      - sum energy, EI30 = I30 * sum(energy in interval)
    Writes one row per event to:
        <output_dir>/<stid>/<year>/<stid>_<year>_<month>_<event>.csv  (if event known)
      or <output_dir>/<stid>/<year>/<stid>_<year>_<month>.csv         (fallback)
    Also writes <output_dir>/all_events_erosivity.csv if write_combined=True.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = _collect_csvs(input_path, recursive)
    all_rows = []

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            all_rows.append({"storm file": os.path.basename(fpath), "error": f"read_error: {e}"})
            continue

        # Validate columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            all_rows.append({"storm file": os.path.basename(fpath), "error": f"missing columns: {missing}"})
            continue

        # Sort by time for stable day extraction
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        if df.empty:
            all_rows.append({"storm file": os.path.basename(fpath), "error": "no valid timestamps"})
            continue

        # Summarize one event
        stats = _summarize_event(df, fpath)

        # Determine stid/year/month/event for output path & row
        stid_file, year_file, month_file, event_file = _parse_name_parts(fpath)
        stid_col = df["stid"].dropna().astype(str).iloc[0].strip() if df["stid"].notna().any() else None

        stid = stid_file or stid_col or os.path.basename(os.path.dirname(fpath)) or "unknown"
        year_final = year_file or stats["year"]
        month_final = month_file or stats["month"]
        event_final = event_file  # only from filename if present

        # Destination: <output_dir>/<stid>/<year>/<stid>_<year>_<month>_<event>.csv (if event), else monthly file
        st_dir = os.path.join(output_dir, stid, f"{year_final:04d}")
        os.makedirs(st_dir, exist_ok=True)
        if event_final is not None:
            out_name = f"{stid}_{year_final:04d}_{month_final:02d}_{event_final}.csv"
        else:
            out_name = f"{stid}_{year_final:04d}_{month_final:02d}.csv"
        out_path = os.path.join(st_dir, out_name)

        # Assemble row (now includes 'event' column if known)
        row = {"stid": stid, "event": event_final, **stats}
        out_df = pd.DataFrame([row])
        out_df = out_df[[c for c in OUTPUT_COL_ORDER if c in out_df.columns] +
                        [c for c in out_df.columns if c not in OUTPUT_COL_ORDER]]

        # Append row to the file (create header when new)
        write_header = not os.path.exists(out_path)
        out_df.to_csv(out_path, mode="a", header=write_header, index=False)

        all_rows.append({**row, "output_file": out_path})

    # Combined summary (optional)
    summary_df = pd.DataFrame(all_rows)
    if write_combined and not summary_df.empty:
        summary_df.to_csv(os.path.join(output_dir, "all_events_erosivity.csv"), index=False)
    return summary_df


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate rainfall erosivity from existing rolling columns.")
    parser.add_argument("--input", required=True, help="CSV file or directory of augmented CSVs (with '30-min rainfall (mm)').")
    parser.add_argument("--output", required=True, help="Base output folder, e.g. 'rainfall_erosivity'.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subfolders if --input is a directory.")
    parser.add_argument("--no-combined", action="store_true", help="Do not write the combined all_events_erosivity.csv")
    args = parser.parse_args()

    df = process_erosivity_from_rollings(
        input_path=args.input,
        output_dir=args.output,
        recursive=not args.no_recursive,
        write_combined=not args.no_combined,
    )
    print(f"Processed {len(df)} storms.")
