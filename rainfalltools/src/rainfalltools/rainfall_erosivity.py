# rainfall_erosivity.py
import os
import re
import glob
import argparse
from typing import Union, List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

# ---------------- imports & constants (must be at top) ----------------
# These must exactly match the lowercase headers in your CSV files
REQUIRED_COLUMNS = [
    "time",
    "stid",
    "rainfall intensity (mm/hr)",
    "rain depth in interval (mm)",
    "cumulative rain depth (mm)",
]

# Pattern for parsing file names like STID_YYYY_MM_EVENT.csv (if present)
NAME_RE = re.compile(
    r"^(?P<stid>[^_]+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<event>\d+)\.csv$",
    re.IGNORECASE
)

# Column order for per-event rows we write out
OUTPUT_COL_ORDER = [
    "stid",
    "year",
    "month",
    "day",
    "total rainfall for single event (mm)",
    "rainfall @ max 30-min (mm)",
    "max 30-min intensity (mm/h)",
    "total energy (MJ/ha)",
    "rainfall erosivity ((MJ-mm)/(ha-hr))",
    "storm file",
]


# ---------------- utilities ----------------
def _resolve_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    """Case-insensitive resolution; returns actual column name or None."""
    m = {c.lower(): c for c in df.columns}
    return m.get(wanted.lower())


def _validate_columns(df: pd.DataFrame, file_path: str) -> Dict[str, str]:
    """
    Validate presence of required columns (case-insensitive).
    Returns a dict mapping logical -> actual column names.
    Raises ValueError if missing.
    """
    found: Dict[str, str] = {}
    missing = []
    for want in REQUIRED_COLUMNS:
        actual = _resolve_col(df, want)
        if actual is None:
            missing.append(want)
        else:
            found[want] = actual
    if missing:
        raise ValueError(
            f"Missing required columns in {file_path}: {missing}\n"
            f"Available: {list(df.columns)}"
        )
    return found


def _collect_csvs(input_dir: Union[str, os.PathLike], recursive: bool = True) -> List[str]:
    pattern = os.path.join(str(input_dir), "**/*.csv") if recursive else os.path.join(str(input_dir), "*.csv")
    return sorted(glob.glob(pattern, recursive=recursive))


def _parse_name_parts(path: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """Extract (stid, year, month, event) from STID_YYYY_MM_EVENT.csv (None if not match)."""
    fname = os.path.basename(path)
    m = NAME_RE.match(fname)
    if not m:
        return None, None, None, None
    return m.group("stid"), int(m.group("year")), int(m.group("month")), int(m.group("event"))


# ---------------- computations ----------------
def _add_intensity_and_energy(df: pd.DataFrame,
                              col_intensity: str,
                              col_interval_mm: str) -> pd.DataFrame:
    """Add kinetic energy terms and rolling rainfall (mm) for 10–30 min windows."""
    df = df.copy()

    inten = pd.to_numeric(df[col_intensity], errors="coerce").fillna(0.0)
    interval_mm = pd.to_numeric(df[col_interval_mm], errors="coerce").fillna(0.0)

    # Unit energy (MJ/ha*mm)
    df["unit energy (MJ/ha*mm)"] = 0.29 * (1 - 0.72 * np.exp(-0.082 * inten))

    # Energy in the interval (MJ/ha)
    df["energy in interval (MJ/ha)"] = df["unit energy (MJ/ha*mm)"] * interval_mm

    # 5-min rainfall (same as interval depth here)
    df["5-min rainfall (mm)"] = interval_mm

    # Rolling sums for 10–30 min (assumes 5-min timestep)
    for minutes in [10, 15, 20, 25, 30]:
        win = minutes // 5
        col = f"{minutes}-min rainfall (mm)"
        # require full window so that max-search behavior is consistent
        df[col] = df["5-min rainfall (mm)"].rolling(window=win, min_periods=win).sum().fillna(0)

    return df


def _summarize_event(df: pd.DataFrame, time_col: str, cum_mm_col: str) -> dict:
    """Compute event-level stats required for erosivity (all-lowercase keys)."""
    t = pd.to_datetime(df[time_col], errors="coerce")
    month = int(t.dt.month.iloc[0]) if len(t) else None
    day   = int(t.dt.day.iloc[0]) if len(t) else None
    year  = int(t.dt.year.iloc[0]) if len(t) else None

    cum_vals = pd.to_numeric(df[cum_mm_col], errors="coerce").fillna(0.0)
    total_rain = float(cum_vals.max())

    # I30 proxy: if storm <30 min (no 30-min window), default 2*total_rain (AH 703 logic)
    i30_default = total_rain * 2.0
    if "30-min rainfall (mm)" in df.columns and df["30-min rainfall (mm)"].max() != 0:
        max_row = int(df["30-min rainfall (mm)"].idxmax())
        rainfall_at_max30 = float(cum_vals.iloc[max_row])
        i30 = rainfall_at_max30 * 2.0
    else:
        rainfall_at_max30 = i30_default / 2.0
        i30 = i30_default

    total_energy = float(pd.to_numeric(df["energy in interval (MJ/ha)"], errors="coerce").fillna(0).sum())
    erosivity = total_energy * i30

    return {
        "year": year,
        "month": month,
        "day": day,
        "total rainfall for single event (mm)": total_rain,
        "rainfall @ max 30-min (mm)": rainfall_at_max30,
        "max 30-min intensity (mm/h)": i30,
        "total energy (MJ/ha)": total_energy,
        "rainfall erosivity ((MJ-mm)/(ha-hr))": erosivity,
    }


# ---------------- main ----------------
def process_rainfall_erosivity(
    input_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Compute rainfall erosivity for all storm CSVs in input_dir and:
      • Append one single-row *event* record into:
          <output_dir>/<stid>/<stid>_<year>_<month>.csv
        (Multiple storms in the same month are appended to the same file.)
      • Also write a combined table to:
          <output_dir>/all_events_erosivity.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = _collect_csvs(input_dir, recursive=recursive)

    all_rows = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            all_rows.append({"storm file": os.path.basename(file_path), "error": f"read_error: {e}"})
            continue

        # Validate + map actual columns (case-insensitive)
        try:
            mapping = _validate_columns(df, file_path)
        except ValueError as e:
            all_rows.append({"storm file": os.path.basename(file_path), "error": str(e)})
            continue

        time_col     = mapping["time"]
        stid_col     = mapping["stid"]
        inten_col    = mapping["rainfall intensity (mm/hr)"]
        interval_col = mapping["rain depth in interval (mm)"]
        cum_mm_col   = mapping["cumulative rain depth (mm)"]

        # Clean types & order
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col, kind="mergesort").reset_index(drop=True)
        df[inten_col] = pd.to_numeric(df[inten_col], errors="coerce").fillna(0)
        df[interval_col] = pd.to_numeric(df[interval_col], errors="coerce").fillna(0)
        df[cum_mm_col] = pd.to_numeric(df[cum_mm_col], errors="coerce").fillna(0)

        # Derived metrics
        df = _add_intensity_and_energy(df, col_intensity=inten_col, col_interval_mm=interval_col)
        stats = _summarize_event(df, time_col=time_col, cum_mm_col=cum_mm_col)

        # Station ID
        if stid_col in df.columns and not df[stid_col].isna().all():
            stid = str(df[stid_col].dropna().astype(str).iloc[0])
        else:
            # fallback to filename or parent folder if needed
            stid_from_name, _, _, _ = _parse_name_parts(file_path)
            stid = stid_from_name or os.path.basename(os.path.dirname(file_path)) or "unknown"

        # Prefer parts from filename; fallback to stats (time)
        stid_n, year_n, month_n, _event_n = _parse_name_parts(file_path)
        year_final = year_n if year_n is not None else stats.get("year")
        month_final = month_n if month_n is not None else stats.get("month")
        year_final = int(year_final) if year_final is not None else 9999
        month_final = int(month_final) if month_final is not None else 99

        # Output path: <output_dir>/<stid>/<stid>_<year>_<month>.csv
        st_dir = os.path.join(output_dir, stid)
        os.makedirs(st_dir, exist_ok=True)
        out_name = f"{stid}_{year_final:04d}_{month_final:02d}.csv"
        out_path = os.path.join(st_dir, out_name)

        # Build single-row record, lowercase columns
        row = {
            "stid": stid,
            **stats,
            "storm file": os.path.basename(file_path),
        }
        # Ensure consistent column ordering when writing/appending
        out_row_df = pd.DataFrame([row])
        out_row_df = out_row_df[[c for c in OUTPUT_COL_ORDER if c in out_row_df.columns]
                                + [c for c in out_row_df.columns if c not in OUTPUT_COL_ORDER]]

        # Append (or create) per-month CSV
        write_header = not os.path.exists(out_path)
        out_row_df.to_csv(out_path, mode="a", header=write_header, index=False)

        # For the combined table
        row_with_path = {**row, "output_file": out_path}
        all_rows.append(row_with_path)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(output_dir, "all_events_erosivity.csv"), index=False)
    return summary_df


# ---------------- CLI ----------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute rainfall erosivity from per-storm CSVs.")
    p.add_argument("--input", required=True, help="Input folder with storm CSV files (e.g., erosive_storms)")
    p.add_argument("--output", required=True, help="Output folder for per-event summaries (e.g., rainfall_erosivity)")
    p.add_argument("--no-recursive", action="store_true", help="Do not search subfolders")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = process_rainfall_erosivity(
        input_dir=args.input,
        output_dir=args.output,
        recursive=not args.no_recursive,
    )
    print(f"Processed {len(df)} storms. Wrote per-event rows under: {args.output}")
