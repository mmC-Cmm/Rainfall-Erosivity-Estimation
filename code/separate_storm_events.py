# separate_storm_events.py
# -*- coding: utf-8 -*-
"""
Separate single-storm events from interval tables and save **one CSV per event**:
  <output>/<STATION>/<YEAR>/<STATION>_<YEAR>_<MONTH>_<EVENTNUMBER>.csv

This version:
  • Accepts a directory, a single CSV, or a list of CSVs
  • Does NOT assume lowercase headers (you pass the column names you use)
  • Works with or without a station column; if missing, station id is inferred
  • Can split events whenever "rain depth in interval (mm)" == 0 (recommended)
  • Returns a summary DataFrame (one row per event)

Event definitions (choose via `split_on_interval_zero`):
  A) interval-based (default, True)
     • One event = contiguous block where <interval_col> > 0
     • Includes the row immediately BEFORE a positive block (leading 0), if it exists
     • Ends at the last >0 row (the first 0 after is not included)
  B) cumulative-based (legacy, False)
     • Start when cumulative 0 -> >0
     • End when cumulative decreases OR returns to 0
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Union, List, Dict, Any, Tuple, Callable
import pandas as pd
import numpy as np

__all__ = ["separate_storm_events"]

PathLike = Union[str, Path]
Inputs = Union[PathLike, Iterable[PathLike]]

# Columns to set to 0 on the FIRST row of each extracted event (if present)
DEFAULT_COLUMNS_TO_RESET = [
    "cumulative rain depth (mm)",
    "duration of interval (min)",
    "rain depth in interval (mm)",
    "rainfall intensity (mm/hr)",
]

# ----------------------- helpers -----------------------

def _iter_input_files(inputs: Inputs, recursive: bool) -> List[Path]:
    """Normalize `inputs` into a sorted list of CSV Paths."""
    if isinstance(inputs, (str, Path)):
        p = Path(inputs)
        if p.is_file():
            return [p]
        if p.is_dir():
            pat = "**/*.csv" if recursive else "*.csv"
            return sorted(q for q in p.glob(pat) if q.is_file())
        raise FileNotFoundError(f"Input path not found: {p}")
    files: List[Path] = []
    for item in inputs:
        q = Path(item)
        if q.is_file() and q.suffix.lower() == ".csv":
            files.append(q)
    return sorted(files)

def _default_station_id(fp: Path) -> str:
    """Infer station from filename 'STID_...' else parent folder, else stem."""
    stem = fp.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return fp.parent.name or stem

def _zero_first_row(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Set the first row of each listed column to 0 (if the column exists)."""
    if df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df.iat[0, df.columns.get_loc(c)] = 0
    return df

def _positive_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx) inclusive for contiguous True runs in mask.
    Example: [F,F,T,T,T,F,T] -> [(2,4),(6,6)]
    """
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs

def _event_stats(event_df: pd.DataFrame, time_col: Optional[str], cum_col: str) -> Dict[str, Any]:
    """Compute simple per-event stats; tolerant to missing columns."""
    # Time-based stats
    if time_col and time_col in event_df.columns:
        t = pd.to_datetime(event_df[time_col], errors="coerce")
        t0 = t.iloc[0]
        t1 = t.iloc[-1]
        dur = (t1 - t0).total_seconds() / 60.0 if (pd.notna(t0) and pd.notna(t1)) else np.nan
        year = int(t0.year) if pd.notna(t0) else 9999
        month = int(t0.month) if pd.notna(t0) else 99
    else:
        t0 = t1 = pd.NaT
        dur = np.nan
        year = 9999
        month = 99

    # Depth-based stats
    total_mm = np.nan
    if cum_col in event_df.columns:
        cum_vals = pd.to_numeric(event_df[cum_col], errors="coerce").fillna(0.0).to_numpy()
        if cum_vals.size:
            total_mm = float(cum_vals.max() - cum_vals[0])

    peak_int = np.nan
    if "rainfall intensity (mm/hr)" in event_df.columns:
        peak_int = float(pd.to_numeric(event_df["rainfall intensity (mm/hr)"], errors="coerce").fillna(0).max())

    return {
        "event_start": t0,
        "event_end": t1,
        "event_duration_min": round(dur, 2) if pd.notna(dur) else np.nan,
        "event_total_depth_mm": round(total_mm, 2) if pd.notna(total_mm) else np.nan,
        "event_peak_intensity_mm_hr": round(peak_int, 2) if pd.notna(peak_int) else np.nan,
        "year": year,
        "month": month,
    }

# ---------------- core segmentation ----------------

def _extract_events_for_station(
    df: pd.DataFrame,
    station_id: str,
    cum_col: str,
    reset_cols: List[str],
    time_col: Optional[str],
    starting_event_id: int = 0,
    *,
    split_on_interval_zero: bool = True,
    interval_col: str = "rain depth in interval (mm)",
    zero_tol: float = 1e-12,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract events for a single station.

    If split_on_interval_zero=True (recommended):
      • One event = contiguous run where interval_col > 0
      • Include the immediate row BEFORE the run if it exists (leading zero)
      • End at last >0 row (the first zero after that is not included)

    Else (legacy cumulative-based):
      • Start at cumulative 0 -> >0
      • End when cumulative decreases OR returns to 0
    """
    events: List[Dict[str, Any]] = []
    ev_id = starting_event_id

    if split_on_interval_zero:
        if interval_col not in df.columns:
            raise KeyError(f"Missing required interval column '{interval_col}' for zero-split mode.")

        interval = pd.to_numeric(df[interval_col], errors="coerce").fillna(0.0).to_numpy()
        pos = interval > zero_tol
        runs = _positive_runs(pos)

        for i0, i1 in runs:
            start_idx = max(0, i0 - 1)  # include leading zero row if present
            end_idx = i1
            ev_df = df.iloc[start_idx:end_idx + 1].copy()
            ev_df = _zero_first_row(ev_df, reset_cols)
            stats = _event_stats(ev_df, time_col, cum_col)

            ev_id += 1
            events.append({
                "stid": station_id,
                "event_id": ev_id,
                "year": stats["year"],
                "month": stats["month"],
                "event_df": ev_df,
                **{k: v for k, v in stats.items() if k not in {"year", "month"}}
            })

        return events, ev_id

    # ---- cumulative-based (legacy) ----
    c = pd.to_numeric(df[cum_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    n = len(c)
    i_start = None
    in_storm = False

    for i in range(n - 1):
        cur_val = float(c[i])
        nxt_val = float(c[i + 1])

        # Start: 0 -> >0
        if (abs(cur_val) <= 1e-12) and (nxt_val > 0) and (not in_storm):
            i_start = i
            in_storm = True

        # End: reset (nxt < cur) OR next is 0
        elif in_storm and ((nxt_val < cur_val) or (abs(nxt_val) <= 1e-12)):
            ev_df = df.iloc[i_start:i + 1].copy()
            ev_df = _zero_first_row(ev_df, reset_cols)
            stats = _event_stats(ev_df, time_col, cum_col)

            ev_id += 1
            events.append({
                "stid": station_id,
                "event_id": ev_id,
                "year": stats["year"],
                "month": stats["month"],
                "event_df": ev_df,
                **{k: v for k, v in stats.items() if k not in {"year", "month"}}
            })

            in_storm = False
            i_start = i + 1 if abs(nxt_val) <= 1e-12 else None

    if in_storm and i_start is not None:
        ev_df = df.iloc[i_start:n].copy()
        ev_df = _zero_first_row(ev_df, reset_cols)
        stats = _event_stats(ev_df, time_col, cum_col)
        ev_id += 1
        events.append({
            "stid": station_id,
            "event_id": ev_id,
            "year": stats["year"],
            "month": stats["month"],
            "event_df": ev_df,
            **{k: v for k, v in stats.items() if k not in {"year", "month"}}
        })

    return events, ev_id

# ----------------------- public API -----------------------

def separate_storm_events(
    inputs: Inputs,
    output_dir: PathLike,
    *,
    recursive: bool = True,
    station_col: Optional[str] = "stid",
    cumulative_col: str = "cumulative rain depth (mm)",
    time_col: Optional[str] = "time",
    columns_to_reset: Optional[List[str]] = None,
    station_id_func: Optional[Callable[[Path, pd.DataFrame], str]] = None,
    # NEW options for interval-based splitting:
    split_on_interval_zero: bool = True,
    interval_col: str = "rain depth in interval (mm)",
    zero_tol: float = 1e-12,
) -> pd.DataFrame:
    """
    Separate storm events and save **one CSV per event**:
        <output>/<STATION>/<YEAR>/<STATION>_<YEAR>_<MONTH>_<EVENTNUMBER>.csv

    Parameters
    ----------
    inputs : path | list[path]
        Directory, single CSV, or list of CSVs to process.
    output_dir : path
        Destination root folder.
    recursive : bool
        If inputs is a directory, search subfolders for CSVs.
    station_col : Optional[str]
        Column containing station id. If None or missing, station is inferred from file path.
    cumulative_col : str
        Name of cumulative rainfall column (mm).
    time_col : Optional[str]
        Timestamp column (optional, used for stats and year/month inference).
    columns_to_reset : list[str] | None
        Columns to zero in the first row of an event (defaults provided).
    station_id_func : callable(Path, DataFrame) -> str, optional
        Override station id inference from path/data.
    split_on_interval_zero : bool
        If True, split events on zeros in `interval_col` (recommended).
        If False, use legacy cumulative-based logic.
    interval_col : str
        Interval depth column to use when `split_on_interval_zero=True`.
    zero_tol : float
        Numerical tolerance for treating intervals as zero.

    Returns
    -------
    pandas.DataFrame
        One row per event with file, station_id, event_id, year, month, stats, and output path.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(inputs, recursive=recursive)
    if not files:
        return pd.DataFrame(columns=[
            "file", "station_id", "event_id", "year", "month",
            "event_start", "event_end", "event_duration_min",
            "event_total_depth_mm", "event_peak_intensity_mm_hr",
            "output_file", "note", "error"
        ])

    if columns_to_reset is None:
        columns_to_reset = list(DEFAULT_COLUMNS_TO_RESET)

    station_id_func = station_id_func or (lambda p, df: _default_station_id(p))

    event_summary_rows: List[Dict[str, Any]] = []
    station_event_counter: Dict[str, int] = {}

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            event_summary_rows.append({
                "file": file_path.name, "station_id": None, "event_id": None,
                "year": None, "month": None, "output_file": None, "error": f"read_error: {e}"
            })
            continue

        # Required columns check (minimal)
        missing_cols = []
        if cumulative_col not in df.columns:
            missing_cols.append(cumulative_col)
        if split_on_interval_zero and (interval_col not in df.columns):
            missing_cols.append(interval_col)

        if missing_cols:
            event_summary_rows.append({
                "file": file_path.name, "station_id": None, "event_id": None,
                "year": None, "month": None, "output_file": None,
                "error": f"missing required column(s): {missing_cols}"
            })
            continue

        # Optional sort by time for stable stats
        t_col = time_col if (time_col and time_col in df.columns) else None
        if t_col is not None:
            df[t_col] = pd.to_datetime(df[t_col], errors="coerce")
            df = df.sort_values(t_col, kind="mergesort").reset_index(drop=True)

        # Clean cumulative (ensure non-negative numeric)
        df[cumulative_col] = pd.to_numeric(df[cumulative_col], errors="coerce").fillna(0.0).clip(lower=0.0)

        # Partition by station (or treat whole file as one station)
        if station_col and station_col in df.columns and df[station_col].notna().any():
            stations = df[station_col].dropna().astype(str).unique().tolist()
            per_station = [(st, df[df[station_col].astype(str) == st].copy().reset_index(drop=True)) for st in stations]
        else:
            st = station_id_func(file_path, df)
            per_station = [(st, df.copy().reset_index(drop=True))]

        for stid, sub in per_station:
            start_id = station_event_counter.get(stid, 0)

            try:
                events, next_id = _extract_events_for_station(
                    sub,
                    station_id=stid,
                    cum_col=cumulative_col,
                    reset_cols=columns_to_reset,
                    time_col=t_col,
                    starting_event_id=start_id,
                    split_on_interval_zero=split_on_interval_zero,
                    interval_col=interval_col,
                    zero_tol=zero_tol,
                )
                station_event_counter[stid] = next_id
            except Exception as e:
                event_summary_rows.append({
                    "file": file_path.name, "station_id": stid, "event_id": None,
                    "year": None, "month": None, "output_file": None,
                    "error": f"event_extract_error: {e}"
                })
                continue

            if not events:
                event_summary_rows.append({
                    "file": file_path.name, "station_id": stid, "event_id": None,
                    "year": None, "month": None, "output_file": None,
                    "note": "no_events_in_file_for_station"
                })
                continue

            # Save one CSV per event
            for ev in events:
                yr   = int(ev["year"])
                mo   = int(ev["month"])
                eid  = int(ev["event_id"])
                ev_df = ev["event_df"]

                year_dir = out_root / stid / f"{yr:04d}"
                year_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{stid}_{yr:04d}_{mo:02d}_{eid}.csv"
                out_path = year_dir / out_name

                try:
                    ev_df.to_csv(out_path, index=False)
                except Exception as e:
                    event_summary_rows.append({
                        "file": file_path.name, "station_id": stid, "event_id": eid,
                        "year": yr, "month": mo, "output_file": None,
                        "error": f"write_error: {e}"
                    })
                    continue

                event_summary_rows.append({
                    "file": file_path.name,
                    "station_id": stid,
                    "event_id": eid,
                    "year": yr,
                    "month": mo,
                    "event_start": ev.get("event_start"),
                    "event_end": ev.get("event_end"),
                    "event_duration_min": ev.get("event_duration_min"),
                    "event_total_depth_mm": ev.get("event_total_depth_mm"),
                    "event_peak_intensity_mm_hr": ev.get("event_peak_intensity_mm_hr"),
                    "output_file": str(out_path)
                })

    # Summary DataFrame
    cols_order = [
        "file", "station_id", "event_id", "year", "month",
        "event_start", "event_end", "event_duration_min",
        "event_total_depth_mm", "event_peak_intensity_mm_hr",
        "output_file", "note", "error"
    ]
    out_df = pd.DataFrame(event_summary_rows)
    cols = [c for c in cols_order if c in out_df.columns] + [c for c in out_df.columns if c not in cols_order]
    return out_df[cols]
