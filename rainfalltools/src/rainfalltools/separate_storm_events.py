# separate_storm_events.py
# -*- coding: utf-8 -*-
"""
Separate single-storm events from interval tables (mm-only) and save per event:
  <output>/<STID>/<YEAR>/<STID>_<YEAR>_<MONTH>_<EVENTNUMBER>.csv

Assumes ALL input column names are already lowercase:
  - stid
  - time
  - cumulative rain depth (mm)
  - duration of interval (min)           [optional but useful for summaries]
  - rain depth in interval (mm)          [optional]
  - rainfall intensity (mm/hr)           [optional]

Event definition:
  - Start when cumulative goes 0 -> >0
  - End when cumulative decreases or returns to 0

Event_ID is continuous per station across all files/months (does not reset).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

__all__ = ["separate_storm_events"]

DEFAULT_COLUMNS_TO_RESET = [
    "cumulative rain depth (mm)",
    "duration of interval (min)",
    "rain depth in interval (mm)",
    "rainfall intensity (mm/hr)",
]

def _zero_first_row(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Set first-row values in 'cols' to 0 if the columns exist."""
    if df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df.iat[0, df.columns.get_loc(c)] = 0
    return df

def _event_stats(event_df: pd.DataFrame, time_col: Optional[str], cum_col: str) -> Dict[str, Any]:
    """Compute basic stats for a single event (lowercase keys for consistency)."""
    if time_col and time_col in event_df.columns:
        t0 = pd.to_datetime(event_df[time_col], errors="coerce").iloc[0]
        t1 = pd.to_datetime(event_df[time_col], errors="coerce").iloc[-1]
        dur = (t1 - t0).total_seconds() / 60.0 if pd.notna(t0) and pd.notna(t1) else np.nan
    else:
        t0 = t1 = pd.NaT
        dur = np.nan

    cum_vals = pd.to_numeric(event_df[cum_col], errors="coerce").fillna(0)
    total_mm = float(cum_vals.max() - cum_vals.iloc[0])

    peak_int = np.nan
    if "rainfall intensity (mm/hr)" in event_df.columns:
        peak_int = float(pd.to_numeric(event_df["rainfall intensity (mm/hr)"], errors="coerce")
                         .fillna(0).max())

    return {
        "event_start": t0,
        "event_end": t1,
        "event_duration_min": round(dur, 2) if pd.notna(dur) else np.nan,
        "event_total_depth_mm": round(total_mm, 2),
        "event_peak_intensity_mm_hr": round(peak_int, 2) if pd.notna(peak_int) else np.nan,
    }

def _extract_events_for_station(
    df: pd.DataFrame,
    station_id: str,
    cum_col: str,
    reset_cols: List[str],
    time_col: Optional[str],
    starting_event_id: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract events from a single-station dataframe.
    Returns (list_of_event_records, next_event_id).
    Each event_record includes event_id, event_df, year, month, stats.
    """
    ev_id = starting_event_id
    i_start = None
    in_storm = False
    events: List[Dict[str, Any]] = []

    n = len(df)
    for i in range(n - 1):
        cur_val = df.loc[i, cum_col]
        nxt_val = df.loc[i + 1, cum_col]

        # Start: 0 -> >0
        if (cur_val == 0) and (nxt_val > 0) and (not in_storm):
            i_start = i
            in_storm = True

        # End: reset (nxt < cur) OR next is 0
        elif in_storm and ((nxt_val < cur_val) or (nxt_val == 0)):
            ev_id += 1
            ev_df = df.loc[i_start:i].copy()
            ev_df = _zero_first_row(ev_df, reset_cols)
            stats = _event_stats(ev_df, time_col, cum_col)

            if pd.notna(stats["event_start"]):
                yr = int(stats["event_start"].year)
                mo = int(stats["event_start"].month)
            else:
                yr, mo = 9999, 99

            events.append({
                "stid": station_id,
                "event_id": ev_id,
                "year": yr,
                "month": mo,
                "event_df": ev_df,
                **stats
            })

            in_storm = False
            i_start = i + 1 if nxt_val == 0 else None

    # Tail close
    if in_storm and i_start is not None:
        ev_id += 1
        ev_df = df.loc[i_start:n - 1].copy()
        ev_df = _zero_first_row(ev_df, reset_cols)
        stats = _event_stats(ev_df, time_col, cum_col)

        if pd.notna(stats["event_start"]):
            yr = int(stats["event_start"].year)
            mo = int(stats["event_start"].month)
        else:
            yr, mo = 9999, 99

        events.append({
            "stid": station_id,
            "event_id": ev_id,
            "year": yr,
            "month": mo,
            "event_df": ev_df,
            **stats
        })

    return events, ev_id

def separate_storm_events(
    input_dir: str | Path,
    output_dir: str | Path,
    recursive: bool = True,
    station_col: str = "stid",
    cumulative_col: str = "cumulative rain depth (mm)",
    columns_to_reset: Optional[List[str]] = None,
    sort_by_time_col: Optional[str] = "time",
) -> pd.DataFrame:
    """
    Separate storm events and save **one CSV per event**:
      <output>/<STID>/<YEAR>/<STID>_<YEAR>_<MONTH>_<EVENTNUMBER>.csv

    Returns a summary DataFrame (lowercase columns) with one row per event.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if columns_to_reset is None:
        columns_to_reset = list(DEFAULT_COLUMNS_TO_RESET)

    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(input_dir.glob(pattern))

    event_summary_rows: List[Dict[str, Any]] = []
    station_event_counter: Dict[str, int] = {}  # continuous IDs per station

    for file_path in files:
        try:
            df = pd.read_csv(file_path)

            # Basic checks (no case conversion—expect exact lowercase headers)
            for need in [station_col, cumulative_col]:
                if need not in df.columns:
                    raise KeyError(f"{file_path.name}: missing required column '{need}'")

            # Optional sort by time
            t_col = sort_by_time_col if sort_by_time_col and sort_by_time_col in df.columns else None
            if t_col is not None:
                df[t_col] = pd.to_datetime(df[t_col], errors="coerce")
                df = df.sort_values(t_col, kind="mergesort").reset_index(drop=True)

            # Clean cumulative: numeric, negatives -> 0
            df[cumulative_col] = pd.to_numeric(df[cumulative_col], errors="coerce").fillna(0)
            df.loc[df[cumulative_col] < 0, cumulative_col] = 0

            stations = df[station_col].dropna().astype(str).unique().tolist()
            if not stations:
                event_summary_rows.append({
                    "file": file_path.name, "station_id": None,
                    "event_id": None, "year": None, "month": None,
                    "output_file": None, "note": "no_station_ids"
                })
                continue

            for st in stations:
                sub = df[df[station_col].astype(str) == st].copy().reset_index(drop=True)

                start_id = station_event_counter.get(st, 0)
                events, next_id = _extract_events_for_station(
                    sub,
                    station_id=st,
                    cum_col=cumulative_col,
                    reset_cols=columns_to_reset,
                    time_col=t_col,
                    starting_event_id=start_id
                )
                station_event_counter[st] = next_id

                # Save one CSV per event
                for ev in events:
                    stid = ev["stid"]
                    yr   = ev["year"]
                    mo   = ev["month"]
                    eid  = ev["event_id"]
                    ev_df = ev["event_df"]

                    # Build output path and write
                    year_dir = output_dir / stid / f"{yr:04d}"
                    year_dir.mkdir(parents=True, exist_ok=True)

                    out_name = f"{stid}_{yr:04d}_{mo:02d}_{eid}.csv"
                    out_path = year_dir / out_name

                    ev_df.to_csv(out_path, index=False)

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

                # If no events found for this station in this file
                if not events:
                    event_summary_rows.append({
                        "file": file_path.name,
                        "station_id": st,
                        "event_id": None,
                        "year": None,
                        "month": None,
                        "output_file": None,
                        "note": "no_events_in_file_for_station"
                    })

        except Exception as e:
            event_summary_rows.append({
                "file": file_path.name,
                "station_id": None,
                "event_id": None,
                "year": None,
                "month": None,
                "output_file": None,
                "error": str(e)
            })

    # Build summary (lowercase columns)
    cols_order = [
        "file", "station_id", "event_id", "year", "month",
        "event_start", "event_end", "event_duration_min",
        "event_total_depth_mm", "event_peak_intensity_mm_hr",
        "output_file", "note", "error"
    ]
    out_df = pd.DataFrame(event_summary_rows)
    cols = [c for c in cols_order if c in out_df.columns] + [c for c in out_df.columns if c not in cols_order]
    return out_df[cols]
