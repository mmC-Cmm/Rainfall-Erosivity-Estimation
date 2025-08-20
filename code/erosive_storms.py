# erosive_storms.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import os
import re
import pandas as pd

__all__ = ["filter_erosive_storms"]

# Expect names like STID_YYYY_MM_EVENT.csv (STID can include letters/numbers/underscores/hyphens)
NAME_RE = re.compile(
    r"^(?P<stid>[^_]+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<event>\d+)\.csv$",
    re.IGNORECASE
)

def _parse_from_name(fname: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """Parse stid, year, month, event from a file name like STID_YYYY_MM_EVENT.csv."""
    m = NAME_RE.match(fname)
    if not m:
        return None, None, None, None
    return m.group("stid"), int(m.group("year")), int(m.group("month")), int(m.group("event"))

def _resolve_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    """Case-insensitive column resolver (does NOT rename; just finds)."""
    m = {c.lower(): c for c in df.columns}
    return m.get(wanted.lower())

def filter_erosive_storms(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    threshold_mm: float = 12.7,
    pattern: str = "**/*.csv",
    cumulative_col: str = "cumulative rain depth (mm)",
    station_col: str = "stid",
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Filter per-event CSVs by a cumulative rainfall threshold and copy qualifying
    events to: <output>/<STID>/<YEAR>/<STID>_<YEAR>_<MONTH>_<EVENT>.csv
    """
    in_dir = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(pattern))
    rows: List[Dict[str, Any]] = []

    for f in files:
        if not f.is_file() or f.suffix.lower() != ".csv":
            continue

        stid_from_name, year_from_name, month_from_name, event_from_name = _parse_from_name(f.name)
        stid_for_summary = stid_from_name or f.parent.name

        try:
            df = pd.read_csv(f)

            cum_col = _resolve_col(df, cumulative_col) or cumulative_col
            st_col  = _resolve_col(df, station_col) or station_col
            t_col   = _resolve_col(df, time_col) or time_col

            if st_col in df.columns and pd.notna(df[st_col]).any():
                station_id = str(df[st_col].dropna().astype(str).iloc[0])
            else:
                station_id = stid_for_summary

            if cum_col not in df.columns:
                rows.append({
                    "file": f.name,
                    "station_id": station_id,
                    "kept": False,
                    "reason": f"missing column '{cumulative_col}'",
                    "output_file": None
                })
                continue

            df[cum_col] = pd.to_numeric(df[cum_col], errors="coerce").fillna(0).clip(lower=0)
            max_cum = df[cum_col].max(skipna=True)

            if pd.isna(max_cum) or max_cum < threshold_mm:
                rows.append({
                    "file": f.name,
                    "station_id": station_id,
                    "kept": False,
                    "reason": f"max({cumulative_col})={max_cum} < threshold={threshold_mm}",
                    "output_file": None
                })
                continue

            year = year_from_name
            month = month_from_name
            event_no = event_from_name

            if (year is None or month is None) and (t_col in df.columns):
                t = pd.to_datetime(df[t_col], errors="coerce").dropna()
                if len(t) > 0:
                    year = year or int(t.iloc[0].year)
                    month = month or int(t.iloc[0].month)

            year = year if year is not None else 9999
            month = month if month is not None else 99

            if event_no is None:
                station_dir_existing = out_root / station_id / f"{year:04d}"
                existing = list(station_dir_existing.glob(f"{station_id}_{year:04d}_{month:02d}_*.csv"))
                event_no = len(existing) + 1

            station_safe = station_id.replace("/", "_")
            out_station_year_dir = out_root / station_safe / f"{year:04d}"
            out_station_year_dir.mkdir(parents=True, exist_ok=True)

            new_name = f"{station_safe}_{year:04d}_{month:02d}_{event_no}.csv"
            out_path = out_station_year_dir / new_name

            df.to_csv(out_path, index=False)

            rows.append({
                "file": f.name,
                "station_id": station_id,
                "kept": True,
                "max_cumulative_mm": float(max_cum),
                "threshold_mm": float(threshold_mm),
                "output_file": str(out_path)
            })

        except Exception as e:
            rows.append({
                "file": f.name,
                "station_id": stid_for_summary,
                "kept": False,
                "reason": f"error: {e}",
                "output_file": None
            })

    return pd.DataFrame(rows)
