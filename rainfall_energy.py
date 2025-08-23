# rainfall_energy.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
import glob
from pathlib import Path
from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd

PathLike = Union[str, os.PathLike]

# Expect filenames like STID_YYYY_MM[_EVENT].csv (EVENT optional)
NAME_RE = re.compile(
    r"^(?P<stid>[^_]+)_(?P<year>\d{4})_(?P<month>\d{2})(?:_(?P<event>\d+))?\.csv$",
    re.IGNORECASE
)

# Output column names
COL_UNIT_E = "unit rainfall energy (MJ/ha-mm)"
COL_E_INT  = "energy in interval (MJ/ha)"
COL_E_CUM  = "cumulative energy (MJ/ha)"

def _collect_files(root: PathLike, pattern: str) -> List[Path]:
    root = Path(root)
    if (root.is_file()) and str(root).lower().endswith(".csv"):
        return [root]
    search_glob = str(root / pattern)
    return sorted(Path(p) for p in glob.glob(search_glob, recursive=True) if p.lower().endswith(".csv"))

def _infer_stid_from_name(path: Path) -> Optional[str]:
    m = NAME_RE.match(path.name)
    if m:
        return m.group("stid")
    return path.parent.name if path.parent.name else None

def _rusle_unit_energy_mm(intensity_mm_per_hr: pd.Series) -> pd.Series:
    """
    RUSLE unit rainfall energy (MJ/ha per mm):
        E_u = 0.29 * (1 - 0.72 * exp(-0.05 * I)),  I in mm/hr
    """
    I = pd.to_numeric(intensity_mm_per_hr, errors="coerce").fillna(0.0).clip(lower=0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        Eu = 0.29 * (1.0 - 0.72 * np.exp(-0.05 * I))
    Eu = pd.Series(Eu, index=intensity_mm_per_hr.index).clip(lower=0.0)
    return Eu

def process_rainfall_energy(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    intensity_col: str = "rainfall intensity (mm/hr)",
    interval_mm_col: str = "rain depth in interval (mm)",
    stid_col: Optional[str] = "stid",
    pattern: str = "**/*.csv",
    overwrite: bool = True,
    add_cumulative: bool = True,
) -> pd.DataFrame:
    """
    Compute only rainfall energy terms and write augmented CSVs, mirroring input structure.

    Adds columns:
      - 'unit rainfall energy (MJ/ha-mm)'
      - 'energy in interval (MJ/ha)'
      - 'cumulative energy (MJ/ha)' (optional)

    Output path:
      <output_dir>/<STID>/<YEAR>/<same_filename>.csv
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_files(input_dir, pattern)
    if not files:
        return pd.DataFrame(columns=["input_file", "status", "output_file"])

    rows: List[Dict[str, object]] = []

    for src in files:
        # Determine STID/YEAR for mirroring
        stid, year = None, None
        m = NAME_RE.match(src.name)
        if m:
            stid = m.group("stid")
            year = m.group("year")
        if stid is None:
            stid = _infer_stid_from_name(src) or "unknown"
        if year is None:
            py = src.parent.name
            year = py if py.isdigit() and len(py) == 4 else "0000"

        dst_dir = output_dir / stid / str(year)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if (not overwrite) and dst.exists():
            rows.append({"input_file": str(src), "status": "exists_skipped", "output_file": str(dst)})
            continue

        # Read & validate required columns
        try:
            df = pd.read_csv(src)
        except Exception as e:
            rows.append({"input_file": str(src), "status": f"read_error: {e}", "output_file": None})
            continue

        missing = [c for c in (intensity_col, interval_mm_col) if c not in df.columns]
        if missing:
            rows.append({"input_file": str(src), "status": f"missing_columns: {missing}", "output_file": None})
            continue

        # Ensure station id column exists for downstream consistency
        if stid_col:
            if (stid_col not in df.columns) or (not df[stid_col].notna().any()):
                df[stid_col] = stid

        # Compute energies
        Eu = _rusle_unit_energy_mm(df[intensity_col])
        interval_mm = pd.to_numeric(df[interval_mm_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        e_interval = Eu * interval_mm

        # Attach new columns
        df[COL_UNIT_E] = Eu.values
        df[COL_E_INT]  = e_interval.values
        if add_cumulative:
            df[COL_E_CUM] = df[COL_E_INT].cumsum()

        # Write output
        try:
            df.to_csv(dst, index=False)
            status = "ok"
        except Exception as e:
            rows.append({"input_file": str(src), "status": f"write_error: {e}", "output_file": str(dst)})
            continue

        rows.append({
            "input_file": str(src),
            "status": status,
            "output_file": str(dst),
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
        })

    return pd.DataFrame(rows)
