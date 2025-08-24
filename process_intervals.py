# process_intervals.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union
import argparse
import numpy as np
import pandas as pd

PathLike = Union[str, Path]
Inputs = Union[PathLike, Iterable[PathLike]]

_OUTPUT_COLS = [
    "time",
    "cumulative rain depth (mm)",
    "duration of interval (min)",
    "rain depth in interval (mm)",
    "rainfall intensity (mm/hr)",
]


def _iter_input_files(inputs: Inputs, recursive: bool) -> List[Path]:
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


def _default_station_id(file_path: Path) -> str:
    stem = file_path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return file_path.parent.name or stem


def _process_dataframe(df: pd.DataFrame, rain_col: str, time_col: str) -> pd.DataFrame:
    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0)
    df.loc[df[rain_col] < 0, rain_col] = 0

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    df["cumulative rain depth (mm)"] = df[rain_col]
    df["duration of interval (min)"] = (
        df[time_col].diff().dt.total_seconds().div(60).fillna(0)
    )
    df["rain depth in interval (mm)"] = (
        df["cumulative rain depth (mm)"].diff().fillna(0).clip(lower=0)
    )

    dur_hr = df["duration of interval (min)"] / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inten = df["rain depth in interval (mm)"] / dur_hr.replace(0, np.nan)
    df["rainfall intensity (mm/hr)"] = inten.fillna(0)

    prefer = [c for c in _OUTPUT_COLS if c in df.columns]
    extras = [c for c in df.columns if c not in prefer]
    return df[prefer + extras]


def process_intervals(
    inputs: Inputs,
    output_dir: PathLike,
    *,
    recursive: bool = True,
    rain_col: str = "rain",
    time_col: str = "time",
    station_id_func: Optional[Callable[[Path], str]] = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    inputs : directory, single file, or iterable of files
    output_dir : root output directory
    recursive : search subfolders when inputs is a directory
    rain_col, time_col : expected input column names
    station_id_func : optional function Path -> station id string
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(inputs, recursive)
    if not files:
        if not quiet:
            print(f"No CSV files found for inputs: {inputs}")
        return pd.DataFrame(columns=["input_file", "status", "written_files"])

    station_id_func = station_id_func or _default_station_id
    rows: List[Dict[str, object]] = []

    for src in files:
        try:
            df = pd.read_csv(src)
        except Exception as e:
            rows.append({"input_file": str(src), "status": f"read_error: {e}", "written_files": 0})
            continue

        try:
            proc = _process_dataframe(df, rain_col=rain_col, time_col=time_col)
        except Exception as e:
            rows.append({"input_file": str(src), "status": f"process_error: {e}", "written_files": 0})
            continue

        years = pd.to_datetime(proc[time_col], errors="coerce").dt.year
        months = pd.to_datetime(proc[time_col], errors="coerce").dt.month
        proc["__year__"] = years
        proc["__month__"] = months

        stid = station_id_func(src)
        written = 0

        for (yr, mo), g in proc.groupby(["__year__", "__month__"]):
            if pd.isna(yr) or pd.isna(mo):
                continue
            year_dir = out_root / stid / f"{int(yr):04d}"
            year_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{stid}_{int(yr):04d}{int(mo):02d}.csv"
            g.drop(columns=["__year__", "__month__"]).to_csv(year_dir / out_name, index=False)
            written += 1

        rows.append({"input_file": str(src), "status": "ok", "written_files": written})

    if not quiet:
        print(f"✅ Done! Wrote {sum(r['written_files'] for r in rows)} monthly file(s) under: {out_root}")

    return pd.DataFrame(rows)


# --- Notebook-friendly wrapper to match your call ---
def process_directory(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    recursive: bool = True,
    rain_col: str = "rain",
    time_col: str = "time",
    station_id_func: Optional[Callable[[Path], str]] = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper so notebooks can call:
        process_intervals.process_directory(input_dir=..., output_dir=..., ...)
    """
    return process_intervals(
        inputs=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        rain_col=rain_col,
        time_col=time_col,
        station_id_func=station_id_func,
        quiet=quiet,
    )


# Optional CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Process rainfall interval data.")
    p.add_argument("-i", "--input", required=True, help="Directory, file, or list of files.")
    p.add_argument("-o", "--output", required=True, help="Output directory.")
    p.add_argument("--nonrecursive", action="store_true", help="If directory input, do not search subfolders.")
    p.add_argument("--rain", default="rain", help="Cumulative rainfall column name.")
    p.add_argument("--time", default="time", help="Timestamp column name.")
    args = p.parse_args()

    process_intervals(
        inputs=args.input,
        output_dir=args.output,
        recursive=not args.nonrecursive,
        rain_col=args.rain,
        time_col=args.time,
    )
