# monthly_erosivity.py
import os
import re
import glob
from pathlib import Path
from typing import Union, List, Optional
import pandas as pd

# Match names like STID_YYYY_MM*.csv
NAME_RE = re.compile(r"(?P<stid>[^/_]+)_(?P<year>\d{4})_(?P<month>\d{2})", re.IGNORECASE)

def _collect_csvs(input_dir: Union[str, os.PathLike]) -> List[str]:
    """Recursively collect all CSV files under input_dir (includes subfolders)."""
    pattern = os.path.join(str(input_dir), "**", "*.csv")
    return sorted(glob.glob(pattern, recursive=True))

def _parse_from_filename(path: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Parse (stid, year, month) from filename."""
    fname = os.path.basename(path)
    m = NAME_RE.search(fname)
    if not m:
        # fallback: station from parent folder
        stid = os.path.basename(os.path.dirname(path)) or None
        return stid, None, None
    stid = m.group("stid")
    year = int(m.group("year"))
    month = int(m.group("month"))
    return stid, year, month

def process_monthly_erosivity(
    input_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    erosivity_col: str,
    stid_col: Optional[str] = "stid",
    year_col: Optional[str] = "year",
    month_col: Optional[str] = "month",
    time_col: Optional[str] = None,
    round_decimals: Optional[int] = 2
) -> pd.DataFrame:
    """Aggregate per-storm rainfall erosivity files into monthly totals per station."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_csvs(input_dir)
    if not files:
        print(f"ℹ️ No CSV files found under {input_dir}")
        return pd.DataFrame()

    all_rows: List[pd.DataFrame] = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"❌ Failed to read {fp}: {e}")
            continue

        if erosivity_col not in df.columns:
            print(f"❌ {fp}: Missing erosivity column '{erosivity_col}'; skipping.")
            continue

        # STID
        stid_val: Optional[str] = None
        if stid_col and (stid_col in df.columns) and df[stid_col].notna().any():
            stid_val = str(df[stid_col].dropna().astype(str).iloc[0])
        if not stid_val:
            st_from_name, _, _ = _parse_from_filename(fp)
            stid_val = st_from_name or "unknown"

        # Year/Month
        if (year_col in df.columns) and (month_col in df.columns):
            year_series = pd.to_numeric(df[year_col], errors="coerce")
            month_series = pd.to_numeric(df[month_col], errors="coerce")
        elif time_col and (time_col in df.columns):
            dt = pd.to_datetime(df[time_col], errors="coerce")
            year_series = dt.dt.year
            month_series = dt.dt.month
        else:
            _, y_from_name, m_from_name = _parse_from_filename(fp)
            if y_from_name is None or m_from_name is None:
                print(f"❌ {fp}: Cannot determine year/month; skipping.")
                continue
            year_series = pd.Series([y_from_name] * len(df))
            month_series = pd.Series([m_from_name] * len(df))

        eros = pd.to_numeric(df[erosivity_col], errors="coerce").fillna(0.0)

        tmp = pd.DataFrame({
            "stid": [stid_val] * len(df),
            "year": year_series,
            "month": month_series,
            "erosivity": eros,
        }).dropna(subset=["year", "month"])

        if not tmp.empty:
            all_rows.append(tmp)

    if not all_rows:
        print(f"ℹ️ No valid rows found.")
        return pd.DataFrame()

    big = pd.concat(all_rows, ignore_index=True)

    # Group & sum
    monthly = (
        big.groupby(["stid", "year", "month"], dropna=False)["erosivity"]
           .sum()
           .reset_index()
           .rename(columns={"erosivity": "monthly_erosivity"})
    )

    if round_decimals is not None:
        monthly["monthly_erosivity"] = monthly["monthly_erosivity"].round(round_decimals)

    # Fill missing months with 0 for each station
    for st in monthly["stid"].astype(str).unique():
        sub = monthly[monthly["stid"] == st].copy()
        y_min, y_max = int(sub["year"].min()), int(sub["year"].max())

        # full year-month grid
        scaffold = pd.MultiIndex.from_product(
            [[st], range(y_min, y_max + 1), range(1, 13)],
            names=["stid", "year", "month"]
        )
        scaffold_df = pd.DataFrame(index=scaffold).reset_index()

        # merge and fill
        filled = scaffold_df.merge(sub, on=["stid", "year", "month"], how="left")
        filled["monthly_erosivity"] = filled["monthly_erosivity"].fillna(0.0)

        if round_decimals is not None:
            filled["monthly_erosivity"] = filled["monthly_erosivity"].round(round_decimals)

        out_fp = output_dir / f"{st}_monthly_erosivity.csv"
        filled.to_csv(out_fp, index=False)

    print(f"✅ Monthly erosivity written under: {output_dir}")
    return monthly
