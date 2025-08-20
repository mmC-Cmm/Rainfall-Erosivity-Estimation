# filter_high_quality_sites.py

from pathlib import Path
import pandas as pd
import shutil

def filter_high_quality_sites(
    missing_overall_csv: str | Path,
    rain_data_root: str | Path,
    output_root: str | Path,
    threshold: float = 5.0,
    dry_run: bool = False
):
    """
    Filter out sites with Missing_Percentage > threshold and copy
    the remaining site folders to output_root.

    If dry_run=True, only prints which sites would be copied.
    """
    missing_overall_csv = Path(missing_overall_csv)
    rain_data_root = Path(rain_data_root)
    output_root = Path(output_root)

    if not missing_overall_csv.exists():
        raise FileNotFoundError(f"Missing file not found: {missing_overall_csv}")
    if not rain_data_root.exists():
        raise FileNotFoundError(f"Rain data folder not found: {rain_data_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    # Read stats
    df = pd.read_csv(missing_overall_csv)
    if "STID" not in df.columns or "Missing_Percentage" not in df.columns:
        raise ValueError("missing_overall.csv must contain STID and Missing_Percentage columns.")

    # Filter sites
    keep_sites = df[df["Missing_Percentage"] <= threshold]["STID"].astype(str).tolist()

    if not keep_sites:
        print(f"⚠️ No sites meet the criteria: threshold = {threshold}%")
        return

    print(f"✅ Keeping {len(keep_sites)} sites with Missing_Percentage ≤ {threshold}%")
    if dry_run:
        print("Dry run enabled — no files will be copied.")

    for stid in keep_sites:
        src_dir = rain_data_root / stid
        if src_dir.exists() and src_dir.is_dir():
            dest_dir = output_root / stid
            if dry_run:
                print(f"   Would copy {stid}")
            else:
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_dir, dest_dir)
                print(f"   Copied {stid}")
        else:
            print(f"⚠️ Source folder not found for {stid}")

    if not dry_run:
        print(f"\n📂 High-quality sites saved to: {output_root}")
