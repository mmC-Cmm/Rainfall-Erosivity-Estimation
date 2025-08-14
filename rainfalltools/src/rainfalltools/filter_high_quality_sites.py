
# filter_high_quality_sites.py

from pathlib import Path
import pandas as pd
import shutil

def filter_high_quality_sites(
    missing_overall_csv: str | Path,
    rain_data_root: str | Path,
    output_root: str | Path,
    threshold: float = 5.0
):
    """
    Filter out sites with Missing_Percentage > threshold and copy
    the remaining site folders to output_root.

    Parameters
    ----------
    missing_overall_csv : str | Path
        Path to missing_overall.csv file.
    rain_data_root : str | Path
        Root folder containing original Rain_Data/<STID>/...
    output_root : str | Path
        Output folder to copy high-quality sites into.
    threshold : float
        Maximum allowed missing percentage (inclusive).
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

    for stid in keep_sites:
        src_dir = rain_data_root / stid
        if src_dir.exists() and src_dir.is_dir():
            dest_dir = output_root / stid
            if dest_dir.exists():
                shutil.rmtree(dest_dir)  # remove if exists to avoid merge
            shutil.copytree(src_dir, dest_dir)
            print(f"   Copied {stid}")
        else:
            print(f"⚠️ Source folder not found for {stid}")

    print(f"\n📂 High-quality sites saved to: {output_root}")


if __name__ == "__main__":
    # Example usage
    filter_high_quality_sites(
        missing_overall_csv="Result/missing_overall.csv",
        rain_data_root="Rain_Data",
        output_root="Rain_Data_High_Quality",
        threshold=5.0
    )
