#!/usr/bin/env python3
"""
Build SARIMAX-enriched segment parquets and save to disk.

Reads:
  data/processed/preprocessed/fact_table.parquet  (via build_dataframes)
  data/processed/oil.parquet
  data/processed/transactions.parquet
  data/processed/stores.parquet
  data/processed/items.parquet
  data/processed/holidays_events.parquet

Writes:
  data/processed/preprocessed/smooth_daily_sarimax.parquet
  data/processed/preprocessed/erratic_daily_sarimax.parquet

Usage:
    python tools/build_sarima_features.py
"""

import sys
from pathlib import Path

# Make the src/ package importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Favorita_TSA.features.sarima_features import build_sarimax_segments
from Favorita_TSA.utils.paths import PREPROCESSED_DIR


def main() -> None:
    results = build_sarimax_segments()

    print()
    for key, df in results.items():
        out_path = PREPROCESSED_DIR / f"{key}_sarimax.parquet"
        print(f"Saving → {out_path.name} …")
        df.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1_000_000
        print(f"  ✓ {len(df):,} rows | {len(df.columns)} columns | {size_mb:.1f} MB")

    print("\nOutput columns:")
    for key, df in results.items():
        print(f"\n  {key}:")
        for col in df.columns:
            print(f"    {col:<35} {df[col].dtype}")


if __name__ == "__main__":
    main()
