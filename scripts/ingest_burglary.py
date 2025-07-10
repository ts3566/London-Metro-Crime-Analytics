"""
Ingest and process Metropolitan Police residential burglary data

================================================================
Run this file directly
================================================================

Outputs:
**ward_month_burglary.parquet** - ward/month table (fast Arrow format)
**ward_month_burglary.geojson** - ward/month table with geometries
**lsoa_month_burglary.parquet** - LSOA/month table (fast Arrow format)

"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd
import geopandas as gpd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default paths – change here if your layout differs ------------------------
# ---------------------------------------------------------------------------
DEFAULT_RAW_DIR   = Path("data")
DEFAULT_LOOKUP_CSV = Path("data_cache/lookups/LSOA21_WD24_Lookup.csv")
DEFAULT_GEOJSON = Path("data_cache/lookups/wards_2024.geojson")
DEFAULT_OUT_DIR   = Path("data_cache/processed")

RAW_GLOB      = "**/*-street.csv*"   #matches .csv
POLICE_FORCE  = "Metropolitan Police Service"
TARGET_CRIME  = "Burglary"
DATE_COL      = "Month"

#normalise header variants that appear over years
COL_RENAME = {
    "Crime type": "Crime type",
    "CrimeType": "Crime type",
    "Falls within": "Falls within",
    "Falls Within": "Falls within",
}

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    """Open plain CSV **or** first member inside a .zip archive."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as handle:
                return pd.read_csv(handle, low_memory=False, **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


def ingest_raw(raw_dir: Path) -> pd.DataFrame:
    """Concatenate monthly files; keep Met residential burglaries only."""
    csvs = sorted(raw_dir.glob(RAW_GLOB))
    if not csvs:
        raise FileNotFoundError(f"No '*-street.csv' files found under {raw_dir!s}")

    frames: list[pd.DataFrame] = []
    for p in tqdm(csvs, desc="Reading CSVs"):
        df = _read_csv_any(p)
        df.rename(columns={c: COL_RENAME.get(c, c) for c in df.columns}, inplace=True)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    mask = (df["Falls within"] == POLICE_FORCE) & (df["Crime type"] == TARGET_CRIME)
    df = df.loc[mask].copy()
    df.dropna(subset=["LSOA code", DATE_COL], inplace=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m")
    return df


def attach_ward(df: pd.DataFrame, lookup_csv: Path) -> pd.DataFrame:
    """Merge LSOA→Ward lookup. Drops rows without ward match."""
    look = pd.read_csv(lookup_csv, usecols=["LSOA21CD", "WD24CD", "WD24NM"])
    merged = df.merge(look, left_on="LSOA code", right_on="LSOA21CD", how="left")
    miss = merged["WD24CD"].isna().mean()
    if miss > 0:
        print(f"  {miss:.2%} rows lacked ward code and were discarded.")
    return merged.dropna(subset=["WD24CD"])


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy ward-month panel aggregating by code, then combining duplicate names."""
    #first aggregate by code
    base = (
        df.groupby([pd.Grouper(key=DATE_COL, freq="MS"), "WD24CD", "WD24NM"])
          .size()
          .rename("burglaries")
          .reset_index()
    )
    #then combine rows sharing the same ward name across different codes
    agg_df = (
        base.groupby([DATE_COL, "WD24NM"])
            .agg(
                burglaries=("burglaries", "sum"),
                WD24CD=("WD24CD", lambda codes: codes.iloc[0])
            )
            .reset_index()
            .sort_values([DATE_COL, "WD24CD"])
    )
    return agg_df


def aggregate_lsoa(df: pd.DataFrame, lookup_csv: Path) -> pd.DataFrame:
    """Return tidy LSOA-month panel for burglaries aggregating by code then combining duplicate LSOA names."""
    lsoa_lookup = pd.read_csv(lookup_csv, usecols=["LSOA21CD", "LSOA21NM"])
    merged_df = df.merge(lsoa_lookup, left_on="LSOA code", right_on="LSOA21CD", how="left")
    merged_df.dropna(subset=["LSOA21CD", "LSOA21NM", DATE_COL], inplace=True)
    #first aggregate by LSOA code
    base = (
        merged_df.groupby([pd.Grouper(key=DATE_COL, freq="MS"), "LSOA21CD", "LSOA21NM"])
          .size()
          .rename("burglaries")
          .reset_index()
    )
    #then combine rows sharing the same LSOA name across different codes
    agg_df = (
        base.groupby([DATE_COL, "LSOA21NM"])
            .agg(
                burglaries=("burglaries", "sum"),
                LSOA21CD=("LSOA21CD", lambda codes: codes.iloc[0])
            )
            .reset_index()
            .sort_values([DATE_COL, "LSOA21CD"])
    )
    return agg_df


def attach_geometries(df: pd.DataFrame, geojson_path: Path) -> gpd.GeoDataFrame:
    """Attach ward geometries from GeoJSON file."""
    wards_gdf = gpd.read_file(geojson_path)
    
    #merge geometries with ward data
    merged = df.merge(wards_gdf[["WD24CD", "geometry"]], on="WD24CD")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=wards_gdf.crs)


def save_outputs(ward_df: pd.DataFrame, ward_gdf: gpd.GeoDataFrame, lsoa_df: pd.DataFrame, out_dir: Path) -> None:
    """Save parquet and GeoJSON versions for wards, and parquet for LSOAs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #save ward parquet (numeric data only)
    ward_pq = out_dir / "ward_month_burglary.parquet"
    ward_df.to_parquet(ward_pq, index=False)
    print(f"✅ Saved {len(ward_df):,} ward rows → {ward_pq}")
    
    #save ward GeoJSON (with geometries)
    ward_geojson = out_dir / "ward_month_burglary.geojson"
    ward_gdf.to_file(ward_geojson, driver="GeoJSON")
    print(f"✅ Saved ward geometries → {ward_geojson}")

    #save LSOA parquet
    lsoa_pq = out_dir / "lsoa_month_burglary.parquet"
    lsoa_df.to_parquet(lsoa_pq, index=False)
    print(f"✅ Saved {len(lsoa_df):,} LSOA rows → {lsoa_pq}")

# ---------------------------------------------------------------------------
# CLI wrapper – all args optional -------------------------------------------
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Metropolitan Police burglary data", add_help=False)
    p.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--lookup_csv", type=Path, default=DEFAULT_LOOKUP_CSV)
    p.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("-h", "--help", action="help")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.lookup_csv.exists():
        raise FileNotFoundError(
            f"Lookup file missing: {args.lookup_csv}.\n"
            "Download it (see README) and place it there, or pass --lookup_csv."
        )
    
    if not args.geojson.exists():
        raise FileNotFoundError(
            f"GeoJSON file missing: {args.geojson}.\n"
            "Download it and place it there, or pass --geojson."
        )

    print("Ingesting raw burglary CSVs …")
    crimes = ingest_raw(args.raw_dir)
    print(f"   {len(crimes):,} incidents loaded.")

    print("Mapping to wards …")
    with_w = attach_ward(crimes, args.lookup_csv)

    print("Aggregating …")
    panel_ward = aggregate(with_w)
    
    print("Aggregating by LSOA …")
    panel_lsoa = aggregate_lsoa(crimes, args.lookup_csv)
    
    print("Attaching ward geometries …")
    panel_ward_geo = attach_geometries(panel_ward, args.geojson)

    print("Writing outputs …")
    save_outputs(panel_ward, panel_ward_geo, panel_lsoa, args.out_dir)
    print("Done ✓")


if __name__ == "__main__":
    main()
