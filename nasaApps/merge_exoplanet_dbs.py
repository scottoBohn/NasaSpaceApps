#!/usr/bin/env python3

"""
merge_exoplanet_dbs.py

Usage:
  python merge_exoplanet_dbs.py --tess TESS.csv --kepler KEPLER.csv --nea NEA.csv --outdir ./out --prefix exo

What it does:
- Reads three input CSV files (TESS TOI-like, Kepler KOI-like, NASA Exoplanet Archive-like)
- Standardizes to generic column names
- Merges into a single dataframe (saved as {prefix}_all.pkl)
- Splits into train/test/val as 65% / 20% / 15% (saved as PKLs)
- Writes a simple column dictionary as {prefix}_columns.txt for reference
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Helpers
# -----------------------------


def _pick_series(df, keys):
    """Return the first existing column as a Series, else a NaN Series of the right length."""
    for k in keys:
        if k in df.columns:
            return df[k]
    return pd.Series([np.nan] * len(df))

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _first_present(d: Dict, keys: List[str], default=None):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return default

def _normalize_disposition(x: str):
    if not isinstance(x, str):
        return np.nan
    x = x.strip().upper()
    # Map short codes to full labels when obvious
    if x in ("CP", "CAND", "CONFIRM", "CONFIRMED"):
        return "CONFIRMED"
    if x in ("PC", "CANDIDATE"):
        return "CANDIDATE"
    if x in ("FP", "FALSE POSITIVE", "FALSE_POSITIVE"):
        return "FALSE POSITIVE"
    # Otherwise just keep uppercased original
    return x

def _add_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    df["source"] = source_name
    return df

def _standard_cols() -> List[str]:
    return [
        "host_id", "host_name", "planet_id", "source",
        "ra_deg", "dec_deg",
        "disposition",
        "period_days",
        "radius_rearth",
        "transit_depth",        # unit as in source (often ppm); left unscaled
        "transit_duration_hr",
        "transit_epoch_bjd",
        "insolation_searth",
        "eq_temp_k",
        "star_teff_k",
        "star_radius_rsun",
        "star_logg_cgs",
    ]

# -----------------------------
# Mappers for each catalog
# -----------------------------
def map_tess_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Build a standardized frame
    out = pd.DataFrame()
    out["host_id"] = df.get("tid")
    out["host_name"] = np.nan  # not present in TESS table snippet
    out["planet_id"] = df.get("toi")

    # RA/Dec: prefer numeric if available
    out["ra_deg"] = df.get("ra")
    out["dec_deg"] = df.get("dec")

    # Disposition
    out["disposition"] = df.get("tfopwg_disp").map(_normalize_disposition)

    # Planetary params
    out["period_days"] = df.get("pl_orbper")
    out["radius_rearth"] = df.get("pl_rade")
    out["transit_depth"] = df.get("pl_trandep")
    out["transit_duration_hr"] = df.get("pl_trandurh")
    out["transit_epoch_bjd"] = df.get("pl_tranmid")
    out["insolation_searth"] = df.get("pl_insol")
    out["eq_temp_k"] = df.get("pl_eqt")

    # Stellar params
    out["star_teff_k"] = df.get("st_teff")
    out["star_radius_rsun"] = df.get("st_rad")
    out["star_logg_cgs"] = df.get("st_logg")

    out = _add_source(out, "TESS")
    # Coerce numeric columns
    num_cols = [
        "ra_deg","dec_deg","period_days","radius_rearth","transit_depth",
        "transit_duration_hr","transit_epoch_bjd","insolation_searth","eq_temp_k",
        "star_teff_k","star_radius_rsun","star_logg_cgs"
    ]
    return _coerce_numeric(out, num_cols)


def map_kepler_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    out = pd.DataFrame()
    out["host_id"] = _pick_series(df, ["kepid"])
    out["host_name"] = np.nan
    out["planet_id"] = _pick_series(df, ["kepoi_name"])

    out["ra_deg"] = _pick_series(df, ["ra"])
    out["dec_deg"] = _pick_series(df, ["dec"])

    # Accept either koi_disposition or koi_pdisposition
    disp = _pick_series(df, ["koi_disposition", "koi_pdisposition"])
    out["disposition"] = disp.astype("string").map(_normalize_disposition)

    out["period_days"] = _pick_series(df, ["koi_period"])
    out["radius_rearth"] = _pick_series(df, ["koi_prad"])
    out["transit_depth"] = _pick_series(df, ["koi_depth"])
    out["transit_duration_hr"] = _pick_series(df, ["koi_duration"])
    out["transit_epoch_bjd"] = _pick_series(df, ["koi_time0bk"])
    out["insolation_searth"] = _pick_series(df, ["koi_insol"])
    out["eq_temp_k"] = _pick_series(df, ["koi_teq"])

    out["star_teff_k"] = _pick_series(df, ["koi_steff"])
    out["star_radius_rsun"] = _pick_series(df, ["koi_srad"])
    out["star_logg_cgs"] = _pick_series(df, ["koi_slogg"])

    out = _add_source(out, "KEPLER")

    num_cols = [
        "ra_deg","dec_deg","period_days","radius_rearth","transit_depth",
        "transit_duration_hr","transit_epoch_bjd","insolation_searth","eq_temp_k",
        "star_teff_k","star_radius_rsun","star_logg_cgs"
    ]
    return _coerce_numeric(out, num_cols)


def map_nea_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    out = pd.DataFrame()
    # Host identifiers/names
    out["host_id"] = df.get("hostname")
    out["host_name"] = df.get("hostname")
    out["planet_id"] = df.get("pl_name")

    # RA/Dec columns may or may not be present in a pulled table; set to NaN if missing
    out["ra_deg"] = df.get("ra")
    out["dec_deg"] = df.get("dec")

    out["disposition"] = df.get("disposition").map(_normalize_disposition)

    out["period_days"] = df.get("pl_orbper")
    out["radius_rearth"] = df.get("pl_rade")
    out["transit_depth"] = df.get("pl_trandep") if "pl_trandep" in df.columns else np.nan
    out["transit_duration_hr"] = df.get("pl_trandur") if "pl_trandur" in df.columns else np.nan
    out["transit_epoch_bjd"] = df.get("pl_tranmid") if "pl_tranmid" in df.columns else np.nan
    out["insolation_searth"] = df.get("pl_insol") if "pl_insol" in df.columns else np.nan
    out["eq_temp_k"] = df.get("pl_eqt") if "pl_eqt" in df.columns else np.nan

    out["star_teff_k"] = df.get("st_teff")
    out["star_radius_rsun"] = df.get("st_rad")
    out["star_logg_cgs"] = df.get("st_logg")

    out = _add_source(out, "NEA")
    num_cols = [
        "ra_deg","dec_deg","period_days","radius_rearth","transit_depth",
        "transit_duration_hr","transit_epoch_bjd","insolation_searth","eq_temp_k",
        "star_teff_k","star_radius_rsun","star_logg_cgs"
    ]
    return _coerce_numeric(out, num_cols)


def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate across sources. Priority: exact (host_id, planet_id) if both present,
    else fall back to (host_name, planet_id), else (rounded RA/Dec, period).
    """
    df = df.copy()
    # Use lowercase strings for IDs for safer matching
    for col in ["host_id", "host_name", "planet_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # First pass: drop duplicates on host_id+planet_id
    if {"host_id", "planet_id"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["host_id", "planet_id"], keep="first")

    # Second pass: host_name+planet_id
    if {"host_name", "planet_id"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["host_name", "planet_id"], keep="first")

    # Third pass: spatial+period approximate
    if {"ra_deg", "dec_deg", "period_days"}.issubset(df.columns):
        approx = df.copy()
        approx["ra_r1"] = approx["ra_deg"].round(4)
        approx["dec_r1"] = approx["dec_deg"].round(4)
        approx["per_r6"] = approx["period_days"].round(6)
        approx = approx.drop_duplicates(subset=["ra_r1", "dec_r1", "per_r6"], keep="first")
        df = approx.drop(columns=["ra_r1", "dec_r1", "per_r6"], errors="ignore")

    return df


def stratified_split(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    65% train, 20% test, 15% val.
    Try to stratify by 'disposition' if feasible (enough samples in each class).
    """
    df = df.reset_index(drop=True)
    y = df["disposition"].fillna("UNKNOWN")

    # Compute sizes
    test_size = 0.20
    val_size = 0.15
    train_size = 1.0 - test_size - val_size  # 0.65

    # First split: train vs (temp)
    stratify = None
    # Check if every class has at least 2 members for splitting; otherwise fallback to None
    class_counts = y.value_counts()
    if (class_counts >= 2).all() and class_counts.size > 1:
        stratify = y

    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y, test_size=(1.0 - train_size), random_state=seed, stratify=stratify
    )

    # Second split: temp into val and test with proportions 15/(15+20) and 20/(15+20)
    temp_ratio = val_size / (val_size + test_size)  # 0.428571...
    stratify_temp = None
    y_temp2 = y_temp.fillna("UNKNOWN")
    class_counts2 = y_temp2.value_counts()
    if (class_counts2 >= 2).all() and class_counts2.size > 1:
        stratify_temp = y_temp2

    X_val, X_test, _, _ = train_test_split(
        X_temp, y_temp2, test_size=(1 - temp_ratio), random_state=seed, stratify=stratify_temp
    )
    return X_train, X_test, X_val


def main():
    ap = argparse.ArgumentParser(
        description="Merge multiple exoplanet CSVs (TESS/Kepler/NEA) into a standardized dataset and split into train/test/val."
    )
    # Accept 1+ CSVs as positional args (no --tess/--kepler/--nea flags needed)
    ap.add_argument("csv_files", nargs="+", help="Paths to input CSV files")
    ap.add_argument("--outdir", required=True, help="Output directory for PKLs")
    ap.add_argument("--prefix", default="merged", help="Filename prefix for outputs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load & route each CSV to the right mapper by filename
    dataframes = []
    for path in args.csv_files:
        df = pd.read_csv(path)
        fname = os.path.basename(path).lower()

        if "tess" in fname or "toi" in fname:
            mapped = map_tess_columns(df)
        elif "kepler" in fname or "koi" in fname:
            mapped = map_kepler_columns(df)
        elif "nea" in fname or "nasa" in fname or "exoplanet" in fname:
            mapped = map_nea_columns(df)
        else:
            print(f"⚠️  Skipping unrecognized file (can't infer source): {path}")
            continue

        dataframes.append(mapped)

    if not dataframes:
        raise ValueError("No valid CSV files were processed. Rename files to include 'tess', 'toi', 'kepler', 'koi', 'nea', 'nasa', or 'exoplanet' so they can be detected.")

    # Align columns and merge
    std_cols = _standard_cols()
    for df in dataframes:
        for c in std_cols:
            if c not in df.columns:
                df[c] = np.nan
    merged = pd.concat([df[std_cols] for df in dataframes], ignore_index=True)

    # Deduplicate
    merged = dedupe_rows(merged)

    # Save full merged
    all_pkl = os.path.join(args.outdir, f"{args.prefix}_all.pkl")
    merged.to_pickle(all_pkl)

    # Column dictionary (keep your original descriptions)
    col_info = {
        "host_id": "Catalog host identifier (string; TIC/KepID/hostname)",
        "host_name": "Host star name if available",
        "planet_id": "Planet designation",
        "source": "Source catalog (TESS/KEPLER/NEA)",
        "ra_deg": "Right Ascension in degrees",
        "dec_deg": "Declination in degrees",
        "disposition": "CONFIRMED / CANDIDATE / FALSE POSITIVE / other",
        "period_days": "Orbital period (days)",
        "radius_rearth": "Planet radius (Earth radii)",
        "transit_depth": "Transit depth (units as in source; commonly ppm)",
        "transit_duration_hr": "Transit duration (hours)",
        "transit_epoch_bjd": "Reference transit mid-time (BJD)",
        "insolation_searth": "Insolation relative to Earth",
        "eq_temp_k": "Equilibrium temperature (K)",
        "star_teff_k": "Stellar effective temperature (K)",
        "star_radius_rsun": "Stellar radius (solar radii)",
        "star_logg_cgs": "Stellar surface gravity (log10 cm s^-2)",
    }
    with open(os.path.join(args.outdir, f"{args.prefix}_columns.txt"), "w", encoding="utf-8") as f:
        for k, v in col_info.items():
            f.write(f"{k}: {v}\n")

    # Split: 65% / 20% / 15% (train / test / val)
    train_df, test_df, val_df = stratified_split(merged, seed=args.seed)

    # Save splits
    train_df.to_pickle(os.path.join(args.outdir, f"{args.prefix}_train.pkl"))
    test_df.to_pickle(os.path.join(args.outdir, f"{args.prefix}_test.pkl"))
    val_df.to_pickle(os.path.join(args.outdir, f"{args.prefix}_val.pkl"))

    print("Wrote:")
    print(" -", all_pkl)
    print(" -", os.path.join(args.outdir, f"{args.prefix}_train.pkl"))
    print(" -", os.path.join(args.outdir, f"{args.prefix}_test.pkl"))
    print(" -", os.path.join(args.outdir, f"{args.prefix}_val.pkl"))
    print("Also wrote column reference:", os.path.join(args.outdir, f"{args.prefix}_columns.txt"))

if __name__ == "__main__":
    main()
