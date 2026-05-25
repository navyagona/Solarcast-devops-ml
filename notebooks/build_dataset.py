"""
build_dataset.py – Merge all regional ERA5 files into one unified dataset.
SolarCast Region-Based Solar Power Generation Forecasting System

ERA5 Column Reference:
  t2m   → 2-metre temperature (Kelvin)
  d2m   → 2-metre dewpoint temperature (Kelvin)
  sp    → Surface pressure (Pa)
  tp    → Total precipitation (m/hour)
  ssrd  → Surface solar radiation downward (J/m²/hour)
  strd  → Surface thermal radiation downward (J/m²/hour)
  u10   → 10m eastward wind component (m/s)
  v10   → 10m northward wind component (m/s)

Outputs:
  data/unified_solar_dataset.csv   ← Used by train_model.py
"""

import os
import sys
import glob
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("solarcast.build_dataset")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DATA = r"C:\Users\DELL\Downloads\solar-power-generation-forecasting-main\solar-power-generation-forecasting-main\model\data"

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "unified_solar_dataset.csv"
)

REGIONS = {
    "North": os.path.join(BASE_DATA, "north"),
    "South": os.path.join(BASE_DATA, "south"),
    "East":  os.path.join(BASE_DATA, "east"),
    "West":  os.path.join(BASE_DATA, "west"),
}

# ─── ERA5 Column Pattern Mapping ──────────────────────────────────────────────

# Map filename keyword → list of columns that file contains
FILE_COL_MAP = {
    "temperature": ["valid_time", "t2m", "d2m", "latitude", "longitude"],
    "pressure":    ["valid_time", "sp", "tp", "latitude", "longitude"],
    "radiation":   ["valid_time", "ssrd", "strd", "latitude", "longitude"],
    "wind":        ["valid_time", "u10", "v10", "latitude", "longitude"],
}


def identify_file_type(filename: str) -> str:
    """Identify what type of ERA5 file this is based on filename keywords."""
    fname = filename.lower()
    if "temperature" in fname:
        return "temperature"
    elif "pressure" in fname or "precipitation" in fname:
        return "pressure"
    elif "radiation" in fname:
        return "radiation"
    elif "wind" in fname:
        return "wind"
    elif "merged" in fname:
        return "merged"
    return "unknown"


def load_region(region_name: str, region_dir: str) -> pd.DataFrame:
    """
    Load and merge all ERA5 files for one region.
    If a pre-merged file exists, use it directly.
    Otherwise, merge individual variable files on valid_time.
    """
    logger.info(f"Loading region: {region_name} from {region_dir}")
    csv_files = glob.glob(os.path.join(region_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {region_dir}")

    # Check for pre-merged file
    merged_files = [f for f in csv_files if "merged" in os.path.basename(f).lower()]
    if merged_files:
        logger.info(f"  Using pre-merged file: {os.path.basename(merged_files[0])}")
        df = pd.read_csv(merged_files[0])
        df["region"] = region_name
        logger.info(f"  Loaded merged: {df.shape}")
        return df

    # Merge individual files
    individual_files = [f for f in csv_files if "merged" not in os.path.basename(f).lower()]
    dfs = {}
    for fpath in individual_files:
        ftype = identify_file_type(os.path.basename(fpath))
        if ftype == "unknown":
            continue
        sub_df = pd.read_csv(fpath)
        # Normalize valid_time
        sub_df["valid_time"] = pd.to_datetime(sub_df["valid_time"], errors="coerce")
        dfs[ftype] = sub_df
        logger.info(f"  Loaded [{ftype}]: {os.path.basename(fpath)} → {sub_df.shape}")

    if not dfs:
        raise ValueError(f"No recognizable ERA5 files in: {region_dir}")

    # Merge all on valid_time
    merge_cols = ["valid_time", "latitude", "longitude"]
    merged = None
    for ftype, df in dfs.items():
        if merged is None:
            merged = df
        else:
            # Drop lat/lon from right to avoid duplicates
            right = df.drop(columns=[c for c in ["latitude", "longitude"] if c in df.columns], errors="ignore")
            merged = merged.merge(right, on="valid_time", how="outer")

    merged["region"] = region_name
    logger.info(f"  Merged result: {merged.shape}")
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features from ERA5 raw variables.
    """
    # ── Temperature: convert Kelvin → Celsius ────────────────────
    if "t2m" in df.columns:
        df["temperature_c"] = df["t2m"] - 273.15
    if "d2m" in df.columns:
        df["dewpoint_c"] = df["d2m"] - 273.15

    # ── Pressure: Pa → hPa ──────────────────────────────────────
    if "sp" in df.columns:
        df["pressure_hpa"] = df["sp"] / 100.0

    # ── Precipitation: m → mm ────────────────────────────────────
    if "tp" in df.columns:
        df["precipitation_mm"] = df["tp"] * 1000.0

    # ── Solar radiation: J/m²/hr → W/m² ─────────────────────────
    # ssrd is accumulated over the hour, divide by 3600
    if "ssrd" in df.columns:
        df["radiation_wm2"] = df["ssrd"] / 3600.0
        df["radiation_wm2"] = df["radiation_wm2"].clip(lower=0)

    # ── Wind speed from u10/v10 components ───────────────────────
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed_ms"] = np.sqrt(df["u10"]**2 + df["v10"]**2)

    # ── Time features ────────────────────────────────────────────
    if "valid_time" in df.columns:
        dt = pd.to_datetime(df["valid_time"], errors="coerce")
        df["hour"]         = dt.dt.hour
        df["month"]        = dt.dt.month
        df["day_of_year"]  = dt.dt.dayofyear
        df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)

    return df


def create_synthetic_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a realistic synthetic solar power generation target (kW).

    Formula rationale:
      - Solar power is primarily driven by solar radiation (ssrd)
      - Temperature reduces panel efficiency (panels degrade above ~25°C)
      - Rain/clouds reduce output
      - Wind has slight cooling benefit at low speeds, reduces at high
      - Nighttime hours produce zero output

    Formula (physics-inspired):
      base_power   = radiation_wm2 * panel_efficiency * panel_area
      temp_penalty = 1 - 0.004 * max(0, temp_c - 25)  [~0.4%/°C above 25°C]
      rain_penalty = 1 - 2.0 * precipitation_mm (clipped 0-1)
      wind_factor  = 1 + 0.01*wind - 0.001*wind^2 (clipped 0.8-1.1)
      solar_power  = base_power * temp_penalty * rain_penalty * wind_factor
    """
    PANEL_EFFICIENCY = 0.18   # 18% efficiency (typical poly-silicon)
    PANEL_AREA       = 50.0   # 50 m² per installation

    rad   = df.get("radiation_wm2",    pd.Series(np.zeros(len(df)), index=df.index))
    temp  = df.get("temperature_c",    pd.Series(np.full(len(df), 25.0), index=df.index))
    rain  = df.get("precipitation_mm", pd.Series(np.zeros(len(df)), index=df.index))
    wind  = df.get("wind_speed_ms",    pd.Series(np.zeros(len(df)), index=df.index))

    # Base power from radiation
    base = rad * PANEL_EFFICIENCY * PANEL_AREA / 1000.0  # kW

    # Temperature efficiency derating (~0.4%/°C above 25°C)
    temp_factor = 1.0 - 0.004 * np.maximum(0, temp - 25.0)
    temp_factor = np.clip(temp_factor, 0.5, 1.05)

    # Rain/cloud penalty
    rain_factor = 1.0 - np.clip(2.0 * rain, 0, 0.9)

    # Wind factor: slight cooling benefit at low speeds
    wind_factor = 1.0 + 0.01 * wind - 0.001 * wind**2
    wind_factor = np.clip(wind_factor, 0.8, 1.1)

    # Synthetic noise for realism
    np.random.seed(42)
    noise = np.random.normal(1.0, 0.03, size=len(df))

    power = base * temp_factor * rain_factor * wind_factor * noise
    power = np.clip(power, 0, None)   # Non-negative
    df["solar_power_generation"] = np.round(power, 4)

    logger.info(f"  Synthetic target stats: min={power.min():.3f} | "
                f"max={power.max():.3f} | mean={power.mean():.3f} kW")
    return df


def build_unified_dataset() -> pd.DataFrame:
    """
    Main pipeline: load all regions → engineer features → create target → save.
    """
    logger.info("=" * 60)
    logger.info("SolarCast – Building Unified Regional Dataset")
    logger.info("=" * 60)

    all_dfs = []
    for region_name, region_dir in REGIONS.items():
        df = load_region(region_name, region_dir)
        df = engineer_features(df)
        df = create_synthetic_target(df)
        all_dfs.append(df)
        logger.info(f"  {region_name}: {df.shape[0]:,} rows added")

    unified = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Unified dataset shape: {unified.shape}")
    logger.info(f"Regions: {unified['region'].value_counts().to_dict()}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    unified.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved unified dataset: {OUTPUT_PATH}")
    logger.info("=" * 60)

    return unified


if __name__ == "__main__":
    df = build_unified_dataset()
    print("\nSample rows:")
    print(df[["region", "temperature_c", "radiation_wm2", "wind_speed_ms",
              "precipitation_mm", "pressure_hpa", "solar_power_generation"]].head(10).to_string())
