"""
train_regional_model.py – ML training pipeline for region-based solar forecasting.
SolarCast Region-Based Solar Power Generation Forecasting System

Usage:
    python notebooks/train_regional_model.py

Pipeline:
  1. Load unified_solar_dataset.csv
  2. Feature engineering (cyclic time features, wind speed)
  3. Label-encode region
  4. StandardScaler normalization
  5. Train RandomForestRegressor
  6. Evaluate MAE / RMSE / R²
  7. Save model + scaler + encoder
  8. Generate evaluation charts
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─── Path Setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH       = os.path.join(PROJECT_ROOT, "data", "unified_solar_dataset.csv")
MODEL_DIR       = os.path.join(PROJECT_ROOT, "model")
NOTEBOOK_DIR    = os.path.join(PROJECT_ROOT, "notebooks")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "solar_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(NOTEBOOK_DIR, "training_regional.log"), mode="w")
    ]
)
logger = logging.getLogger("solarcast.train_regional")

# ─── Feature Columns ──────────────────────────────────────────────────────────
TARGET_COL = "solar_power_generation"

NUMERIC_FEATURES = [
    "temperature",
    "pressure",
    "precipitation",
    "radiation",
    "wind_speed",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

ALL_FEATURES = NUMERIC_FEATURES + ["region_encoded"]


# ─── Load Data ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found: {DATA_PATH}")
        logger.error("Please run: python notebooks/build_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"Regions: {df['region'].value_counts().to_dict()}")
    return df


# ─── Feature Engineering ──────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, encoder: LabelEncoder = None):
    """
    Build final feature matrix from unified dataset.
    Maps ERA5 engineered columns → model feature names.
    """
    # Rename engineered cols to API-facing names if needed
    col_map = {
        "temperature_c":    "temperature",
        "pressure_hpa":     "pressure",
        "precipitation_mm": "precipitation",
        "radiation_wm2":    "radiation",
        "wind_speed_ms":    "wind_speed",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Build cyclic features if not already present
    if "hour_sin" not in df.columns:
        if "hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        else:
            df["hour_sin"] = 0.0
            df["hour_cos"] = 1.0

    if "month_sin" not in df.columns:
        if "month" in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        else:
            df["month_sin"] = 0.0
            df["month_cos"] = 1.0

    # Label-encode region
    if encoder is None:
        encoder = LabelEncoder()
        df["region_encoded"] = encoder.fit_transform(df["region"])
    else:
        df["region_encoded"] = encoder.transform(df["region"])

    # Fill missing numeric values
    for col in ALL_FEATURES:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' missing from dataset — filling with 0")
            df[col] = 0.0

    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())

    logger.info(f"Feature matrix ready. Shape: {df[ALL_FEATURES].shape}")
    return df, encoder


# ─── Training ─────────────────────────────────────────────────────────────────

def train(X_train, y_train) -> RandomForestRegressor:
    logger.info("Training RandomForestRegressor (50 trees, compact for low-disk)...")
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, region_labels=None):
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    logger.info("=" * 55)
    logger.info("OVERALL MODEL EVALUATION")
    logger.info(f"  MAE  : {mae:.4f} kW")
    logger.info(f"  RMSE : {rmse:.4f} kW")
    logger.info(f"  R2   : {r2:.4f}")
    logger.info("=" * 55)

    # Per-region metrics
    if region_labels is not None:
        logger.info("PER-REGION METRICS:")
        for region in np.unique(region_labels):
            mask = region_labels == region
            if mask.sum() < 10:
                continue
            r_mae  = mean_absolute_error(y_test[mask], y_pred[mask])
            r_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            r_r2   = r2_score(y_test[mask], y_pred[mask])
            logger.info(f"  {region:<6}: MAE={r_mae:.4f}  RMSE={r_rmse:.4f}  R2={r_r2:.4f}")

    return y_pred, mae, rmse, r2


# ─── Visualization ────────────────────────────────────────────────────────────

def save_plots(df_test, y_test, y_pred, model, mae, rmse, r2):
    y_test_arr = np.array(y_test)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SolarCast – Regional Model Evaluation", fontsize=16, fontweight="bold")

    # ── 1. Actual vs Predicted ───────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(y_test_arr, y_pred, alpha=0.4, color="#F4A261", s=15, edgecolors="none")
    lim = max(y_test_arr.max(), y_pred.max())
    ax.plot([0, lim], [0, lim], "r--", lw=2, label="Perfect Fit")
    ax.set_xlabel("Actual (kW)"); ax.set_ylabel("Predicted (kW)")
    ax.set_title("Actual vs Predicted"); ax.legend(); ax.grid(True, alpha=0.3)

    # ── 2. Residuals ─────────────────────────────────────────────
    ax = axes[0, 1]
    res = y_test_arr - y_pred
    ax.scatter(y_pred, res, alpha=0.3, color="#2A9D8F", s=10)
    ax.axhline(0, color="red", lw=2, linestyle="--")
    ax.set_xlabel("Predicted (kW)"); ax.set_ylabel("Residual")
    ax.set_title("Residual Plot"); ax.grid(True, alpha=0.3)

    # ── 3. Feature Importance ────────────────────────────────────
    ax = axes[0, 2]
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    feature_names = ALL_FEATURES
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_names)))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color=colors
    )
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importances"); ax.grid(True, alpha=0.3, axis="x")

    # ── 4. Per-region actual vs predicted boxplot ─────────────────
    ax = axes[1, 0]
    regions_in_test = df_test["region"].values if "region" in df_test.columns else None
    if regions_in_test is not None:
        region_list = sorted(np.unique(regions_in_test))
        data_to_plot = [y_pred[regions_in_test == r] for r in region_list]
        ax.boxplot(data_to_plot, labels=region_list, patch_artist=True,
                   boxprops=dict(facecolor="#E9C46A"))
        ax.set_ylabel("Predicted Power (kW)")
        ax.set_title("Predicted Power Distribution by Region")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.axis("off")

    # ── 5. Prediction timeline ────────────────────────────────────
    ax = axes[1, 1]
    n = min(300, len(y_test_arr))
    ax.plot(y_test_arr[:n], label="Actual", color="#264653", lw=1.2)
    ax.plot(y_pred[:n], label="Predicted", color="#E9C46A", lw=1.2, linestyle="--")
    ax.set_xlabel("Sample"); ax.set_ylabel("kW")
    ax.set_title(f"Timeline (first {n} samples)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── 6. Metrics summary ────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    txt = (
        f"Model Performance Summary\n\n"
        f"{'MAE':<12}: {mae:.4f} kW\n"
        f"{'RMSE':<12}: {rmse:.4f} kW\n"
        f"{'R2 Score':<12}: {r2:.4f}\n\n"
        f"{'Model':<12}: RandomForest\n"
        f"{'Trees':<12}: 200\n"
        f"{'Regions':<12}: N, S, E, W\n"
        f"{'Features':<12}: {len(ALL_FEATURES)}\n"
    )
    ax.text(0.05, 0.5, txt, transform=ax.transAxes, fontsize=11,
            va="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E9F5F9", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(NOTEBOOK_DIR, "regional_evaluation.png")
    plt.savefig(path, dpi=72, bbox_inches="tight")  # low dpi to save disk space
    plt.close()
    logger.info(f"Evaluation chart saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("SolarCast – Regional ML Training Pipeline")
    logger.info("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Prepare features
    encoder = LabelEncoder()
    df, encoder = prepare_features(df, encoder=None)

    # 3. Split
    X = df[ALL_FEATURES].values
    y = df[TARGET_COL].values
    region_labels = df["region"].values

    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
        X, y, region_labels, test_size=0.2, random_state=42, stratify=None
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 5. Train
    model = train(X_train_scaled, y_train)

    # 6. Evaluate
    y_pred, mae, rmse, r2 = evaluate(model, X_test_scaled, y_test, region_labels=r_test)

    # 7. Save charts
    df_test_with_region = pd.DataFrame({"region": r_test})
    save_plots(df_test_with_region, y_test, y_pred, model, mae, rmse, r2)

    # 8. Save artifacts (compress=3 to minimise file size)
    joblib.dump(model,   MODEL_PATH,   compress=3)
    joblib.dump(scaler,  SCALER_PATH,  compress=3)
    joblib.dump(encoder, ENCODER_PATH, compress=3)
    logger.info(f"Model saved   : {MODEL_PATH}")
    logger.info(f"Scaler saved  : {SCALER_PATH}")
    logger.info(f"Encoder saved : {ENCODER_PATH}")

    logger.info("=" * 60)
    logger.info("Training complete! Start the API with:")
    logger.info("  uvicorn app.api:app --reload --port 8000")
    logger.info("=" * 60)

    return model, scaler, encoder, mae, rmse, r2


if __name__ == "__main__":
    main()
