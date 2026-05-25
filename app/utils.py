"""
utils.py – Model loading, preprocessing, and inference utilities.
SolarCast Region-Based Solar Power Generation Forecasting System

Feature pipeline (must match training order):
  Numeric features:
    temperature, pressure, precipitation, radiation, wind_speed,
    hour_sin, hour_cos, month_sin, month_cos
  Encoded feature:
    region_encoded  (LabelEncoder: East=0, North=1, South=2, West=3)
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Any, Optional, Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("solarcast.utils")

# ─── Artifact paths ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(_HERE, "..", "model", "solar_model.pkl")
SCALER_PATH  = os.path.join(_HERE, "..", "model", "scaler.pkl")
ENCODER_PATH = os.path.join(_HERE, "..", "model", "encoder.pkl")

# ─── Feature order (MUST match training) ─────────────────────────────────────
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


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH) -> Optional[Any]:
    try:
        m = joblib.load(path)
        logger.info(f"Model loaded: {path}")
        return m
    except FileNotFoundError:
        logger.warning(f"Model not found: {path}. Run train_regional_model.py first.")
        return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def load_scaler(path: str = SCALER_PATH) -> Optional[Any]:
    try:
        s = joblib.load(path)
        logger.info(f"Scaler loaded: {path}")
        return s
    except FileNotFoundError:
        logger.warning(f"Scaler not found: {path}. Run train_regional_model.py first.")
        return None
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        return None


def load_encoder(path: str = ENCODER_PATH) -> Optional[Any]:
    try:
        e = joblib.load(path)
        logger.info(f"Encoder loaded: {path}")
        return e
    except FileNotFoundError:
        logger.warning(f"Encoder not found: {path}. Run train_regional_model.py first.")
        return None
    except Exception as e:
        logger.error(f"Failed to load encoder: {e}")
        return None


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_request(input_data: Dict[str, Any], encoder: Any) -> pd.DataFrame:
    """
    Convert API request dict → feature DataFrame ready for scaling + inference.

    Steps:
      1. Compute cyclic hour/month encodings
      2. Label-encode region
      3. Return DataFrame with ALL_FEATURES column order
    """
    hour  = int(input_data["hour"])
    month = int(input_data["month"])

    row = {
        "temperature":   float(input_data["temperature"]),
        "pressure":      float(input_data["pressure"]),
        "precipitation": float(input_data["precipitation"]),
        "radiation":     float(input_data["radiation"]),
        "wind_speed":    float(input_data["wind_speed"]),
        "hour_sin":      np.sin(2 * np.pi * hour  / 24),
        "hour_cos":      np.cos(2 * np.pi * hour  / 24),
        "month_sin":     np.sin(2 * np.pi * month / 12),
        "month_cos":     np.cos(2 * np.pi * month / 12),
        "region_encoded": int(encoder.transform([input_data["region"]])[0])
    }

    df = pd.DataFrame([row])[ALL_FEATURES]
    return df


def scale_and_predict(df: pd.DataFrame, scaler: Any, model: Any) -> float:
    """
    Scale the feature DataFrame and run model inference.

    Args:
        df      : DataFrame with ALL_FEATURES columns
        scaler  : Fitted StandardScaler
        model   : Trained RandomForestRegressor

    Returns:
        Predicted solar power in kW (non-negative float)
    """
    X_scaled = scaler.transform(df)
    pred = model.predict(X_scaled)[0]
    return max(0.0, float(pred))


def validate_request(input_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Additional business-logic validation beyond Pydantic constraints.
    """
    radiation = input_data.get("radiation", 0)
    hour      = input_data.get("hour", 12)

    # Nighttime hours should have near-zero radiation
    if hour < 5 or hour > 20:
        if radiation > 100:
            return False, (
                f"Solar radiation {radiation} W/m² seems too high for hour {hour} "
                f"(nighttime). Expected < 100 W/m²."
            )
    return True, "OK"
