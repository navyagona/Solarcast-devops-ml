"""
schemas.py – Pydantic v2 request/response models for SolarCast Region-Based API
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional


VALID_REGIONS = ["North", "South", "East", "West"]


class RegionalPredictionRequest(BaseModel):
    """
    Input schema for region-based solar power prediction.

    All ERA5 features are provided as pre-processed engineering values:
      - temperature : °C  (ERA5 t2m converted from Kelvin)
      - pressure    : hPa (ERA5 sp / 100)
      - precipitation: mm (ERA5 tp * 1000)
      - radiation   : W/m² (ERA5 ssrd / 3600)
      - wind_speed  : m/s (√(u10² + v10²))
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "region": "North",
                "temperature": 28.5,
                "pressure": 1010.2,
                "precipitation": 0.0,
                "radiation": 650.0,
                "wind_speed": 3.5,
                "hour": 12,
                "month": 6
            }
        }
    )

    region:        Literal["North", "South", "East", "West"] = Field(
        ..., description="Region identifier (North / South / East / West)"
    )
    temperature:   float = Field(..., description="2m air temperature in Celsius")
    pressure:      float = Field(..., description="Surface pressure in hPa", ge=800, le=1100)
    precipitation: float = Field(..., description="Total precipitation in mm", ge=0)
    radiation:     float = Field(..., description="Solar radiation in W/m²", ge=0, le=1400)
    wind_speed:    float = Field(..., description="Wind speed in m/s", ge=0, le=100)
    hour:          int   = Field(..., description="Hour of day (0-23)", ge=0, le=23)
    month:         int   = Field(..., description="Month (1-12)", ge=1, le=12)


class RegionalPredictionResponse(BaseModel):
    """
    Response schema for solar power prediction.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "region": "North",
                "predicted_solar_power_kw": 5.234,
                "model_version": "2.0.0",
                "status": "success",
                "message": "Prediction completed successfully"
            }
        }
    )

    region:                   str   = Field(..., description="Input region")
    predicted_solar_power_kw: float = Field(..., description="Predicted solar power generation in kW")
    model_version:            str   = Field(default="2.0.0", description="Model version")
    status:                   str   = Field(default="success")
    message:                  str   = Field(default="Prediction completed successfully")


class HealthResponse(BaseModel):
    status:        str  = Field(default="healthy")
    model_loaded:  bool = Field(...)
    scaler_loaded: bool = Field(...)
    encoder_loaded: bool = Field(...)
    regions:       list = Field(default=VALID_REGIONS)
    version:       str  = Field(default="2.0.0")


class ErrorResponse(BaseModel):
    status:  str           = Field(default="error")
    message: str           = Field(...)
    detail:  Optional[str] = Field(None)
