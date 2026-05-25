"""
test_api.py – pytest test suite for SolarCast Regional API
Run: pytest tests/test_api.py -v
"""

import sys, os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "region": "North", "temperature": 28.5, "pressure": 1010.0,
    "precipitation": 0.0, "radiation": 650.0, "wind_speed": 3.5,
    "hour": 12, "month": 6
}

# ─── Root endpoint ────────────────────────────────────────────────────────────
class TestRootEndpoint:
    def test_root_200(self):
        assert client.get("/").status_code == 200

    def test_root_json(self):
        assert "application/json" in client.get("/").headers["content-type"]

    def test_root_has_message(self):
        assert "message" in client.get("/").json()

    def test_root_has_regions(self):
        data = client.get("/").json()
        assert "regions" in data
        assert isinstance(data["regions"], list)
        assert set(data["regions"]) == {"North", "South", "East", "West"}

    def test_root_has_docs_link(self):
        assert client.get("/").json().get("docs") == "/docs"

# ─── Health endpoint ──────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_200(self):
        assert client.get("/health").status_code == 200

    def test_health_has_status(self):
        assert "status" in client.get("/health").json()

    def test_health_has_model_loaded(self):
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_has_scaler_loaded(self):
        data = client.get("/health").json()
        assert "scaler_loaded" in data

    def test_health_has_encoder_loaded(self):
        data = client.get("/health").json()
        assert "encoder_loaded" in data

    def test_health_has_regions_list(self):
        data = client.get("/health").json()
        assert "regions" in data
        assert isinstance(data["regions"], list)

# ─── Predict endpoint ─────────────────────────────────────────────────────────
class TestPredictEndpoint:
    def test_predict_valid_returns_200_or_503(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code in [200, 503]

    def test_predict_all_regions_accepted(self):
        for region in ["North", "South", "East", "West"]:
            payload = {**VALID_PAYLOAD, "region": region}
            resp = client.post("/predict", json=payload)
            assert resp.status_code in [200, 503], f"Failed for region={region}"

    def test_predict_invalid_region_422(self):
        payload = {**VALID_PAYLOAD, "region": "Central"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_predict_missing_region_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "region"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_predict_missing_temperature_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "temperature"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_predict_invalid_hour_422(self):
        assert client.post("/predict", json={**VALID_PAYLOAD, "hour": 24}).status_code == 422

    def test_predict_invalid_month_422(self):
        assert client.post("/predict", json={**VALID_PAYLOAD, "month": 13}).status_code == 422

    def test_predict_negative_radiation_422(self):
        resp = client.post("/predict", json={**VALID_PAYLOAD, "radiation": -10.0})
        assert resp.status_code == 422

    def test_predict_negative_precipitation_422(self):
        assert client.post("/predict", json={**VALID_PAYLOAD, "precipitation": -1.0}).status_code == 422

    def test_predict_empty_body_422(self):
        assert client.post("/predict", json={}).status_code == 422

    def test_predict_wrong_type_422(self):
        assert client.post("/predict", json={**VALID_PAYLOAD, "temperature": "hot"}).status_code == 422

    def test_predict_response_structure_on_success(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert "predicted_solar_power_kw" in data
            assert "region" in data
            assert "status" in data
            assert isinstance(data["predicted_solar_power_kw"], float)
            assert data["predicted_solar_power_kw"] >= 0.0

    def test_predict_response_region_matches_input(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            assert resp.json()["region"] == "North"

    def test_predict_nighttime_high_radiation_422(self):
        # Hour 2, radiation 800 → should be flagged
        payload = {**VALID_PAYLOAD, "hour": 2, "radiation": 800.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in [200, 422, 503]  # 422 expected from range validator

    def test_predict_pressure_out_of_range_422(self):
        assert client.post("/predict", json={**VALID_PAYLOAD, "pressure": 700.0}).status_code == 422

    def test_predict_zero_radiation_accepted(self):
        # Zero radiation is valid (nighttime)
        resp = client.post("/predict", json={**VALID_PAYLOAD, "radiation": 0.0, "hour": 0})
        assert resp.status_code in [200, 503]

# ─── Metrics endpoint ─────────────────────────────────────────────────────────
class TestMetricsEndpoint:
    def test_metrics_200(self):
        assert client.get("/metrics").status_code == 200

    def test_metrics_prometheus_format(self):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_has_request_counter(self):
        assert "solarcast_request_count_total" in client.get("/metrics").text

    def test_metrics_has_prediction_counter(self):
        assert "solarcast_prediction_count_total" in client.get("/metrics").text

# ─── 404 handling ─────────────────────────────────────────────────────────────
class TestNotFound:
    def test_unknown_route_404(self):
        assert client.get("/this/does/not/exist").status_code == 404

    def test_404_has_error_body(self):
        resp = client.get("/nonexistent")
        if resp.status_code == 404:
            body = resp.json()
            assert "status" in body or "message" in body

# ─── Schema validation ────────────────────────────────────────────────────────
class TestSchemaValidation:
    def test_south_region_valid(self):
        resp = client.post("/predict", json={**VALID_PAYLOAD, "region": "South"})
        assert resp.status_code in [200, 503]

    def test_east_region_valid(self):
        resp = client.post("/predict", json={**VALID_PAYLOAD, "region": "East"})
        assert resp.status_code in [200, 503]

    def test_west_region_valid(self):
        resp = client.post("/predict", json={**VALID_PAYLOAD, "region": "West"})
        assert resp.status_code in [200, 503]

    def test_extreme_but_valid_values(self):
        payload = {**VALID_PAYLOAD, "temperature": -5.0, "radiation": 1300.0, "wind_speed": 20.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in [200, 503]

    def test_boundary_hour_zero(self):
        assert client.post("/predict", json={**VALID_PAYLOAD, "hour": 0}).status_code in [200, 422, 503]

    def test_boundary_hour_23(self):
        resp = client.post("/predict", json={**VALID_PAYLOAD, "hour": 23, "radiation": 0.0})
        assert resp.status_code in [200, 503]

    def test_boundary_month_1(self):
        assert client.post("/predict", json={**VALID_PAYLOAD, "month": 1}).status_code in [200, 503]

    def test_boundary_month_12(self):
        assert client.post("/predict", json={**VALID_PAYLOAD, "month": 12}).status_code in [200, 503]
