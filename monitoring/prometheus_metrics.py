"""
prometheus_metrics.py – Prometheus monitoring metrics for SolarCast
SolarCast Region-Based Solar Power Generation Forecasting System
"""

from prometheus_client import Counter, Histogram, Gauge
import psutil

# ─── Counters ──────────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "solarcast_request_count_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)
PREDICTION_COUNT = Counter(
    "solarcast_prediction_count_total",
    "Total solar power predictions made"
)
PREDICTION_ERROR_COUNT = Counter(
    "solarcast_prediction_error_count_total",
    "Total failed predictions"
)

# ─── Histograms ────────────────────────────────────────────────────────────────
REQUEST_LATENCY = Histogram(
    "solarcast_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
PREDICTION_LATENCY = Histogram(
    "solarcast_prediction_latency_seconds",
    "Model inference latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
)
PREDICTION_VALUE = Histogram(
    "solarcast_predicted_power_kw",
    "Distribution of predicted solar power values (kW)",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
)

# ─── Gauges ────────────────────────────────────────────────────────────────────
CPU_USAGE    = Gauge("solarcast_cpu_usage_percent",    "Current CPU usage %")
MEMORY_USAGE = Gauge("solarcast_memory_usage_percent", "Current memory usage %")
MODEL_LOADED = Gauge("solarcast_model_loaded",         "1 if ML model is loaded")


# ─── Helper functions ──────────────────────────────────────────────────────────

def update_system_metrics():
    try:
        CPU_USAGE.set(psutil.cpu_percent(interval=None))
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
    except Exception:
        pass


def record_request(method: str, endpoint: str, status_code: int, duration: float):
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


def record_prediction(value: float, duration: float):
    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(duration)
    PREDICTION_VALUE.observe(value)


def record_prediction_error():
    PREDICTION_ERROR_COUNT.inc()


def set_model_loaded(loaded: bool):
    MODEL_LOADED.set(1 if loaded else 0)
