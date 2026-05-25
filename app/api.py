"""
api.py – FastAPI backend for SolarCast Region-Based Solar Power Forecasting
"""

import time, logging, os, sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.schemas import RegionalPredictionRequest, RegionalPredictionResponse, HealthResponse
from app.utils import load_model, load_scaler, load_encoder, preprocess_request, scale_and_predict, validate_request

# Ensure project root is always on the path (needed for pytest TestClient context)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from monitoring.prometheus_metrics import (
    record_request, record_prediction, record_prediction_error,
    set_model_loaded, update_system_metrics
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("solarcast.api")

model = scaler = encoder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, encoder
    logger.info("SolarCast Regional API starting up...")
    model, scaler, encoder = load_model(), load_scaler(), load_encoder()
    all_loaded = all(x is not None for x in [model, scaler, encoder])
    set_model_loaded(all_loaded)
    if all_loaded:
        logger.info(f"All artifacts loaded. Regions: {list(encoder.classes_)}")
    else:
        logger.warning("Artifacts not found. Run build_dataset.py then train_regional_model.py")
    yield
    logger.info("SolarCast Regional API shutting down.")

app = FastAPI(
    title="SolarCast Regional API",
    description=(
        "## SolarCast – Region-Based Solar Power Generation Forecasting\n\n"
        "Predicts solar power generation (kW) for **North / South / East / West** regions.\n\n"
        "**Quick Start:**\n"
        "```\npython notebooks/build_dataset.py\n"
        "python notebooks/train_regional_model.py\n"
        "uvicorn app.api:app --reload\n```"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    update_system_metrics()
    record_request(method=request.method, endpoint=request.url.path, status_code=response.status_code, duration=time.time()-start)
    return response

@app.get("/", tags=["General"], summary="Welcome")
async def root():
    return {
        "message": "SolarCast – Region-Based Solar Power Forecasting API",
        "version": "2.0.0",
        "regions": ["North", "South", "East", "West"],
        "docs": "/docs", "health": "/health", "predict": "/predict", "metrics": "/metrics"
    }

@app.get("/health", tags=["General"], response_model=HealthResponse, summary="Health Check")
async def health_check():
    return HealthResponse(
        status="healthy" if all(x is not None for x in [model, scaler, encoder]) else "degraded",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        encoder_loaded=encoder is not None,
        regions=list(encoder.classes_) if encoder else ["North","South","East","West"],
        version="2.0.0"
    )

@app.post(
    "/predict", tags=["Prediction"], response_model=RegionalPredictionResponse,
    summary="Predict Regional Solar Power Generation",
    responses={200: {"description": "Prediction"}, 422: {"description": "Validation error"}, 503: {"description": "Model not loaded"}}
)
async def predict(request: RegionalPredictionRequest):
    """
    Predict solar power generation (kW) for a given region and weather conditions.

    - **region**: North / South / East / West
    - **temperature**: °C  |  **pressure**: hPa  |  **precipitation**: mm
    - **radiation**: W/m²  |  **wind_speed**: m/s
    - **hour**: 0-23  |  **month**: 1-12
    """
    if any(x is None for x in [model, scaler, encoder]):
        raise HTTPException(status_code=503, detail="ML artifacts not loaded. Run training scripts first.")

    input_dict = request.model_dump()
    is_valid, msg = validate_request(input_dict)
    if not is_valid:
        raise HTTPException(status_code=422, detail=msg)

    try:
        t0 = time.time()
        df = preprocess_request(input_dict, encoder)
        result = scale_and_predict(df, scaler, model)
        duration = time.time() - t0
        record_prediction(value=result, duration=duration)
        logger.info(f"[{request.region}] {result:.4f} kW | {duration*1000:.1f}ms")
        return RegionalPredictionResponse(
            region=request.region,
            predicted_solar_power_kw=round(result, 4),
            model_version="2.0.0", status="success",
            message="Prediction completed successfully"
        )
    except Exception as e:
        record_prediction_error()
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics", tags=["Monitoring"], summary="Prometheus Metrics")
async def metrics():
    update_system_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"status": "error", "message": f"'{request.url.path}' not found."})

@app.exception_handler(500)
async def server_error(request: Request, exc):
    return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
