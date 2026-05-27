# ☀️ SolarCast v2.0 – Region-Based Solar Power Generation Forecasting System

> **Production-ready ML + DevOps project** — Predicts solar power generation for North / South / East / West regions using ERA5 reanalysis weather data, RandomForestRegressor, FastAPI, Prometheus, Grafana, and Docker.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset & Regions](#dataset--regions)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Dashboard](#dashboard)
- [Monitoring](#monitoring)
- [Docker Usage](#docker-usage)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [Grafana Integration](#grafana-integration)
- [Running Tests](#running-tests)
- [CI/CD Pipeline](#cicd-pipeline)
- [Model Details](#model-details)

---

## 🎯 Overview

**SolarCast v2.0** combines 4 regional ERA5 climate datasets into a unified ML pipeline that predicts hourly solar power generation from real weather variables.

| Component | Technology |
|-----------|------------|
| Data | ERA5 Land Reanalysis (ECMWF) |
| ML Model | RandomForestRegressor (scikit-learn) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Monitoring | Prometheus + Grafana |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## 🌍 Dataset & Regions

ERA5 hourly data for 2023 across 4 Indian climate zones:

| Region | Coordinates | Climate |
|--------|-------------|---------|
| North | 28°N, 73.3°E | Semi-arid (Rajasthan) |
| South | 13°N, 77.6°E | Tropical (Bangalore) |
| East | 20.3°N, 85.8°E | Humid subtropical (Odisha) |
| West | 19.6°N, 75.5°E | Semi-arid (Maharashtra) |

**ERA5 Variables Used:**

| Variable | Symbol | Unit | Used As |
|----------|--------|------|---------|
| 2m Temperature | t2m | K → °C | temperature |
| Surface Pressure | sp | Pa → hPa | pressure |
| Total Precipitation | tp | m → mm | precipitation |
| Surface Solar Radiation | ssrd | J/m² → W/m² | radiation |
| 10m Wind (U+V) | u10, v10 | m/s | wind_speed |

**Synthetic Target Formula:**
```
base_power = radiation × 0.18 × 50 m² / 1000        (kW)
temp_factor = 1 - 0.004 × max(0, temp_°C - 25)
rain_factor = 1 - min(0.9, 2 × precip_mm)
wind_factor = clip(1 + 0.01×wind - 0.001×wind², 0.8, 1.1)
solar_power = base × temp_factor × rain_factor × wind_factor
```

---

## 🗂️ Project Structure

```
solar power predictions devops ml/
│
├── app/
│   ├── __init__.py
│   ├── api.py                  ← FastAPI (GET /, /health, POST /predict, GET /metrics)
│   ├── schemas.py              ← Pydantic v2 models with Literal region validation
│   └── utils.py                ← Loaders, preprocessing, inference pipeline
│
├── model/
│   ├── solar_model.pkl         ← Trained RandomForest (generated)
│   ├── scaler.pkl              ← StandardScaler (generated)
│   └── encoder.pkl             ← LabelEncoder for regions (generated)
│
├── notebooks/
│   ├── build_dataset.py        ← Merges 4 regional ERA5 datasets → unified CSV
│   ├── train_regional_model.py ← Full ML training pipeline
│   └── training_regional.log   ← Training logs (generated)
│
├── dashboard/
│   └── dashboard.py            ← 5-tab Streamlit + Plotly dashboard
│
├── monitoring/
│   ├── __init__.py
│   └── prometheus_metrics.py   ← Counters, Histograms, Gauges
│
├── data/
│   └── unified_solar_dataset.csv  ← 70,272 hourly rows (generated)
│
├── tests/
│   ├── __init__.py
│   └── test_api.py             ← 41 pytest tests across 7 classes
│
├── .github/workflows/
│   └── ci_cd.yml               ← GitHub Actions CI/CD
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- ERA5 datasets in their respective folders (already at the configured paths)

### Step 1: Install dependencies
```bash
cd "solar power predictions devops ml"
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Step 2: Build unified dataset
```bash
python notebooks/build_dataset.py
```
Creates `data/unified_solar_dataset.csv` (70,272 rows × 26 columns).

### Step 3: Train the model
```bash
python notebooks/train_regional_model.py
```
Saves `model/solar_model.pkl`, `model/scaler.pkl`, `model/encoder.pkl`.

Expected results:
- **MAE**: ~0.065 kW | **RMSE**: ~0.132 kW | **R²**: ~0.997

### Step 4: Start the API
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```
Swagger UI: **http://localhost:8000/docs**

### Step 5: Launch dashboard
```bash
streamlit run dashboard/dashboard.py
```
Dashboard: **http://localhost:8501**

---

## 🌐 API Usage

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "region": "North",
    "temperature": 28.5,
    "pressure": 1010.2,
    "precipitation": 0.0,
    "radiation": 650.0,
    "wind_speed": 3.5,
    "hour": 12,
    "month": 6
  }'
```

**Response:**
```json
{
  "region": "North",
  "predicted_solar_power_kw": 5.2341,
  "model_version": "2.0.0",
  "status": "success",
  "message": "Prediction completed successfully"
}
```

**Python:**
```python
import requests

payload = {
    "region": "South",
    "temperature": 32.0, "pressure": 1005.0,
    "precipitation": 0.0, "radiation": 820.0,
    "wind_speed": 4.2, "hour": 11, "month": 4
}
r = requests.post("http://localhost:8000/predict", json=payload)
print(r.json())
```

### All Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health + artifact status |
| POST | `/predict` | Solar power prediction |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

---

## 📊 Dashboard

5-tab interactive Streamlit + Plotly dashboard:

| Tab | Contents |
|-----|----------|
| 📊 Region Overview | Per-region KPIs, actual vs predicted scatter, performance table |
| 🔮 Live Prediction | Input sliders → API/local prediction + power gauge |
| 📈 Feature Importance | Bar chart + sunburst + importance table |
| 🌤️ Radiation Trends | Hourly/monthly radiation profiles, radiation vs power scatter |
| 🌦️ Weather Impact | Temperature/pressure/precipitation/wind vs power with LOWESS trendlines + correlation heatmap |

---

## 📡 Monitoring

| Metric | Type | Description |
|--------|------|-------------|
| `solarcast_request_count_total` | Counter | HTTP requests by method/endpoint/status |
| `solarcast_request_latency_seconds` | Histogram | Request latency |
| `solarcast_prediction_count_total` | Counter | Successful predictions |
| `solarcast_prediction_latency_seconds` | Histogram | Inference time |
| `solarcast_prediction_error_count_total` | Counter | Failed predictions |
| `solarcast_predicted_power_kw` | Histogram | Power distribution |
| `solarcast_cpu_usage_percent` | Gauge | CPU % |
| `solarcast_memory_usage_percent` | Gauge | Memory % |
| `solarcast_model_loaded` | Gauge | Model availability |

```bash
curl http://localhost:8000/metrics
```

---

## 🐳 Docker Usage

### Single container
```bash
docker build -t solarcast-api:2.0.0 .
docker run -d -p 8000:8000 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/data:/app/data \
  solarcast-api:2.0.0
```

### Full stack
```bash
docker-compose up --build

# Background
docker-compose up -d --build

# Logs
docker-compose logs -f solarcast-api

# Stop
docker-compose down
```

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## ☁️ AWS EC2 Deployment

### Step 1: Launch EC2
- AMI: Ubuntu Server 22.04 LTS
- Instance: t2.medium (2 vCPU, 4GB RAM recommended)
- Security Group — open ports: `22`, `8000`, `8501`, `9090`, `3000`

### Step 2: Connect
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
```

### Step 3: Install Docker
```bash
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose-plugin git
sudo systemctl enable --now docker
sudo usermod -aG docker $USER && newgrp docker
```

### Step 4: Deploy
```bash
git clone https://github.com/YOUR_USERNAME/solarcast.git
cd solarcast

# Upload ERA5 data and run dataset builder
# scp -i key.pem -r data/ ubuntu@EC2_IP:~/solarcast/data/

pip3 install -r requirements.txt
python3 notebooks/build_dataset.py
python3 notebooks/train_regional_model.py

docker compose up --build -d
docker compose ps
```

### Step 5: Access
```
http://YOUR_EC2_IP:8000/docs   ← Swagger UI
http://YOUR_EC2_IP:8501         ← Dashboard
http://YOUR_EC2_IP:9090         ← Prometheus
http://YOUR_EC2_IP:3000         ← Grafana
```

---

## ☁️ AWS ECS Deployment with Terraform

This repository now includes a Terraform-based AWS ECS deployment in `infra/terraform` and a GitHub Actions workflow in `.github/workflows/aws_deploy.yml`.

### Prerequisites
- AWS account with a default VPC
- GitHub secrets configured: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ACCOUNT_ID`
- AWS region configured in `infra/terraform/terraform.tfvars.example`

### GitHub deployment
Push to `main` to trigger `.github/workflows/aws_deploy.yml`, which:
1. creates the ECR repository
2. builds and pushes the Docker image
3. deploys the API to ECS Fargate behind an ALB

### Manual Terraform deploy
```bash
cd infra/terraform
terraform init
terraform apply -auto-approve \
  -var="aws_region=us-east-1" \
  -var="project_name=solarcast-api" \
  -var="image_tag=latest"
```

After deployment, retrieve the public endpoint with:
```bash
terraform output alb_dns_name
```

---

## 📈 Grafana Integration

### 1. Add Prometheus data source
- URL: `http://prometheus:9090` (Docker) or `http://localhost:9090` (local)

### 2. Useful PromQL queries

**Request rate:**
```promql
rate(solarcast_request_count_total[5m])
```

**Average prediction latency (ms):**
```promql
1000 * rate(solarcast_prediction_latency_seconds_sum[5m])
       / rate(solarcast_prediction_latency_seconds_count[5m])
```

**Total predictions:**
```promql
solarcast_prediction_count_total
```

**Average predicted power (kW):**
```promql
rate(solarcast_predicted_power_kw_sum[5m])
/ rate(solarcast_predicted_power_kw_count[5m])
```

**Error rate:**
```promql
rate(solarcast_prediction_error_count_total[5m])
```

---

## 🧪 Running Tests

```bash
# All 41 tests
pytest tests/test_api.py -v

# Short output
pytest tests/test_api.py -v --tb=short

# Specific class
pytest tests/test_api.py::TestPredictEndpoint -v

# With coverage
pytest tests/test_api.py --cov=app --cov-report=term-missing
```

---

## 🔄 CI/CD Pipeline

| Job | Trigger | Steps |
|-----|---------|-------|
| Test | Every push | Install → create dummy artifacts → pytest 41 tests |
| Build | After tests | Build Docker → smoke test container |
| Publish | Push to main | Build + push to Docker Hub |

**Required GitHub Secrets (for publish):**
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

---

## 🤖 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | RandomForestRegressor |
| n_estimators | 50 (compressed) |
| max_depth | 12 |
| Training samples | 56,217 |
| Test samples | 14,055 |
| Features | 10 (9 numeric + 1 encoded) |
| MAE | ~0.065 kW |
| RMSE | ~0.132 kW |
| R² | ~0.997 |

**Features:**

| Feature | Description |
|---------|-------------|
| temperature | 2m air temperature (°C) |
| pressure | Surface pressure (hPa) |
| precipitation | Total precipitation (mm) |
| radiation | Solar radiation downward (W/m²) |
| wind_speed | Wind speed √(u²+v²) (m/s) |
| hour_sin / hour_cos | Cyclic hour encoding |
| month_sin / month_cos | Cyclic month encoding |
| region_encoded | LabelEncoded region (0–3) |

---

<div align="center">
  <strong>☀️ SolarCast v2.0</strong><br>
  Region-Based Solar Power Generation Forecasting<br>
  ERA5 · RandomForest · FastAPI · Streamlit · Prometheus · Docker
</div>
