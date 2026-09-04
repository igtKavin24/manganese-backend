"""
FastAPI backend for SIH26009 — Manganese Reserve & Production Shortfall Prediction

Endpoints:
  GET  /                     -> health check
  POST /predict_reserve      -> reserve probability for a lat/lon (nearest-neighbor lookup on precomputed grid)
  POST /predict_shortfall    -> production shortfall prediction for given mine-day conditions

Expected files in the same directory as this script:
  - reserve_cache.csv          (columns: lat, lon, probability)
  - production_model.pkl       (trained RandomForestRegressor)
  - prod_feature_columns.pkl   (list of feature column names, in training order)
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Manganese Reserve & Shortfall API", version="1.0")

# Allow the frontend (any origin) to call this API. Tighten allow_origins
# to your actual frontend domain once it's live, for better security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------

try:
    reserve_cache = pd.read_csv(os.path.join(BASE_DIR, "reserve_cache.csv"))
except FileNotFoundError:
    reserve_cache = None
    print("WARNING: reserve_cache.csv not found — /predict_reserve will fail until it's added.")

try:
    production_model = joblib.load(os.path.join(BASE_DIR, "production_model.pkl"))
    prod_feature_cols = joblib.load(os.path.join(BASE_DIR, "prod_feature_columns.pkl"))
except FileNotFoundError:
    production_model = None
    prod_feature_cols = None
    print("WARNING: production_model.pkl or prod_feature_columns.pkl not found — /predict_shortfall will fail until they're added.")


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ReserveRequest(BaseModel):
    lat: float = Field(..., description="Latitude, e.g. 21.81")
    lon: float = Field(..., description="Longitude, e.g. 80.23")


class ShortfallRequest(BaseModel):
    equipment_availability: float = Field(..., ge=0, le=1)
    equipment_downtime: float = Field(..., ge=0)
    maintenance_hours: float = Field(..., ge=0)
    drilling_delay: float = Field(..., ge=0)
    blast_delay: float = Field(..., ge=0)
    rainfall: float = Field(..., ge=0)
    soil_moisture: float = Field(..., ge=0, le=1)
    temperature: float
    truck_count: int = Field(..., ge=0)
    haulage_delay: float = Field(..., ge=0)
    previous_day_production: float = Field(..., ge=0)
    previous_7day_average: float = Field(..., ge=0)
    target_production: float = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "reserve_cache_loaded": reserve_cache is not None,
        "production_model_loaded": production_model is not None,
    }


@app.post("/predict_reserve")
def predict_reserve(req: ReserveRequest):
    if reserve_cache is None:
        raise HTTPException(status_code=503, detail="Reserve cache not loaded on server.")

    # Nearest-neighbor lookup against the precomputed 0.25-degree grid
    diffs = (reserve_cache["lat"] - req.lat) ** 2 + (reserve_cache["lon"] - req.lon) ** 2
    nearest_idx = diffs.idxmin()
    nearest = reserve_cache.loc[nearest_idx]

    distance_deg = float(np.sqrt(diffs.loc[nearest_idx]))

    return {
        "query_lat": req.lat,
        "query_lon": req.lon,
        "nearest_grid_lat": float(nearest["lat"]),
        "nearest_grid_lon": float(nearest["lon"]),
        "probability": float(nearest["probability"]),
        "grid_distance_degrees": round(distance_deg, 4),
        "note": "Result is looked up from a precomputed 0.25-degree grid over central India (MOIL's operating region), not a live satellite call.",
    }


@app.post("/predict_shortfall")
def predict_shortfall(req: ShortfallRequest):
    if production_model is None or prod_feature_cols is None:
        raise HTTPException(status_code=503, detail="Production model not loaded on server.")

    row = pd.DataFrame([req.dict()])[prod_feature_cols]
    predicted_actual = float(production_model.predict(row)[0])
    predicted_actual = max(0.0, predicted_actual)

    shortfall_pct = round((1 - predicted_actual / req.target_production) * 100, 2)

    # Rule-based recommendation engine
    recommendations = []
    if req.equipment_availability < 0.75:
        recommendations.append("Equipment availability is low — schedule preventive maintenance or bring in backup machinery.")
    if req.rainfall > 20:
        recommendations.append("High rainfall detected — consider adjusting shift schedules or reinforcing drainage at pit access roads.")
    if (req.drilling_delay + req.blast_delay) > 3:
        recommendations.append("Drilling/blasting delays are significant — review crew scheduling and explosive supply chain.")
    if req.truck_count < 10:
        recommendations.append("Truck count is low — consider reallocating haulage vehicles from lower-priority sites.")
    if shortfall_pct > 15:
        recommendations.append("Projected shortfall exceeds 15% — escalate to site supervisor for contingency planning.")

    return {
        "predicted_production": round(predicted_actual, 2),
        "target_production": req.target_production,
        "shortfall_pct": shortfall_pct,
        "risk_flags": len(recommendations),
        "recommendations": recommendations or ["No significant risk factors detected — production on track."],
    }
