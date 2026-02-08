from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd

# =====================================================
# FAST STARTUP LOADING
# =====================================================

app = FastAPI(
    title="Women's Safety Predictor API",
    description="Predicts district-level crime change using a trained Random Forest model.",
    version="1.0"
)

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # VERY IMPORTANT
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/deploy-check")
def deploy_check():
    return {"status": "NEW VERSION LIVE"}

# =====================================================
# LOAD EVERYTHING ON STARTUP (VERY IMPORTANT)
# =====================================================

@app.on_event("startup")
def load_assets():

    global model, district_df, q1, q2, q3

    print("Loading model and artifacts...")

    model = joblib.load("models/womensafety_research_pipeline_xgb.joblib")

    district_df = pd.read_csv("district_features.csv")

    quartiles = joblib.load("models/risk_thresholds.joblib")
    print(quartiles)
    q1 = quartiles["q1"]
    q2 = quartiles["q2"]
    q3 = quartiles["q3"]
    print("API ready 🚀")


# =====================================================
# RISK CLASSIFIER (Aligned with Map)
# =====================================================

def classify_risk(val):

    if val < q1:
        return "Low Risk"
    elif val < q2:
        return "Moderate Risk"
    elif val < q3:
        return "High Risk"
    else:
        return "Very High Risk"


# =====================================================
# DISTRICT PREDICTION
# =====================================================

@app.get("/predict_district")
def predict_district(state: str, district: str):

    row = district_df[
        (district_df['state_norm'] == state.lower()) &
        (district_df['district_norm'] == district.lower())
    ]

    if row.empty:
        return {"error": "District not found"}

    features = row.drop(
        columns=['state_norm','district_norm']
    ).values

    prediction = float(model.predict(features)[0])

    return {
        "district": district,
        "state": state,
        "predicted_change": round(prediction, 3),
        "risk_level": classify_risk(prediction)
    }


# =====================================================
# DISTRICT LIST
# =====================================================

@app.get("/districts")
def get_districts():

    data = {}

    for state in district_df['state_norm'].unique():

        districts = district_df[
            district_df['state_norm'] == state
        ]['district_norm'].tolist()

        data[state] = districts

    return data


# =====================================================
# REQUEST SCHEMA
# =====================================================

class CrimeFeatures(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_5: float
    lag_6: float
    lag_mean: float
    lag_std: float
    lag_last: float
    lag_slope: float


# =====================================================
# GENERIC PREDICTION
# =====================================================

@app.post("/predict")
def predict(features: CrimeFeatures):

    try:

        data = np.array([[
            features.lag_1,
            features.lag_2,
            features.lag_3,
            features.lag_4,
            features.lag_5,
            features.lag_6,
            features.lag_mean,
            features.lag_std,
            features.lag_last,
            features.lag_slope
        ]], dtype=float)

        prediction = float(model.predict(data)[0])

        return {
            "predicted_change": round(prediction, 3),
            "risk_level": classify_risk(prediction)
        }

    except Exception as e:
        return {"error": str(e)}


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():
    return {"status": "API is running 🚀"}


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():
    return {"message": "Women's Safety ML API is running 🚀"}


# =====================================================
# MODEL INFO (Recruiters LOVE this)
# =====================================================

@app.get("/model-info")
def model_info():
    return {
        "model": "Random Forest Regressor",
        "features": 10,
        "task": "Crime Percentage Change Prediction",
        "dataset_years": "2001–2014",
        "prediction_target": "2014–2015",
        "risk_thresholds": {
            "low": f"< {round(q1,2)}",
            "moderate": f"{round(q1,2)} – {round(q2,2)}",
            "high": f"{round(q2,2)} – {round(q3,2)}",
            "very_high": f"> {round(q3,2)}"
        }
    }
