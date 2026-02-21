import joblib
import numpy as np
import pandas as pd
import os
from app.services.cache import redis_client, ENABLE_CACHE
import json
# =====================================================
# MODULE-LEVEL STATE
# =====================================================

model = None
district_df = None
q1 = None
q2 = None
q3 = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================
# LOAD ASSETS ON STARTUP
# =====================================================

def load_assets():
    global model, district_df, q1, q2, q3

    print("Loading model and artifacts...")

    model = joblib.load(os.path.join(BASE_DIR, "artifacts", "womensafety_research_pipeline_xgb.joblib"))
    district_df = pd.read_csv(os.path.join(BASE_DIR, "artifacts", "district_features.csv"))

    quartiles = joblib.load(os.path.join(BASE_DIR, "artifacts", "risk_thresholds.joblib"))
    print(quartiles)
    q1 = quartiles["q1"]
    q2 = quartiles["q2"]
    q3 = quartiles["q3"]

    print("API ready 🚀")


# =====================================================
# RISK CLASSIFIER
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
# PREDICTION LOGIC
# =====================================================

def predict_for_district(state: str, district: str):

    state_norm = state.strip().lower()
    district_norm = district.strip().lower()
    cache_key = f"v1:{state_norm}:{district_norm}"

    # ============================
    # CACHE CHECK
    # ============================

    if ENABLE_CACHE and redis_client:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

    # ============================
    # NORMAL PREDICTION FLOW
    # ============================

    row = district_df[
        (district_df['state_norm'] == state_norm) &
        (district_df['district_norm'] == district_norm)
    ]

    if row.empty:
        return {"error": "District not found"}

    features = row.drop(columns=['state_norm', 'district_norm']).values
    prediction = float(model.predict(features)[0])

    result = {
        "district": district,
        "state": state,
        "predicted_change": round(prediction, 3),
        "risk_level": classify_risk(prediction)
    }

    # ============================
    # STORE IN CACHE
    # ============================

    if ENABLE_CACHE and redis_client:
        redis_client.setex(cache_key, 3600, json.dumps(result))

    return result


def predict_from_features(features_data):
    try:
        data = np.array([[
            features_data.lag_1,
            features_data.lag_2,
            features_data.lag_3,
            features_data.lag_4,
            features_data.lag_5,
            features_data.lag_6,
            features_data.lag_mean,
            features_data.lag_std,
            features_data.lag_last,
            features_data.lag_slope
        ]], dtype=float)

        prediction = float(model.predict(data)[0])

        return {
            "predicted_change": round(prediction, 3),
            "risk_level": classify_risk(prediction)
        }

    except Exception as e:
        return {"error": str(e)}


def get_all_districts():
    data = {}
    for state in district_df['state_norm'].unique():
        districts = district_df[
            district_df['state_norm'] == state
        ]['district_norm'].tolist()
        data[state] = districts
    return data


def get_model_info():
    return {
        "model": "Random Forest Regressor",
        "features": 10,
        "task": "Crime Percentage Change Prediction",
        "dataset_years": "2001–2014",
        "prediction_target": "2014–2015",
        "risk_thresholds": {
            "low": f"< {round(q1, 2)}",
            "moderate": f"{round(q1, 2)} – {round(q2, 2)}",
            "high": f"{round(q2, 2)} – {round(q3, 2)}",
            "very_high": f"> {round(q3, 2)}"
        }
    }
