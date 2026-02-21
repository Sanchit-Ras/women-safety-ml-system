from fastapi import APIRouter
from app.schemas import CrimeFeatures
from app.services.prediction import predict_for_district, predict_from_features

router = APIRouter()


@router.get("/predict_district")
def predict_district(state: str, district: str):
    return predict_for_district(state, district)


@router.post("/predict")
def predict(features: CrimeFeatures):
    return predict_from_features(features)
