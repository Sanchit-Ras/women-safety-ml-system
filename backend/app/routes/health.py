from fastapi import APIRouter
from app.services.prediction import get_model_info

router = APIRouter()


@router.get("/deploy-check")
def deploy_check():
    return {"status": "NEW VERSION LIVE"}


@router.get("/health")
def health():
    return {"status": "API is running 🚀"}


@router.get("/")
def home():
    return {"message": "Women's Safety ML API is running 🚀"}


@router.get("/model-info")
def model_info():
    return get_model_info()
