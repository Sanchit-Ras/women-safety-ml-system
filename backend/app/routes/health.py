from fastapi import APIRouter

from app import API_VERSION
from app.services import data

router = APIRouter()


@router.get("/")
def home():
    return {"message": "Women's Safety ML API is running. See /docs."}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/deploy-check")
def deploy_check():
    """Which version and which data are live: open it after a deploy."""
    return {"api_version": API_VERSION, "data_generated": data.meta()["generated"], "watch_year": data.meta()["watch_year"]}


@router.get("/model-info")
def model_info():
    """The app's settings and evidence: the watch list's model card (what it is, the evidence for it, what the app does
    not claim), the trend comparison's settings with India's and each state's change, and the downloadable table's shape."""
    return data.meta()
