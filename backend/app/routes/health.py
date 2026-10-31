from fastapi import APIRouter

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
    return {"api_version": "2.0", "data_generated": data.meta()["generated"], "watch_year": data.meta()["watch_year"]}


@router.get("/model-info")
def model_info():
    """The model card: what the watch list is, the evidence for it, and what the app does not claim."""
    return data.meta()
