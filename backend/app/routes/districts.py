from fastapi import APIRouter
from app.services.prediction import get_all_districts

router = APIRouter()


@router.get("/districts")
def districts():
    return get_all_districts()
