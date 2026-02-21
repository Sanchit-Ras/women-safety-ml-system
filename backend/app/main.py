from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict, districts, health
from app.services.prediction import load_assets

# =====================================================
# APP CREATION
# =====================================================

app = FastAPI(
    title="Women's Safety Predictor API",
    description="Predicts district-level crime change using a trained Random Forest model.",
    version="1.0"
)

# =====================================================
# CORS MIDDLEWARE
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://women-safety-ml-system-jwlh.vercel.app/"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# INCLUDE ROUTERS
# =====================================================

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(districts.router)

# =====================================================
# STARTUP EVENT
# =====================================================

@app.on_event("startup")
def startup():
    load_assets()
