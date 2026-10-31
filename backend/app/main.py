from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import districts, health
from app.services import data


@asynccontextmanager
async def lifespan(_app: FastAPI):
    data.get()                       # load the data at startup: a broken deploy fails here, visibly, not on a request
    yield


app = FastAPI(
    title="Women's Safety Watch List API",
    description="Measured crimes against women for today's districts of India (NCRB, 2001-2022, re-based to today's "
                "district map) and a watch list of districts likely to be among the fastest-growing next year.",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(districts.router)
