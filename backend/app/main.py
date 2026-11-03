from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import API_VERSION
from app.routes import districts, health
from app.services import data


@asynccontextmanager
async def lifespan(_app: FastAPI):
    data.get()                       # load the data at startup: a broken deploy fails here, visibly, not on a request
    yield


app = FastAPI(
    title="Crimes Against Women in India, by District: API",
    description="Reported crimes against women for every district on today's map of India, 2001-2022 (NCRB, re-based "
                "onto today's districts so every year is comparable): each district's yearly numbers, how it changed "
                "over the 22 years and since 2017, each beside its state and India, and an experimental watch list for "
                "the year after the data. "
                "The whole table is also a CSV download on the website.",
    version=API_VERSION,
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
