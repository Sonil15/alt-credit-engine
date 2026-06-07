import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import consent, frontend, ingestion, scoring
from core.database import init_db
from models.pydantic_schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database tables...")
    await init_db()
    logger.info("Alt-Credit Engine ready.")
    yield
    logger.info("Shutting down Alt-Credit Engine.")


app = FastAPI(
    title="Alt-Credit Engine",
    description="Privacy-preserving alternative credit scoring API",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(consent.router)
app.include_router(ingestion.router)
app.include_router(scoring.router)
app.include_router(frontend.router)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="alt-credit-engine")
