import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import assessment, auth, consent, frontend, ingestion, intake, letters, scoring
from core.bootstrap import ensure_seeded
from core.config import get_settings
from core.database import init_db
from core.model_cache import get_model_version, init_model_cache
from models.pydantic_schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database tables...")
    await init_db()
    if get_settings().seed_on_startup_enabled:
        try:
            await ensure_seeded()
        except Exception:
            logger.exception("Startup seeding failed (continuing without seed data)")
    init_model_cache()
    logger.info("Alt-Credit Engine ready (model=%s).", get_model_version())
    yield
    logger.info("Shut down Alt-Credit Engine.")


app = FastAPI(
    title="Alt-Credit Engine",
    description="Privacy-preserving alternative credit scoring API for thin-file borrowers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend.mount_static(app)

app.include_router(auth.router)
app.include_router(assessment.router)
app.include_router(consent.router)
app.include_router(consent.geo_router)
app.include_router(ingestion.router)
app.include_router(intake.router)
app.include_router(scoring.router)
app.include_router(letters.router)
app.include_router(frontend.router)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="alt-credit-engine",
        model_version=get_model_version(),
    )
