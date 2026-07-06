from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["frontend"])

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "frontend"
STATIC_DIR = TEMPLATES_DIR / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Landing page with paths to dashboard and borrower consent flow."""
    return templates.TemplateResponse(request, "index.html", {"request": request})


@router.get("/consent", response_class=HTMLResponse)
async def consent_gateway(request: Request) -> HTMLResponse:
    """Mobile consent gateway UI."""
    return templates.TemplateResponse(request, "consent.html", {"request": request})


@router.get("/assessment", response_class=HTMLResponse)
async def assessment_ui(request: Request) -> HTMLResponse:
    """Multilingual agentic psychometric assessment UI."""
    return templates.TemplateResponse(request, "assessment.html", {"request": request})


@router.get("/apply", response_class=HTMLResponse)
async def borrower_portal(request: Request) -> HTMLResponse:
    """Borrower portal: start application or view previous results."""
    return templates.TemplateResponse(request, "apply.html", {"request": request})


@router.get("/onboard", response_class=HTMLResponse)
async def borrower_onboarding(request: Request) -> HTMLResponse:
    """Borrower onboarding (loan intent, category, and business profile."""
    return templates.TemplateResponse(request, "onboard.html", {"request": request})


@router.get("/login", response_class=HTMLResponse)
async def borrower_login(request: Request) -> HTMLResponse:
    """Borrower sign-in / account-creation page."""
    return templates.TemplateResponse(request, "login.html", {"request": request})


@router.get("/borrower", response_class=HTMLResponse)
async def borrower_result(request: Request) -> HTMLResponse:
    """Borrower-only score result page (session-authenticated)."""
    return templates.TemplateResponse(request, "borrower.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def bank_dashboard(request: Request, user_id: str | None = None) -> HTMLResponse:
    """Bank LOS dashboard UI."""
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"request": request, "user_id": user_id or ""},
    )


def mount_static(app) -> None:
    """Mount static assets (voice.js) if directory exists."""
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
