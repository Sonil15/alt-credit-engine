from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["frontend"])

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "frontend"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/consent", response_class=HTMLResponse)
async def consent_gateway(request: Request) -> HTMLResponse:
    """Mobile consent gateway UI."""
    return templates.TemplateResponse(request, "consent.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def bank_dashboard(request: Request, user_id: str | None = None) -> HTMLResponse:
    """Bank LOS dashboard UI."""
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"request": request, "user_id": user_id or ""},
    )
