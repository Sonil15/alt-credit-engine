"""Voice-answer transcription for the psychometric assessment.

Audio is transcribed and discarded — only the returned text is used, and the
borrower still reviews/edits it in the answer box before submitting, so a bad
transcription never silently becomes a wrong answer.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from speech import get_speech_provider

router = APIRouter(prefix="/speech", tags=["speech"])


@router.get("/config")
async def speech_config() -> dict:
    """Tells the frontend whether server-side STT is available, so it knows
    whether to fall back to the browser's own Web Speech API."""
    return {"stt_available": get_speech_provider() is not None}


@router.post("/transcribe")
async def transcribe_audio(
    language: str = Form("en"),
    audio: UploadFile = File(...),
) -> dict:
    provider = get_speech_provider()
    if provider is None:
        raise HTTPException(status_code=503, detail="Server-side speech-to-text is not configured")

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        text = await provider.transcribe(data, language, audio.content_type or "audio/webm")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc

    return {"text": text}
