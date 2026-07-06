"""Voice I/O for the psychometric assessment.

STT: audio is transcribed and discarded — only the returned text is used, and the
borrower still reviews/edits it in the answer box before submitting, so a bad
transcription never silently becomes a wrong answer.

TTS: the agent's prompt text is synthesised to a one-shot audio clip and played in the
browser; nothing is stored. Whether a session uses server TTS (Sarvam ``bulbul``) or the
browser's own speech synthesis is a live UI toggle on the assessment page.
"""

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from speech import get_speech_provider, get_tts_provider

router = APIRouter(prefix="/speech", tags=["speech"])


@router.get("/config")
async def speech_config() -> dict:
    """Tells the frontend which server-side capabilities are available, so it knows
    whether to fall back to the browser's own Web Speech API (STT) / speech synthesis
    (TTS)."""
    return {
        "stt_available": get_speech_provider() is not None,
        "tts_available": get_tts_provider() is not None,
    }


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
        detail = str(exc) or exc.__class__.__name__
        raise HTTPException(status_code=502, detail=f"Transcription failed: {detail}") from exc

    return {"text": text}


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "en"


@router.post("/synthesize")
async def synthesize_speech(req: SynthesizeRequest) -> Response:
    """Return WAV audio of ``text`` spoken in ``language`` (Indian-language voices)."""
    provider = get_tts_provider()
    if provider is None:
        raise HTTPException(status_code=503, detail="Server-side text-to-speech is not configured")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        audio = await provider.synthesize(text, req.language)
    except Exception as exc:
        # Some transport errors stringify to "" — include the type so a real
        # failure isn't reported as a blank "Synthesis failed: ".
        detail = str(exc) or exc.__class__.__name__
        raise HTTPException(status_code=502, detail=f"Synthesis failed: {detail}") from exc

    if not audio:
        raise HTTPException(status_code=502, detail="Synthesis returned no audio")

    return Response(content=audio, media_type="audio/wav")
