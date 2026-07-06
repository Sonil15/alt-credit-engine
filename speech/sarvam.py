"""Sarvam AI speech — STT (saarika) and TTS (bulbul), both tuned for Indian
languages and accents."""

from __future__ import annotations

import asyncio
import base64

import httpx

from speech.provider import SpeechProvider, TTSProvider

_STT_URL = "https://api.sarvam.ai/speech-to-text"
_TTS_URL = "https://api.sarvam.ai/text-to-speech"

_LANGUAGE_CODES = {"en": "en-IN", "hi": "hi-IN", "bn": "bn-IN"}

# bulbul:v2 accepts up to ~1500 characters per request; cap defensively so a long
# greeting can never trip a 400 (assessment prompts are far shorter in practice).
_TTS_MAX_CHARS = 1500
_TTS_SPEAKER = "anushka"  # multilingual bulbul:v2 voice; works across en/hi/bn


class SarvamSpeechProvider(SpeechProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def transcribe(self, audio: bytes, language: str, mime_type: str) -> str:
        language_code = _LANGUAGE_CODES.get(language, "unknown")
        # Sarvam matches content-type as an exact string and rejects codec
        # parameters (e.g. "audio/webm;codecs=opus", what browsers send) even
        # though the bare "audio/webm" is on its allow-list.
        clean_mime_type = (mime_type or "audio/webm").split(";")[0].strip()
        files = {"file": ("audio.webm", audio, clean_mime_type)}
        data = {"model": "saarika:v2.5", "language_code": language_code}
        headers = {"api-subscription-key": self._api_key}

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(_STT_URL, headers=headers, data=data, files=files)
        if response.status_code >= 400:
            # Sarvam puts the actual reason in the body; httpx's default error text
            # only has the status code, which hides why a 400 happened.
            raise httpx.HTTPStatusError(
                f"Sarvam STT {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )
        return (response.json().get("transcript") or "").strip()


class SarvamTTSProvider(TTSProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def synthesize(self, text: str, language: str) -> bytes:
        language_code = _LANGUAGE_CODES.get(language, "en-IN")
        body = {
            "text": text[:_TTS_MAX_CHARS],
            "target_language_code": language_code,
            "speaker": _TTS_SPEAKER,
            "model": "bulbul:v2",
        }
        headers = {"api-subscription-key": self._api_key, "Content-Type": "application/json"}

        # Sarvam's free tier throttles rapid successive connections with a
        # transport-level reset (not an HTTP error). Assessment prompts are
        # naturally seconds apart so this is rare, but one short retry smooths over
        # a burst (e.g. a presenter clicking quickly) without masking real errors.
        response = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(_TTS_URL, headers=headers, json=body)
                break
            except httpx.TransportError:
                if attempt == 0:
                    await asyncio.sleep(0.6)
                    continue
                raise
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Sarvam TTS {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )
        # Sarvam returns a list of base64-encoded WAV chunks; join their decoded
        # bytes. For our short single-utterance prompts there is only ever one.
        audios = response.json().get("audios") or []
        return b"".join(base64.b64decode(chunk) for chunk in audios)
