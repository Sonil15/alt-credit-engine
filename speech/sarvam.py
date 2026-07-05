"""Sarvam AI speech-to-text (saarika) — primary STT provider, tuned for Indian
languages and accents."""

from __future__ import annotations

import httpx

from speech.provider import SpeechProvider

_STT_URL = "https://api.sarvam.ai/speech-to-text"

_LANGUAGE_CODES = {"en": "en-IN", "hi": "hi-IN", "bn": "bn-IN"}


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
