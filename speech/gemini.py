"""Gemini speech-to-text fallback — used when Sarvam is unavailable or unconfigured."""

from __future__ import annotations

import base64

import httpx

from speech.provider import SpeechProvider

_MODEL = "gemini-2.0-flash"
_GENERATE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:generateContent"

_LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali"}


class GeminiSpeechProvider(SpeechProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def transcribe(self, audio: bytes, language: str, mime_type: str) -> str:
        language_name = _LANGUAGE_NAMES.get(language, "English")
        prompt = (
            f"Transcribe this {language_name} audio exactly as spoken. "
            "Respond with only the transcription, no commentary."
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type or "audio/webm",
                                "data": base64.b64encode(audio).decode("ascii"),
                            }
                        },
                    ]
                }
            ]
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                _GENERATE_URL,
                params={"key": self._api_key},
                json=payload,
            )
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip()
