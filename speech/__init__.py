"""Speech-to-text provider layer for voice answers in the psychometric assessment.

Borrowers dictate their answer instead of typing; the audio never leaves this layer as
anything but transcribed text, and no audio is persisted. Provider is chosen by
``SPEECH_STT_PROVIDER`` so a live API hiccup during a demo is an env-var flip, not a
code change. Falls back to the browser's own Web Speech API (see static/voice.js) when
no server provider is configured.
"""

from __future__ import annotations

from functools import lru_cache

from core.config import get_settings
from speech.provider import SpeechProvider


@lru_cache
def get_speech_provider() -> SpeechProvider | None:
    settings = get_settings()
    name = settings.SPEECH_STT_PROVIDER.strip().lower()

    if name == "sarvam" and settings.SARVAM_API_KEY:
        from speech.sarvam import SarvamSpeechProvider

        return SarvamSpeechProvider(settings.SARVAM_API_KEY)
    if name == "gemini" and settings.GEMINI_API_KEY:
        from speech.gemini import GeminiSpeechProvider

        return GeminiSpeechProvider(settings.GEMINI_API_KEY)
    return None


__all__ = ["SpeechProvider", "get_speech_provider"]
