"""Speech provider layer for the psychometric assessment.

Speech-to-text (STT) lets borrowers dictate open-ended answers; text-to-speech (TTS)
reads the agent's prompts aloud. The audio never leaves this layer as anything but
transcribed text (STT) or a synthesised clip played once (TTS) — nothing is persisted.

STT provider is chosen by ``SPEECH_STT_PROVIDER`` so a live API hiccup during a demo is
an env-var flip; it falls back to the browser's own Web Speech API (see static/voice.js)
when no server provider is configured. Server TTS (Sarvam ``bulbul``) is available
whenever a Sarvam key is set, but whether it is *used* is a live per-session UI toggle on
the assessment page — toggling it off reverts to the browser's built-in speech synthesis.
"""

from __future__ import annotations

from functools import lru_cache

from core.config import get_settings
from speech.provider import SpeechProvider, TTSProvider


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


@lru_cache
def get_tts_provider() -> TTSProvider | None:
    """Server-side text-to-speech. Available whenever Sarvam is configured; the
    frontend's UI toggle decides whether each session actually uses it."""
    settings = get_settings()
    if settings.SARVAM_API_KEY:
        from speech.sarvam import SarvamTTSProvider

        return SarvamTTSProvider(settings.SARVAM_API_KEY)
    return None


__all__ = ["SpeechProvider", "TTSProvider", "get_speech_provider", "get_tts_provider"]
