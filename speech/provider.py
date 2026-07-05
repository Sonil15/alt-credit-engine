"""Abstract speech-to-text provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SpeechProvider(ABC):
    """Transcribes a single recorded utterance to text in the given language."""

    @abstractmethod
    async def transcribe(self, audio: bytes, language: str, mime_type: str) -> str:
        """Return the transcribed text for ``audio`` (assumed spoken in ``language``)."""
