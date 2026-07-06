"""Human-readable, per-language date formatting.

Any date shown to a borrower (a letter's decision date, a retry-after date in an
error message, ...) should read as a spoken sentence, not a raw ISO string, an
ISO date like "2026-08-05" gets read digit-by-digit by both a human eye and any
TTS engine, whereas "5 August 2026" reads naturally. Used anywhere a date is
embedded in borrower-facing text (see convergence/decision_letter.py and
api/routes/assessment.py).
"""

from __future__ import annotations

from datetime import date, datetime

_MONTH_NAMES: dict[str, list[str]] = {
    "en": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "hi": [
        "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
        "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
    ],
    "bn": [
        "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
        "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
    ],
}


def format_human_date(value: date | datetime, lang: str | None = "en") -> str:
    """Render ``value`` as "5 August 2026" (or the hi/bn equivalent).

    Falls back to English month names for an unrecognised language rather than
    raising, since a formatting helper should never be the reason a borrower-facing
    message fails to render.
    """
    months = _MONTH_NAMES.get(lang or "en", _MONTH_NAMES["en"])
    return f"{value.day} {months[value.month - 1]} {value.year}"
