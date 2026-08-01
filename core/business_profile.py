"""Borrower onboarding: business-profile extraction and intake-derived features.

The onboarding page lets an MSME borrower (Vendor / Farmer cohorts) describe
their business in their own words (English, Hindi, or Bengali). The LLM turns
that free text into structured fields the borrower then confirms or edits.
Nothing enters the pipeline unconfirmed. When the LLM is unavailable or
self-reports low confidence we fall back to a deterministic regex/keyword
extractor instead of guessing, mirroring the open-ended answer scorer in
``psychometric/session.py``.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from uuid import UUID

from groq import Groq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.feature_store import fetch_user_features_wide, upsert_feature
from models.db_models import ApplicationIntake

# Loan purposes recommended per borrower category - used to order the
# dropdown and to flag purpose–category *consistency* for the officer
# (`purpose_consistent` in the score payload). The server no longer hard-blocks
# a cross-cohort purpose (see ALL_PURPOSES below): a mismatch is a soft signal,
# not a wall, because real borrowers straddle categories (e.g. a student doing
# gig work). Every list ends in "other" with a free-text reason so a borrower
# is never forced into a purpose that doesn't fit.
PURPOSES_BY_COHORT: dict[str, list[str]] = {
    "Salaried": ["personal", "medical", "home_improvement", "vehicle", "other"],
    "GigWorker": ["personal", "medical", "home_improvement", "vehicle", "other"],
    "Student": ["education", "skill_course", "device_equipment", "hostel_rent", "medical", "other"],
    "Vendor": ["working_capital", "inventory", "equipment", "other"],
    "Farmer": ["crop_inputs", "equipment", "irrigation", "other"],
    "Homemaker": ["household", "small_home_business", "other"],
}

# Full set of known purpose codes, used only to reject typos/garbage - not to
# enforce cohort membership (that's the soft `purpose_consistent` signal).
ALL_PURPOSES: set[str] = {p for purposes in PURPOSES_BY_COHORT.values() for p in purposes}

# Cohorts that get the free-text business section at onboarding by default.
# Homemaker also gets it, conditionally, when the loan purpose is
# "small_home_business" (handled client-side; the server accepts a
# business_profile for any cohort regardless of this list).
BUSINESS_COHORTS = ("Vendor", "Farmer", "GigWorker")

# Model features derived from the borrower-confirmed business profile. They may still
# be structurally N/A (0) for non-business applicants, but must not surface in score
# explanations unless the borrower actually onboarded with a business profile.
BUSINESS_MODEL_FEATURES: tuple[str, ...] = (
    "business_vintage_years",
    "is_new_business",
    "turnover_income_consistency",
    "has_udyam_registration",
    "years_informal",
)


def business_features_applicable(
    cohort: str | None,
    loan_purpose: str | None = None,
    *,
    has_business_profile: bool = False,
) -> bool:
    """Whether business-profile features belong in borrower-facing explanations.

    Mirrors the onboarding UI: Vendor/Farmer always collect a business profile;
    GigWorker collects optionally; Homemaker does when the purpose is
    ``small_home_business``; any cohort that submitted a confirmed profile counts too.
    """
    if has_business_profile:
        return True
    if not cohort:
        return False
    if cohort in ("Vendor", "Farmer"):
        return True
    if cohort == "GigWorker":
        return has_business_profile
    return cohort == "Homemaker" and loan_purpose == "small_home_business"


# Minimum self-reported confidence before we trust the LLM's extraction over
# the deterministic fallback (same threshold philosophy as the answer scorer).
EXTRACTION_CONFIDENCE_THRESHOLD = 0.5

# Cache keyed by (model, language, normalized text) so repeated demo runs are
# stable and call-free.
_extraction_cache: dict[tuple[str, str, str], tuple[dict[str, Any], float, str]] = {}

_PROFILE_KEYS = ("sector", "years_in_business", "monthly_turnover", "seasonality", "employees")

# --- Deterministic fallback -------------------------------------------------

# Amount words across EN/HI/BN with their multipliers (lakh = 1e5, crore = 1e7,
# hazaar/k = 1e3). Devanagari and Bengali numerals are normalized first.
_MULTIPLIERS = [
    (r"(?:lakhs?|lacs?|लाख|লাখ|লক্ষ)", 100_000.0),
    (r"(?:crores?|करोड़|কোটি)", 10_000_000.0),
    (r"(?:thousands?|hazaar|hazar|हज़ार|हजार|হাজার|k)", 1_000.0),
]

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Ordered specific -> generic: retail goes last because its keywords
# (shop/stall/market) also appear in descriptions of other sectors; ties on
# match count resolve to the earlier (more specific) sector.
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "services": [
        "tailor", "silai", "सिलाई", "সেলাই", "salon", "parlour", "पार्लर",
        "repair", "मरम्मत", "মেরামত", "mechanic", "मैकेनिक", "wash", "cleaning",
        "tuition", "ट्यूशन", "টিউশন", "barber", "नाई",
    ],
    "food": [
        "food", "khana", "खाना", "খাবার", "tea", "chai", "चाय", "চা",
        "tiffin", "टिफ़िन", "hotel", "होटल", "dhaba", "ढाबा", "restaurant", "sweets", "मिठाई", "মিষ্টি",
    ],
    "manufacturing": [
        "manufactur", "factory", "फैक्ट्री", "কারখানা", "workshop", "handicraft",
        "हस्तशिल्प", "হস্তশিল্প", "weav", "बुनाई", "তাঁত", "pottery", "furniture",
    ],
    "agriculture": [
        "farm", "kheti", "खेती", "চাষ", "crop", "फसल", "ফসল", "harvest",
        "paddy", "धान", "ধান", "wheat", "गेहूं", "গম", "dairy", "दूध", "দুধ",
        "cattle", "पशु", "livestock", "poultry", "मुर्गी", "মুরগি", "acre", "बीघा", "जमीन",
    ],
    "retail": [
        "kirana", "किराना", "shop", "store", "dukaan", "dukan", "दुकान", "দোকান",
        "sabzi", "सब्ज़ी", "सब्जी", "সবজি", "vegetable", "fruit", "फल", "ফল",
        "stall", "thela", "ठेला", "market", "बाज़ार", "বাজার", "grocery", "sell", "बेच", "বিক্রি",
    ],
}

_SEASONALITY_KEYWORDS = {
    "high": [
        "seasonal", "मौसमी", "মৌসুমি", "festival", "त्योहार", "উৎসব",
        "harvest", "कटाई", "ফসল কাটা", "monsoon", "बरसात", "বর্ষা", "wedding season", "शादी",
    ],
    "low": ["year round", "साल भर", "সারা বছর", "steady", "regular", "रोज़", "daily", "प्रतिदिन"],
}

_YEARS_PATTERN = re.compile(
    r"(\d{1,2})\s*(?:\+\s*)?(?:years?|yrs?|saal|saalo?n|साल|वर्ष|बरस|বছর)",
    re.IGNORECASE,
)

_EMPLOYEES_PATTERN = re.compile(
    r"(\d{1,4})\s*(?:employees?|workers?|staff|helpers?|log|लोग|कर्मचारी|मजदूर|কর্মচারী|লোক|শ্রমিক)",
    re.IGNORECASE,
)

_AMOUNT_PATTERN = re.compile(
    r"(?:₹|rs\.?|rupees?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*({units})".format(
        units="|".join(m[0] for m in _MULTIPLIERS)
    ),
    re.IGNORECASE,
)

_PLAIN_AMOUNT_PATTERN = re.compile(r"(?:₹|rs\.?|rupees?|inr)\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)

# "40,000 rupees" / "৪০,০০০ টাকা": currency word AFTER the number.
_SUFFIX_AMOUNT_PATTERN = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:rupees?|rs\.?|₹|inr|रुपये|रुपए|টাকা|taka)", re.IGNORECASE
)

# Bare formatted number (e.g. "earning about 40,000 a month"): only trusted
# when income context words are present and the figure is money-sized.
_BARE_AMOUNT_PATTERN = re.compile(r"(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d{4,9})")

_TURNOVER_CONTEXT = re.compile(
    r"(?:earn|income|turnover|sales?|kamai|kamata|कमा|आमदनी|बिक्री|আয়|বিক্রি|রোজগার|profit|मुनाफ़ा)",
    re.IGNORECASE,
)


def _normalize_digits(text: str) -> str:
    return text.translate(_DEVANAGARI_DIGITS).translate(_BENGALI_DIGITS)


def _extract_amount(text: str) -> float | None:
    """Largest plausible money figure mentioned with a unit word or ₹ marker."""
    candidates: list[float] = []
    for match in _AMOUNT_PATTERN.finditer(text):
        number = float(match.group(1).replace(",", ""))
        unit = match.group(2).lower()
        for pattern, mult in _MULTIPLIERS:
            if re.fullmatch(pattern, unit, re.IGNORECASE):
                candidates.append(number * mult)
                break
    for pattern in (_PLAIN_AMOUNT_PATTERN, _SUFFIX_AMOUNT_PATTERN):
        for match in pattern.finditer(text):
            candidates.append(float(match.group(1).replace(",", "")))
    if not candidates:
        for match in _BARE_AMOUNT_PATTERN.finditer(text):
            value = float(match.group(1).replace(",", ""))
            if value >= 1000:
                candidates.append(value)
    if not candidates:
        return None
    return max(candidates)


def fallback_extract_business_profile(text: str) -> dict[str, Any]:
    """Deterministic regex/keyword extraction, no network, reproducible."""
    text = _normalize_digits(text)
    lowered = text.lower()
    profile: dict[str, Any] = {key: None for key in _PROFILE_KEYS}

    best_sector, best_hits = None, 0
    for sector, keywords in _SECTOR_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best_sector, best_hits = sector, hits
    profile["sector"] = best_sector

    years = _YEARS_PATTERN.search(text)
    if years:
        profile["years_in_business"] = float(years.group(1))

    employees = _EMPLOYEES_PATTERN.search(text)
    if employees:
        profile["employees"] = int(employees.group(1))

    # Only read an amount as turnover when income-ish context words appear;
    # otherwise a loan figure in the description would masquerade as turnover.
    if _TURNOVER_CONTEXT.search(text):
        amount = _extract_amount(text)
        if amount and amount > 0:
            profile["monthly_turnover"] = amount

    for level, keywords in _SEASONALITY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            profile["seasonality"] = level
            break

    return profile


# --- LLM extraction with confidence routing ---------------------------------


def _parse_groq_json(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _sanitize_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Clamp an LLM response to the known schema; junk values become None."""
    profile: dict[str, Any] = {key: None for key in _PROFILE_KEYS}

    sector = raw.get("sector")
    if isinstance(sector, str) and sector.strip():
        profile["sector"] = sector.strip().lower()[:50]

    try:
        years = float(raw.get("years_in_business"))
        if 0 <= years <= 80:
            profile["years_in_business"] = years
    except (TypeError, ValueError):
        pass

    try:
        turnover = float(raw.get("monthly_turnover"))
        if turnover > 0:
            profile["monthly_turnover"] = turnover
    except (TypeError, ValueError):
        pass

    seasonality = raw.get("seasonality")
    if isinstance(seasonality, str) and seasonality.strip().lower() in {"low", "medium", "high"}:
        profile["seasonality"] = seasonality.strip().lower()

    try:
        employees = int(raw.get("employees"))
        if 0 <= employees <= 10_000:
            profile["employees"] = employees
    except (TypeError, ValueError):
        pass

    return profile


def _resolve_extraction(parsed: dict[str, Any], text: str) -> tuple[dict[str, Any], float, str]:
    """Confidence routing (pure, no network): LLM result or deterministic fallback."""
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = 1.0

    profile = _sanitize_profile(parsed)
    if confidence < EXTRACTION_CONFIDENCE_THRESHOLD or all(
        profile[key] is None for key in _PROFILE_KEYS
    ):
        return fallback_extract_business_profile(text), 0.0, "fallback"
    return profile, max(0.0, min(1.0, confidence)), "llm"


async def extract_business_profile(text: str, language: str) -> tuple[dict[str, Any], float, str]:
    """Extract a structured business profile from free text.

    Returns ``(profile, confidence, method)`` where method is ``"llm"`` or
    ``"fallback"``. Works with no API key configured (fallback only).
    """
    settings = get_settings()
    cache_key = (settings.GROQ_MODEL, language, text.strip())
    if cache_key in _extraction_cache:
        return _extraction_cache[cache_key]

    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your_groq_api_key_here":
        result = (fallback_extract_business_profile(text), 0.0, "fallback")
        _extraction_cache[cache_key] = result
        return result

    prompt = (
        "A micro-enterprise borrower describes their business in their own words "
        "(English, Hindi, or Bengali). Extract ONLY facts they actually state, "
        "use null for anything not mentioned. Never guess or embellish. "
        "monthly_turnover is in Indian Rupees per month (convert lakh=100000, "
        "hazaar=1000; if they state a daily or yearly figure, convert to monthly). "
        "seasonality reflects how seasonal their income is. "
        "If they state they are just starting out, planning a business, or haven't started yet, "
        "set years_in_business to 0.\n"
        "Return ONLY JSON:\n"
        '{"sector": <string or null, one of retail/agriculture/services/food/manufacturing/other>, '
        '"years_in_business": <number or null>, '
        '"monthly_turnover": <number or null>, '
        '"seasonality": <"low"|"medium"|"high" or null>, '
        '"employees": <integer or null>, '
        '"confidence": <float 0.0 to 1.0>}\n'
        f"Language hint: {language}\nText:\n{text}"
    )

    def _call() -> dict[str, Any]:
        client = Groq(api_key=settings.GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You extract structured business facts. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            seed=7,
            max_tokens=200,
        )
        return _parse_groq_json(completion.choices[0].message.content or "{}")

    try:
        parsed = await asyncio.to_thread(_call)
        result = _resolve_extraction(parsed, text)
    except Exception:
        result = (fallback_extract_business_profile(text), 0.0, "fallback")

    _extraction_cache[cache_key] = result
    return result


# --- Intake-derived model features ------------------------------------------


async def fetch_latest_intake(session: AsyncSession, user_id: str) -> ApplicationIntake | None:
    result = await session.execute(
        select(ApplicationIntake)
        .where(ApplicationIntake.user_id == UUID(str(user_id)))
        .order_by(ApplicationIntake.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


def intake_to_dict(intake: ApplicationIntake | None) -> dict[str, Any] | None:
    """Plain-dict view of an intake row for the (DB-free) scoring payload builder."""
    if intake is None:
        return None
    return {
        "cohort": intake.cohort,
        "loan_purpose": intake.loan_purpose,
        "loan_purpose_other_text": intake.loan_purpose_other_text,
        "requested_amount": intake.requested_amount,
        "has_business_profile": bool(intake.business_profile_json),
    }


async def fetch_all_latest_intakes(session: AsyncSession) -> dict[str, dict[str, Any]]:
    """Latest intake per user as plain dicts, keyed by user_id (one query)."""
    result = await session.execute(
        select(ApplicationIntake).order_by(ApplicationIntake.created_at)
    )
    latest: dict[str, dict[str, Any]] = {}
    for intake in result.scalars():
        latest[str(intake.user_id)] = intake_to_dict(intake)  # later rows win
    return latest


def turnover_income_consistency(
    declared_monthly: float,
    observed_monthly: float,
    cohort: str | None = None,
    business_vintage_years: float | None = None,
) -> float:
    """Agreement between declared turnover and observed cash-flow income, in [0, 1].

    Tolerates cash-heavy operations by adjusting the expected digital footprint
    based on the borrower's cohort (e.g. MSMEs, Vendors, Farmers).
    1.0 = the self-report matches what the data shows (or is consistent with cash thresholds); near 0 = wildly apart.
    """
    if business_vintage_years is not None and business_vintage_years < 0.5:
        # Tier 1: Projection phase (<6 months), don't penalize consistency
        return 1.0

    if declared_monthly <= 0 or observed_monthly <= 0:
        return 0.0

    if declared_monthly > observed_monthly:
        # Determine the expected ratio of declared turnover to be seen in the bank statement
        if cohort == "Farmer":
            expected_digital_ratio = 0.20
        elif cohort in ("Vendor", "Homemaker"):
            expected_digital_ratio = 0.40
        elif cohort in ("GigWorker", "Student"):
            expected_digital_ratio = 0.80
        else:  # "Salaried" or default
            expected_digital_ratio = 0.90

        # Tier 2: Ramp-up phase (0.5 to 1.5 years), apply a 50% grace factor
        if business_vintage_years is not None and business_vintage_years < 1.5:
            expected_digital_ratio *= 0.5

        expected_observed = declared_monthly * expected_digital_ratio
        if observed_monthly >= expected_observed:
            return 1.0
        else:
            ratio = observed_monthly / expected_observed
    else:
        # observed >= declared
        ratio = declared_monthly / observed_monthly

    return round(max(0.0, min(1.0, ratio)), 4)


async def upsert_intake_features(session: AsyncSession, user_id: str) -> None:
    """Derive the intake model features and upsert them into ml_features.

    - ``business_vintage_years``: 0 means "no business / not stated".
    - ``is_new_business``: 1.0 if vintage < 1.0 else 0.0.
    - ``turnover_income_consistency``: declared turnover vs observed
      ``monthly_income_mean``; 0 means "nothing to cross-check".
    Individuals without a business profile simply get no rows. The model's
    fill-missing handles absent as 0.0, same as every other sparse feature.
    """
    intake = await fetch_latest_intake(session, user_id)
    if intake is None or not intake.business_profile_json:
        return
    try:
        profile = json.loads(intake.business_profile_json)
    except (TypeError, ValueError):
        return

    vintage = profile.get("years_in_business")
    vintage_val = None
    if vintage is not None:
        try:
            vintage_val = max(0.0, min(80.0, float(vintage)))
            await upsert_feature(session, user_id, "business_vintage_years", vintage_val)
            
            # Upsert is_new_business
            is_new = 1.0 if vintage_val < 1.0 else 0.0
            await upsert_feature(session, user_id, "is_new_business", is_new)
        except (TypeError, ValueError):
            pass

    declared = profile.get("monthly_turnover")
    if declared is not None:
        try:
            declared_val = float(declared)
        except (TypeError, ValueError):
            declared_val = 0.0
        if declared_val > 0:
            row = await fetch_user_features_wide(session, user_id)
            observed = 0.0
            if not row.empty and "monthly_income_mean" in row.columns:
                observed = float(row.iloc[0].get("monthly_income_mean") or 0.0)
            if observed > 0:
                await upsert_feature(
                    session,
                    user_id,
                    "turnover_income_consistency",
                    turnover_income_consistency(
                        declared_val,
                        observed,
                        cohort=intake.cohort,
                        business_vintage_years=vintage_val,
                    ),
                )

    udyam_num = profile.get("udyam_number")
    has_udyam = 1.0 if udyam_num else 0.0
    await upsert_feature(session, user_id, "has_udyam_registration", has_udyam)

    yrs_informal = profile.get("years_informal")
    if yrs_informal is not None:
        try:
            await upsert_feature(
                session, user_id, "years_informal", max(0.0, min(80.0, float(yrs_informal)))
            )
        except (TypeError, ValueError):
            pass

