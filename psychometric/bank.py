"""Load and validate the psychometric item bank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_LANGUAGES = ("en", "hi", "bn")
ITEM_BANK_PATH = Path(__file__).parent / "item_bank.json"


@dataclass(frozen=True)
class Item:
    id: str
    construct: str
    type: str
    options: list[str]
    scoring_key: dict[str, float]
    text: dict[str, str]
    consistency_pair: str | None = None

    def prompt(self, language: str) -> str:
        lang = language if language in SUPPORTED_LANGUAGES else "en"
        return self.text.get(lang, self.text["en"])

    def option_labels(self, language: str, likert_labels: dict[str, list[str]]) -> list[str]:
        if self.type != "likert":
            return []
        labels = likert_labels.get(language, likert_labels["en"])
        return labels[: len(self.options)]


def load_item_bank(path: Path | None = None) -> dict[str, Any]:
    target = path or ITEM_BANK_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def load_items(path: Path | None = None) -> list[Item]:
    bank = load_item_bank(path)
    items: list[Item] = []
    for raw in bank["items"]:
        items.append(
            Item(
                id=raw["id"],
                construct=raw["construct"],
                type=raw["type"],
                options=list(raw.get("options", [])),
                scoring_key={str(k): float(v) for k, v in raw.get("scoring_key", {}).items()},
                text=dict(raw["text"]),
                consistency_pair=raw.get("consistency_pair"),
            )
        )
    return items


def get_item_by_id(item_id: str, path: Path | None = None) -> Item | None:
    return next((item for item in load_items(path) if item.id == item_id), None)


def validate_item_bank(path: Path | None = None) -> list[str]:
    """Return list of validation errors (empty if valid)."""
    errors: list[str] = []
    bank = load_item_bank(path)
    items = load_items(path)
    ids = {item.id for item in items}

    for item in items:
        for lang in SUPPORTED_LANGUAGES:
            if lang not in item.text or not item.text[lang].strip():
                errors.append(f"Item {item.id} missing translation for '{lang}'")
        if item.type == "likert":
            if not item.options:
                errors.append(f"Likert item {item.id} has no options")
            if set(item.scoring_key.keys()) != set(item.options):
                errors.append(f"Item {item.id} scoring_key does not match options")
        if item.consistency_pair and item.consistency_pair not in ids:
            errors.append(f"Item {item.id} references unknown consistency_pair {item.consistency_pair}")

    for lang in SUPPORTED_LANGUAGES:
        if lang not in bank.get("likert_labels", {}):
            errors.append(f"Missing likert_labels for '{lang}'")

    return errors
