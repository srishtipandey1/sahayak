"""Deterministic, headline-grounded multilingual question answering."""

from __future__ import annotations

import json
import re
from typing import Any

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id
from sahayak.report.i18n import message
from sahayak.report.verify import verify_text

INTENTS: dict[str, tuple[str, ...]] = {
    "where": (
        "where",
        "hit",
        "kahan",
        "कहाँ",
        "कहां",
        "असर कहाँ",
        "कहाँ असर",
        "कहाँ प्रभाव",
        "कहाँ भयो",
    ),
    "damage": (
        "damage",
        "damaged",
        "क्षति",
        "क्षतिग्रस्त",
        "नुकसान",
        "नोक्सान",
        "के बिग्रियो",
    ),
    "counts": (
        "count",
        "how many",
        "numbers",
        "कितने",
        "कितनी",
        "संख्या",
        "कति",
        "कति वटा",
    ),
    "villages": (
        "cut off",
        "cut-off",
        "cutoff",
        "villages",
        "settlements cut",
        "कौन सी बस्ती",
        "कौन सी बस्तियाँ",
        "कौन सी बस्तियां",
        "कौन-सी बस्ती",
        "कटी बस्ती",
        "बस्तियाँ कटी",
        "कटे हुए",
        "कुन बस्ती",
        "सम्पर्कविच्छेद",
        "सम्पर्क विच्छेद",
    ),
    "hospital": (
        "hospital",
        "nearest clinic",
        "nearest health",
        "अस्पताल",
        "चिकित्सालय",
        "नजिकको अस्पताल",
        "नजिक अस्पताल",
        "स्वास्थ्य केन्द्र",
    ),
    "reliability": (
        "reliability",
        "reliable",
        "विश्वसनीयता",
        "भरोसेमंद",
        "विश्वसनीय",
        "कत्तिको भरपर्दो",
    ),
    "limitations": (
        "limitation",
        "limitations",
        "सीमा",
        "सीमाएँ",
        "सीमाएं",
        "सीमित",
        "सीमाहरू",
        "कमजोरी",
    ),
    "data": (
        "data used",
        "what data",
        "which data",
        "dataset",
        "data source",
        "डेटा",
        "तथ्यांक",
        "तथ्याङ्क",
        "कुन तथ्याङ्क",
        "प्रयोग गरिएको डेटा",
        "प्रयोग गरिएका तथ्याङ्क",
    ),
}


def route_intent(question: str) -> str | None:
    """Route an English, Hindi, or Nepali question using keyword patterns."""
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    if not normalized:
        return None
    for intent, keywords in INTENTS.items():
        if any(keyword.casefold() in normalized for keyword in keywords):
            return intent
    return None


def _value(headline: list[dict[str, Any]], fact_id: str) -> Any:
    """Find a curated headline value by its stable identifier."""
    for item in headline:
        if item.get("id") == fact_id:
            return item.get("value")
    return None


def _fmt(value: Any, unit: str | None = None) -> str:
    """Format a headline value using Western digits and explicit missing text."""
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if unit in {"buildings", "bridges", "settlements", "minutes"}:
            return str(int(value)) if float(value).is_integer() else f"{value:g}"
        return f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(str(part) for part in value) if value else "not available"
    return str(value)


def _item(headline: list[dict[str, Any]], fact_id: str) -> dict[str, Any] | None:
    return next((item for item in headline if item.get("id") == fact_id), None)


def _answer(
    intent: str,
    language: str,
    question: str,
    headline: list[dict[str, Any]],
) -> str:
    """Create an answer from curated headline values only."""
    none = message(language, "none")
    if intent == "where":
        return message(
            language,
            "answer_hit",
            area=_fmt(_value(headline, "hazard_area_default")),
            bbox=_fmt(_value(headline, "focus_bbox")),
        )
    if intent == "damage":
        return message(
            language,
            "answer_damage",
            buildings=_fmt(_value(headline, "buildings_inside_default"), "buildings"),
            near=_fmt(_value(headline, "buildings_near_30m_default"), "buildings"),
            roads=_fmt(_value(headline, "road_km_hit_default")),
            bridges=_fmt(_value(headline, "bridges_hit_within_20m_default"), "bridges"),
        )
    if intent == "counts":
        return message(
            language,
            "answer_counts",
            cut_off=_fmt(_value(headline, "cut_off_default"), "settlements"),
            degraded=_fmt(_value(headline, "degraded_default"), "settlements"),
            unaffected=_fmt(_value(headline, "unaffected_default"), "settlements"),
            not_assessable=_fmt(_value(headline, "not_assessable_default"), "settlements"),
        )
    if intent == "villages":
        names = _value(headline, "cut_off_names_default")
        displayed_names = _fmt(names) if names is not None else none
        return message(language, "answer_villages", names=displayed_names)
    if intent == "hospital":
        candidates = [
            item
            for item in headline
            if item.get("id", "").startswith("hospital_route_")
            and item.get("id", "").endswith("_name")
            and item.get("value")
        ]
        requested = question.casefold()
        chosen = next(
            (
                item
                for item in candidates
                if (str(item.get("label_en", "")).split(" for ", 1)[-1].casefold() in requested)
            ),
            candidates[0] if len(candidates) == 1 else None,
        )
        if chosen is None:
            return message(language, "unsupported")
        settlement = str(chosen["label_en"]).split(" for ", 1)[-1]
        route_id = str(chosen["id"]).removesuffix("_name")
        hospital = chosen.get("value") or message(language, "hospital_missing")
        distance = _value(headline, f"{route_id}_km")
        minutes = _value(headline, f"{route_id}_minutes")
        return message(
            language,
            "answer_hospital",
            settlement=settlement,
            hospital=hospital,
            distance=_fmt(distance),
            minutes=_fmt(minutes, "minutes"),
        )
    if intent == "reliability":
        valid = _value(headline, "optical_valid_fraction")
        valid_display = f"{float(valid) * 100:.1f}%" if isinstance(valid, (int, float)) else none
        return message(
            language,
            "answer_reliability",
            reliability=_fmt(_value(headline, "optical_reliability")),
            valid=valid_display,
        )
    if intent == "limitations":
        return message(language, "answer_limitations")
    if intent == "data":
        sources = _value(headline, "data_sources")
        dates = [
            str(item["value"])[:10]
            for item in headline
            if item.get("id") in {"event_date", "optical_pre_date", "optical_post_date"}
            and item.get("value")
        ]
        return message(
            language,
            "answer_data",
            sources=_fmt(sources) if sources is not None else none,
            dates=", ".join(dates) if dates else none,
        )
    return message(language, "unsupported")


def answer_question(
    *,
    run_id: str,
    language: str,
    question: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Answer using headline.json and reject any ungrounded numeric token."""
    if language not in {"en", "hi", "ne"}:
        raise ValueError("Language must be en, hi, or ne.")
    if not question.strip():
        raise ValueError("Question must not be empty.")
    validate_run_id(run_id)
    active_settings = settings or Settings()
    headline_path = active_settings.resolved_runs_dir / run_id / "headline.json"
    if not headline_path.is_file():
        raise FileNotFoundError(f"Missing curated headline file: {headline_path.name}")
    headline = json.loads(headline_path.read_text(encoding="utf-8"))
    if not isinstance(headline, list):
        raise ValueError("headline.json must contain a list of curated facts.")
    return answer_from_headline(
        headline=headline,
        language=language,
        question=question,
    )


def answer_from_headline(
    *,
    headline: list[dict[str, Any]],
    language: str,
    question: str,
) -> dict[str, Any]:
    """Answer a question from an already loaded curated headline."""
    if language not in {"en", "hi", "ne"}:
        raise ValueError("Language must be en, hi, or ne.")
    if not question.strip():
        raise ValueError("Question must not be empty.")
    intent = route_intent(question)
    if intent is None:
        answer = message(language, "unsupported")
    else:
        answer = _answer(intent, language, question, headline)
    verification = verify_text(answer, headline)
    if not verification["verified"]:
        raise RuntimeError(
            "Grounded answer contains unverified numeric token(s): "
            + ", ".join(verification["unverified_numbers"])
        )
    return {"intent": intent, "answer": answer, "verification": verification}
