from __future__ import annotations

import re

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "mechanics": [
        "gerak",
        "peluru",
        "parabola",
        "gaya",
        "gesekan",
        "katrol",
        "pegas",
        "momentum",
        "domino",
        "fluida",
        "tekanan",
        "debit",
        "dimensi",
        "balok",
        "bidang miring",
        "tumbukan",
        "hukum",
        "inkersia",
        "percepatan",
        "kecepatan",
        "gravitasi",
        "viskositas",
        "tabung",
        "aliran",
        "massa",
        "katapel",
        "plinteng",
        "lintasan",
        "tali",
        "rantai",
        "batang",
        "tongkat",
        "silinder",
        "bola",
        "orbit",
        "satelit",
        "inklinasi",
        "elevasi",
        "torsi",
        "momen",
        "inersia",
        "berputar",
        "menggelinding",
        "osilasi",
        "harmonik",
        "archimedes",
        "apung",
    ],
    "electromagnetism": [
        "listrik",
        "muatan",
        "kapasitor",
        "coulomb",
        "medan",
        "induksi",
        "arus",
        "resistor",
        "konduktor",
        "elektrostat",
        "elektromagnet",
        "baterai",
        "potensial",
        "permitivitas",
        "resistivitas",
        "rangkaian",
        "lampu",
        "kabel",
        "sensor",
        "kapasitansi",
        "induktansi",
        "mikrofon",
        "transformator",
    ],
    "thermodynamics": [
        "suhu",
        "kalor",
        "entropi",
        "gas",
        "termodinamika",
        "panas",
        "termal",
        "entropi",
        "termal",
        "entropy",
        "heat",
        "ideal gas",
        "mesin",
        "carnot",
        "kalorimetri",
        "kapasitas panas",
    ],
    "waves_optics": [
        "gelombang",
        "interferensi",
        "difraksi",
        "optik",
        "cahaya",
        "frekuensi",
        "lens",
        "cermin",
        "snell",
        "polar",
        "foton",
        "sound",
        "doppler",
        "resonansi",
        "standing wave",
        "gelombang berdiri",
        "lensa",
        "cermin",
    ],
    "modern_physics": [
        "relativitas",
        "quantum",
        "radioaktif",
        "fisika modern",
        "partikel",
        "nucleus",
        "decay",
        "fotoelektrik",
        "compton",
        "bohr",
        "nuklir",
        "radioaktif",
    ],
}

WORD_BOUNDARY = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    return set(WORD_BOUNDARY.findall(text.lower()))


def _score_topic(text: str, keywords: list[str]) -> float:
    tokens = _tokenize(text)
    score = 0.0
    for kw in keywords:
        kw_tokens = kw.split()
        if len(kw_tokens) == 1:
            if kw in tokens or kw in text.lower():
                score += 1.0
        elif kw in text.lower():
            score += 1.0
    return score


def classify_topic(
    title: str,
    body_md: str,
    *,
    title_weight: float = 3.0,
    body_weight: float = 1.0,
    mixed_threshold: float = 0.2,
) -> tuple[str, float, dict[str, float]]:
    scores: dict[str, float] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = (
            _score_topic(title, keywords) * title_weight
            + _score_topic(body_md, keywords) * body_weight
        )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return "mixed", 0.0, scores

    top_topic, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(scores.values()) or 1.0
    confidence = top_score / total

    if len(ranked) > 1 and second_score > 0:
        gap = (top_score - second_score) / top_score if top_score else 0
        if gap < mixed_threshold:
            return "mixed", confidence, scores

    return top_topic, confidence, scores


def llm_classify_topic(title: str, body_md: str) -> str | None:
    """Optional LLM tie-breaker; returns None if unavailable."""
    try:
        import os

        from src.llm.llm_client import ChatCompletionFailure, _llm_provider, chat_completion_json

        if _llm_provider() == "netra" and not os.environ.get("NETRA_API_KEY"):
            return None

        topics = ", ".join(TOPIC_KEYWORDS.keys()) + ", mixed"
        snippet = body_md[:2000]
        content = chat_completion_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Classify this physics olympiad problem into exactly one topic: "
                        f"{topics}. Reply with only the topic slug."
                    ),
                },
                {"role": "user", "content": f"Title: {title}\n\n{snippet}"},
            ],
            temperature=0,
            model=os.environ.get("TOPIC_TAG_MODEL", os.environ.get("HALLIDAY_TAG_MODEL", "qwen2.5:3b")),
        )
        if isinstance(content, ChatCompletionFailure):
            return None
        label = content.content.strip().lower()
        allowed = set(TOPIC_KEYWORDS) | {"mixed"}
        return label if label in allowed else None
    except Exception:
        return None
