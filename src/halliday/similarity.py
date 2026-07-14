from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.halliday.classify import ProblemTags
from src.schema import ProblemRecord

TOKEN_RE = re.compile(r"[a-zA-Z]{3,}|\d+")


@dataclass
class SimilarNeighbor:
    id: str
    score: float
    shared_topics: list[str]
    shared_details: list[str]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "score": round(self.score, 4),
            "shared_topics": self.shared_topics,
            "shared_details": self.shared_details,
            # Legacy keys for older clients
            "shared_chapters": self.shared_topics,
            "shared_sections": self.shared_details,
        }


def tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"\$+[^$]+\$+", " ", text)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    return TOKEN_RE.findall(cleaned.lower())


def _tfidf_vectors(records: list[ProblemRecord]) -> tuple[list[dict[str, float]], dict[str, float]]:
    docs = [tokenize(f"{r.title} {r.body_md}") for r in records]
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    n = len(docs)
    idf = {term: math.log((1 + n) / (1 + freq)) + 1.0 for term, freq in df.items()}
    vectors: list[dict[str, float]] = []
    for doc in docs:
        tf = Counter(doc)
        total = sum(tf.values()) or 1
        vec = {t: (c / total) * idf.get(t, 0.0) for t, c in tf.items()}
        vectors.append(vec)
    return vectors, idf


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _tag_similarity(
    a: ProblemTags,
    b: ProblemTags,
) -> tuple[float, list[str], list[str]]:
    top_a, top_b = set(a.topics), set(b.topics)
    det_a, det_b = set(a.details), set(b.details)
    shared_topics = sorted(top_a & top_b)
    shared_details = sorted(det_a & det_b)
    topic_j = _jaccard(top_a, top_b)
    detail_j = _jaccard(det_a, det_b)
    score = 0.4 * topic_j + 0.6 * detail_j
    return score, shared_topics, shared_details


def build_similarity_index(
    records: list[ProblemRecord],
    tags_by_id: dict[str, ProblemTags],
    *,
    top_k: int = 8,
    text_weight: float = 0.55,
    tag_weight: float = 0.45,
    same_level_bonus: float = 0.05,
) -> dict[str, list[SimilarNeighbor]]:
    vectors, _ = _tfidf_vectors(records)
    index: dict[str, list[SimilarNeighbor]] = {}

    for i, rec in enumerate(records):
        tags_i = tags_by_id.get(rec.id)
        if tags_i is None:
            continue
        neighbors: list[SimilarNeighbor] = []
        for j, other in enumerate(records):
            if i == j:
                continue
            tags_j = tags_by_id.get(other.id)
            if tags_j is None:
                continue
            text_sim = _cosine(vectors[i], vectors[j])
            tag_sim, shared_topics, shared_details = _tag_similarity(tags_i, tags_j)
            score = text_weight * text_sim + tag_weight * tag_sim
            if rec.level and rec.level == other.level:
                score += same_level_bonus
            if score < 0.08:
                continue
            neighbors.append(
                SimilarNeighbor(
                    id=other.id,
                    score=score,
                    shared_topics=shared_topics,
                    shared_details=shared_details,
                )
            )
        neighbors.sort(key=lambda n: (-n.score, n.id))
        index[rec.id] = neighbors[:top_k]

    return index
