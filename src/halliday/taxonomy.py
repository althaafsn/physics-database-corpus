from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "data" / "physics-tags-taxonomy.json"


@dataclass(frozen=True)
class PhysicsDetail:
    id: str
    title: str


@dataclass(frozen=True)
class PhysicsTopic:
    id: str
    title: str
    discipline: str
    details: tuple[PhysicsDetail, ...]


@dataclass(frozen=True)
class PhysicsDiscipline:
    id: str
    title: str
    topics: tuple[str, ...]


@dataclass(frozen=True)
class PhysicsTagTaxonomy:
    version: int
    disciplines: tuple[PhysicsDiscipline, ...]
    topics: tuple[PhysicsTopic, ...]

    def topic_by_id(self, topic_id: str) -> PhysicsTopic | None:
        for topic in self.topics:
            if topic.id == topic_id:
                return topic
        return None

    def detail_by_id(self, detail_id: str) -> PhysicsDetail | None:
        for topic in self.topics:
            for detail in topic.details:
                if detail.id == detail_id:
                    return detail
        return None

    def topic_for_detail(self, detail_id: str) -> str | None:
        for topic in self.topics:
            if any(d.id == detail_id for d in topic.details):
                return topic.id
        return None

    def discipline_for_topic(self, topic_id: str) -> str | None:
        topic = self.topic_by_id(topic_id)
        return topic.discipline if topic else None

    def topics_for_discipline(self, discipline_id: str) -> list[PhysicsTopic]:
        return [t for t in self.topics if t.discipline == discipline_id]

    def valid_topic_ids(self) -> set[str]:
        return {t.id for t in self.topics}

    def valid_detail_ids(self) -> set[str]:
        return {d.id for t in self.topics for d in t.details}

    def topic_labels(self) -> dict[str, str]:
        return {t.id: t.title for t in self.topics}

    def detail_labels(self) -> dict[str, str]:
        return {d.id: d.title for t in self.topics for d in t.details}


@lru_cache(maxsize=1)
def load_taxonomy() -> PhysicsTagTaxonomy:
    data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    disciplines = tuple(
        PhysicsDiscipline(
            id=d["id"],
            title=d["title"],
            topics=tuple(d["topics"]),
        )
        for d in data["disciplines"]
    )
    topics = tuple(
        PhysicsTopic(
            id=t["id"],
            title=t["title"],
            discipline=t["discipline"],
            details=tuple(
                PhysicsDetail(id=d["id"], title=d["title"]) for d in t["details"]
            ),
        )
        for t in data["topics"]
    )
    return PhysicsTagTaxonomy(
        version=int(data.get("version", 2)),
        disciplines=disciplines,
        topics=topics,
    )
