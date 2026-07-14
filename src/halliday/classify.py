from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from src.halliday.taxonomy import PhysicsTagTaxonomy, load_taxonomy
from src.schema import ProblemRecord


def _default_tag_model() -> str:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in {"local", "ollama"} or os.environ.get("LOCAL_LLM_BASE_URL", "").strip():
        return os.environ.get("HALLIDAY_TAG_MODEL", "qwen2.5:3b")
    return os.environ.get("HALLIDAY_TAG_MODEL", "qwen3.6-35b")


COARSE_TOPIC_PRIORS: dict[str, list[str]] = {
    "mechanics": [
        "units-and-vectors",
        "kinematics",
        "newtons-laws-and-forces",
        "work-and-energy",
        "momentum-and-collisions",
        "rotational-motion",
        "statics-and-elasticity",
        "gravitation-and-orbits",
        "fluid-mechanics",
        "simple-harmonic-motion",
    ],
    "electromagnetism": [
        "electrostatics",
        "gauss-law",
        "electric-potential",
        "dc-circuits-and-capacitance",
        "magnetostatics",
        "electromagnetic-induction",
        "ac-circuits-and-em-waves",
    ],
    "thermodynamics": [
        "temperature-and-heat-transfer",
        "kinetic-theory-and-gases",
        "laws-of-thermodynamics",
    ],
    "waves_optics": [
        "simple-harmonic-motion",
        "mechanical-waves-and-sound",
        "geometric-optics",
        "physical-optics",
        "ac-circuits-and-em-waves",
    ],
    "waves": ["simple-harmonic-motion", "mechanical-waves-and-sound"],
    "optics": ["geometric-optics", "physical-optics"],
    "modern_physics": [
        "special-relativity",
        "quantum-mechanics",
        "atomic-physics",
        "solid-state-physics",
        "nuclear-physics",
        "particle-physics-and-cosmology",
    ],
    "modern": [
        "special-relativity",
        "quantum-mechanics",
        "atomic-physics",
        "solid-state-physics",
        "nuclear-physics",
        "particle-physics-and-cosmology",
    ],
    "mixed": [t.id for t in load_taxonomy().topics],
}

DETAIL_KEYWORDS: dict[str, list[str]] = {
    "si-units": ["satuan", "si ", "pengukuran", "dimensi"],
    "significant-figures": ["significant", "angka penting"],
    "vector-addition": ["vektor", "vector", "komponen vektor"],
    "dot-product": ["dot product", "perkalian titik", "skalar"],
    "cross-product": ["cross product", "perkalian silang"],
    "one-dimensional-motion": ["perpindahan", "kecepatan rata", "displacement", "1d motion"],
    "constant-acceleration": ["glbb", "glb", "percepatan konstan", "constant acceleration", "jatuhan bebas", "free fall"],
    "two-dimensional-motion": ["gerak 2 dimensi", "2d motion", "dua dimensi"],
    "projectile-motion": ["parabola", "projectile", "gerak parabola", "peluru"],
    "relative-motion": ["relatif", "relative motion"],
    "graphical-integration": ["grafik", "graphical integration"],
    "newtons-laws": ["hukum newton", "newton", "gaya resultan", "hukum i", "hukum ii", "hukum iii"],
    "friction": ["gesekan", "friction", "koefisien gesek"],
    "drag-forces": ["drag", "hambat udara", "terminal speed", "kecepatan terminal"],
    "uniform-circular-motion-dynamics": ["gerak melingkar", "circular motion", "gaya sentripetal", "centripetal"],
    "kinetic-energy": ["energi kinetik", "kinetic energy"],
    "work": ["kerja", "work ", "usaha"],
    "gravitational-work": ["kerja gravitasi", "gravitational work"],
    "spring-work": ["pegas", "spring", "elastis"],
    "variable-force-work": ["gaya variabel", "variable force"],
    "power": ["daya", "power "],
    "potential-energy": ["energi potensial", "potential energy"],
    "conservation-mechanical-energy": ["kekekalan energi mekanik", "conservation of mechanical energy", "energi mekanik"],
    "center-of-mass": ["pusat massa", "center of mass"],
    "impulse": ["impuls", "impulse"],
    "linear-momentum": ["momentum", "impuls momentum"],
    "inelastic-collisions": ["tumbukan tidak elastis", "inelastic"],
    "elastic-collisions": ["tumbukan elastis", "elastic collision"],
    "two-dimensional-collisions": ["tumbukan 2 dimensi", "collision 2d"],
    "variable-mass-systems": ["roket", "rocket", "massa variabel"],
    "angular-kinematics": ["sudut", "angular", "percepatan sudut"],
    "moment-of-inertia": ["momen inersia", "moment of inertia", "inersia"],
    "torque": ["torsi", "torque", "momen gaya"],
    "rolling-motion": ["menggelinding", "rolling"],
    "angular-momentum": ["momentum sudut", "angular momentum"],
    "gyroscope-precession": ["gyroscope", "presesi", "precession"],
    "static-equilibrium": ["kesetimbangan", "equilibrium", "statika"],
    "elasticity": ["elastisitas", "elasticity", "modulus"],
    "newtons-law-gravitation": ["hukum gravitasi", "gravitation", "gravitas"],
    "gravitational-potential": ["potensial gravitasi", "gravitational potential"],
    "gravitation-near-earth": ["di dalam bumi", "inside earth"],
    "keplers-laws": ["kepler", "orbit", "satelit", "satellite"],
    "density-pressure": ["kerapatan", "density", "tekanan", "pressure"],
    "fluids-at-rest": ["fluida diam", "fluid at rest", "hidrostatik"],
    "pascals-principle": ["pascal"],
    "archimedes-principle": ["archimedes", "arkimedes", "apung", "buoyancy"],
    "continuity-equation": ["kontinuitas", "continuity"],
    "bernoullis-equation": ["bernoulli"],
    "simple-harmonic-oscillators": ["harmonik", "simple harmonic", "shm", "osilasi"],
    "pendulums": ["pendulum", "bandul"],
    "damped-oscillations": ["teredam", "damped"],
    "forced-oscillations": ["paksa", "forced oscillation", "resonansi", "resonance"],
    "transverse-waves": ["gelombang pada tali", "transverse wave", "string wave"],
    "wave-equation": ["persamaan gelombang", "wave equation", "gelombang berdiri", "standing wave"],
    "sound-speed": ["kecepatan bunyi", "speed of sound"],
    "doppler-effect": ["doppler"],
    "beats": ["ketukan", "beats"],
    "intensity-sound-level": ["intensitas bunyi", "sound level", "desibel", "decibel"],
    "temperature-scales": ["celsius", "fahrenheit", "suhu", "temperature"],
    "thermal-expansion": ["muai", "thermal expansion", "expansi"],
    "specific-heat": ["kalor jenis", "specific heat", "panas"],
    "heat-transfer-mechanisms": ["konduksi", "konveksi", "radiasi", "heat transfer"],
    "ideal-gas-law": ["gas ideal", "ideal gas", "avogadro"],
    "rms-speed": ["rms", "kecepatan rms"],
    "mean-free-path": ["jarak bebas rata", "mean free path"],
    "molecular-speed-distribution": ["distribusi kecepatan", "maxwell-boltzmann"],
    "molar-specific-heats": ["kalor molar", "molar specific heat", "derajat kebebasan"],
    "adiabatic-expansion": ["adiabatik", "adiabatic"],
    "first-law-thermodynamics": ["hukum pertama termodinamika", "first law"],
    "entropy": ["entropi", "entropy"],
    "second-law-thermodynamics": ["hukum kedua", "second law"],
    "heat-engines": ["mesin kalor", "heat engine", "efisiensi"],
    "refrigerators": ["refrigerator", "pendingin", "kulkas"],
    "statistical-entropy": ["statistik entropi", "statistical entropy"],
    "coulombs-law": ["coulomb", "muatan", "charge"],
    "electric-field": ["medan listrik", "electric field"],
    "electric-dipole": ["dipol", "dipole"],
    "continuous-charge-distributions": ["distribusi muatan", "line charge", "charged disk"],
    "electric-flux": ["fluks listrik", "electric flux"],
    "gauss-law-applications": ["hukum gauss", "gauss law"],
    "cylindrical-symmetry": ["silinder", "cylindrical"],
    "planar-symmetry": ["planar", "bidang tak hingga"],
    "spherical-symmetry": ["bola", "spherical symmetry"],
    "equipotential-surfaces": ["ekuipotensial", "equipotential"],
    "potential-from-field": ["potensial dari medan", "potential from field"],
    "point-charge-potential": ["potensial muatan titik", "point charge potential"],
    "potential-energy-systems": ["energi potensial listrik", "potential energy charge"],
    "electric-current": ["arus listrik", "electric current", "current density"],
    "resistance-resistivity": ["hambatan", "resistivity", "resistor", "ohm"],
    "capacitance": ["kapasitansi", "capacitance", "kapasitor", "capacitor"],
    "capacitors-series-parallel": ["kapasitor seri", "kapasitor paralel", "capacitors in"],
    "dielectrics": ["dielektrik", "dielectric"],
    "multiloop-circuits": ["rangkaian", "circuit", "kirchhoff", "loop"],
    "rc-circuits": ["rc circuit", "rangkaian rc"],
    "magnetic-force-charges": ["gaya magnet", "magnetic force", "lorentz"],
    "hall-effect": ["hall effect", "efek hall"],
    "cyclotron-motion": ["siklotron", "cyclotron", "partikel bermuatan"],
    "force-on-wires": ["kawat berarus", "current-carrying wire"],
    "torque-on-loops": ["torsi kumparan", "torque on loop"],
    "biot-savart-ampere": ["biot-savart", "ampere", "solenoid", "toroid"],
    "faradays-law": ["faraday", "induksi elektromagnetik"],
    "lenzs-law": ["lenz"],
    "induced-electric-fields": ["medan induksi", "induced electric"],
    "inductance": ["induktansi", "inductance", "induktor"],
    "self-induction": ["induksi diri", "self-induction"],
    "rl-circuits": ["rl circuit", "rangkaian rl"],
    "mutual-induction": ["induksi mutual", "mutual induction"],
    "lc-oscillations": ["lc oscillation", "osilasi lc"],
    "rlc-circuits": ["rlc", "rangkaian rlc"],
    "alternating-current": ["arus bolak", "alternating current", "ac circuit"],
    "transformers": ["transformator", "transformer"],
    "maxwells-equations": ["maxwell", "arus perpindahan", "displacement current"],
    "em-waves": ["gelombang elektromagnetik", "electromagnetic wave"],
    "poynting-vector": ["poynting", "tekanan radiasi", "radiation pressure"],
    "polarization": ["polarisasi", "polarization"],
    "reflection-refraction": ["refleksi", "refraksi", "reflection", "refraction", "snell"],
    "total-internal-reflection": ["pemantasan total", "total internal reflection"],
    "spherical-mirrors": ["cermin", "mirror"],
    "thin-lenses": ["lensa", "lens"],
    "optical-instruments": ["mikroskop", "teleskop", "microscope", "telescope"],
    "interference": ["interferensi", "interference", "young", "double slit"],
    "thin-film-interference": ["film tipis", "thin film"],
    "single-slit-diffraction": ["difraksi celah tunggal", "single-slit"],
    "circular-aperture-diffraction": ["difraksi aperture", "circular aperture"],
    "diffraction-gratings": ["kisi difraksi", "diffraction grating"],
    "x-ray-diffraction": ["difraksi sinar-x", "x-ray diffraction"],
    "time-dilation": ["dilatasi waktu", "time dilation", "relativitas"],
    "length-contraction": ["kontraksi panjang", "length contraction"],
    "lorentz-transformations": ["lorentz"],
    "relativistic-velocity": ["relativitas kecepatan", "relativistic velocity"],
    "relativistic-momentum-energy": ["energi relativistik", "relativistic energy"],
    "doppler-light": ["doppler cahaya", "doppler light"],
    "photons": ["foton", "photon"],
    "photoelectric-effect": ["fotoelektron", "photoelectric"],
    "compton-scattering": ["compton"],
    "matter-waves": ["gelombang materi", "de broglie", "matter wave"],
    "schrodinger-equation": ["schrodinger", "persamaan schrodinger"],
    "uncertainty-principle": ["ketidakpastian", "uncertainty"],
    "tunneling": ["tunneling", "tembus", "barrier"],
    "electron-traps": ["elektron terperangkap", "electron trap", "potential well"],
    "hydrogen-atom": ["hidrogen", "hydrogen atom", "bohr"],
    "stern-gerlach": ["stern-gerlach"],
    "pauli-exclusion": ["pauli", "exclusion"],
    "periodic-table": ["tabel periodik", "periodic table"],
    "x-rays-elements": ["sinar-x", "x-ray element"],
    "lasers": ["laser"],
    "metal-conduction": ["konduksi logam", "metal conduction"],
    "insulators": ["isolator", "insulator"],
    "semiconductors": ["semikonduktor", "semiconductor"],
    "pn-junction": ["pn junction", "sambungan pn"],
    "transistors-leds": ["transistor", "led"],
    "nuclear-structure": ["inti atom", "nucleus", "nukleus"],
    "radioactive-decay": ["radioaktif", "radioactive decay"],
    "alpha-decay": ["alpha decay", "peluruhan alpha"],
    "beta-decay": ["beta decay", "peluruhan beta"],
    "radiation-dosage": ["dosis radiasi", "radiation dosage"],
    "nuclear-fission": ["fisi", "fission", "reaktor"],
    "nuclear-fusion": ["fusi", "fusion", "termonuklir"],
    "elementary-particles": ["partikel elementer", "elementary particle"],
    "quarks-leptons": ["quark", "lepton"],
    "hadrons": ["hadron"],
    "fundamental-forces": ["gaya fundamental", "messenger particle"],
    "expanding-universe": ["alam semesta mengembang", "expanding universe"],
    "cosmic-background": ["radiasi latar", "cosmic background"],
    "big-bang": ["big bang"],
}


@dataclass
class ProblemTags:
    problem_id: str
    topics: list[str]
    details: list[str]
    disciplines: list[str]
    confidence: float
    method: str
    model: str | None = None

    def as_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "topics": self.topics,
            "details": self.details,
            "disciplines": self.disciplines,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "model": self.model,
        }


# Backward-compatible alias used by older scripts
HallidayTags = ProblemTags


def _text_blob(rec: ProblemRecord) -> str:
    parts = [rec.title, rec.body_md]
    parts.extend(sp.text for sp in rec.subparts)
    return " ".join(parts).lower()


def _allowed_topics(taxonomy: PhysicsTagTaxonomy, coarse_topic: str) -> set[str]:
    priors = COARSE_TOPIC_PRIORS.get(coarse_topic, COARSE_TOPIC_PRIORS["mixed"])
    return set(priors) & taxonomy.valid_topic_ids()


def _score_details(
    text: str,
    taxonomy: PhysicsTagTaxonomy,
    coarse_topic: str,
) -> list[tuple[str, float]]:
    allowed = _allowed_topics(taxonomy, coarse_topic)
    scores: list[tuple[str, float]] = []
    for topic in taxonomy.topics:
        if topic.id not in allowed:
            continue
        for detail in topic.details:
            score = 0.0
            for kw in DETAIL_KEYWORDS.get(detail.id, []):
                if kw in text:
                    score += 1.0
            title_tokens = re.findall(r"[a-z]{4,}", detail.title.lower())
            for tok in title_tokens:
                if tok in text:
                    score += 0.2
            if score > 0:
                scores.append((detail.id, score))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


def _topics_from_details(
    taxonomy: PhysicsTagTaxonomy,
    detail_ids: list[str],
) -> list[str]:
    seen: set[str] = set()
    topics: list[str] = []
    for detail_id in detail_ids:
        topic_id = taxonomy.topic_for_detail(detail_id)
        if topic_id and topic_id not in seen:
            seen.add(topic_id)
            topics.append(topic_id)
    return topics


def _disciplines_from_topics(
    taxonomy: PhysicsTagTaxonomy,
    topic_ids: list[str],
) -> list[str]:
    seen: set[str] = set()
    disciplines: list[str] = []
    for topic_id in topic_ids:
        discipline = taxonomy.discipline_for_topic(topic_id)
        if discipline and discipline not in seen:
            seen.add(discipline)
            disciplines.append(discipline)
    return disciplines


def classify_heuristic(
    rec: ProblemRecord,
    *,
    top_topics: int = 3,
    top_details: int = 5,
) -> ProblemTags:
    taxonomy = load_taxonomy()
    text = _text_blob(rec)
    ranked_details = _score_details(text, taxonomy, rec.topic)

    if not ranked_details:
        allowed = list(_allowed_topics(taxonomy, rec.topic))
        fallback_topic = allowed[0] if allowed else taxonomy.topics[0].id
        topic = taxonomy.topic_by_id(fallback_topic)
        fallback_detail = topic.details[0].id if topic and topic.details else "one-dimensional-motion"
        ranked_details = [(fallback_detail, 0.1)]

    details = [did for did, _ in ranked_details[:top_details]]
    topics = _topics_from_details(taxonomy, details)[:top_topics]
    if not topics and details:
        topic_id = taxonomy.topic_for_detail(details[0])
        if topic_id:
            topics = [topic_id]

    top_score = ranked_details[0][1] if ranked_details else 0.0
    confidence = min(0.95, 0.35 + 0.12 * top_score)
    return ProblemTags(
        problem_id=rec.id,
        topics=topics,
        details=details,
        disciplines=_disciplines_from_topics(taxonomy, topics),
        confidence=confidence,
        method="heuristic",
    )


def _taxonomy_for_prompt(taxonomy: PhysicsTagTaxonomy, coarse_topic: str) -> str:
    allowed = _allowed_topics(taxonomy, coarse_topic)
    lines: list[str] = []
    for topic in taxonomy.topics:
        if topic.id not in allowed:
            continue
        detail_ids = ", ".join(d.id for d in topic.details)
        lines.append(f"{topic.id}: {topic.title}\n  details: {detail_ids}")
    return "\n".join(lines)


def _build_llm_prompt(rec: ProblemRecord, taxonomy: PhysicsTagTaxonomy) -> list[dict[str, str]]:
    system = (
        "You classify Indonesian physics olympiad problems using a two-level tag taxonomy.\n"
        "Return strict JSON only:\n"
        '{"topics": ["kinematics"], "details": ["projectile-motion", "constant-acceleration"], "confidence": 0.85}\n'
        "Rules:\n"
        "- topics: 1-3 kebab-case topic ids from the taxonomy\n"
        "- details: 1-5 kebab-case detail ids (must belong to chosen topics)\n"
        "- confidence: 0-1\n"
        "Pick tags that match the physics concepts in the problem."
    )
    user = {
        "problem_id": rec.id,
        "coarse_topic": rec.topic,
        "title": rec.title,
        "body_md": rec.body_md[:3500],
        "subparts": [sp.model_dump() for sp in rec.subparts[:8]],
        "taxonomy": _taxonomy_for_prompt(taxonomy, rec.topic),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def classify_llm(
    rec: ProblemRecord,
    *,
    model: str | None = None,
    top_topics: int = 3,
    top_details: int = 5,
) -> ProblemTags:
    from src.llm.llm_client import ChatCompletionFailure, chat_completion_json

    if model is None:
        model = _default_tag_model()

    taxonomy = load_taxonomy()
    valid_topics = taxonomy.valid_topic_ids()
    valid_details = taxonomy.valid_detail_ids()

    completion = chat_completion_json(
        messages=_build_llm_prompt(rec, taxonomy),
        model=model,
        max_tokens=300,
        timeout_s=90.0,
    )
    if isinstance(completion, ChatCompletionFailure):
        fallback = classify_heuristic(rec, top_topics=top_topics, top_details=top_details)
        fallback.method = "heuristic_llm_fallback"
        return fallback

    try:
        data = json.loads(completion.content)
    except json.JSONDecodeError:
        return classify_heuristic(rec, top_topics=top_topics, top_details=top_details)

    topics = [t for t in data.get("topics", []) if t in valid_topics][:top_topics]
    details = [d for d in data.get("details", []) if d in valid_details][:top_details]

    # Ensure details belong to selected topics when possible
    if topics:
        allowed_details = {
            d.id for t in topics for d in (taxonomy.topic_by_id(t).details if taxonomy.topic_by_id(t) else ())
        }
        details = [d for d in details if d in allowed_details] or details

    if not details and topics:
        topic = taxonomy.topic_by_id(topics[0])
        if topic and topic.details:
            details = [topic.details[0].id]
    if not topics and details:
        topic_id = taxonomy.topic_for_detail(details[0])
        if topic_id:
            topics = [topic_id]

    confidence = float(data.get("confidence", 0.7))
    confidence = max(0.0, min(1.0, confidence))

    if not topics and not details:
        return classify_heuristic(rec, top_topics=top_topics, top_details=top_details)

    return ProblemTags(
        problem_id=rec.id,
        topics=topics,
        details=details,
        disciplines=_disciplines_from_topics(taxonomy, topics),
        confidence=confidence,
        method="llm",
        model=model,
    )


def classify_problem(
    rec: ProblemRecord,
    *,
    use_llm: bool = False,
    model: str | None = None,
) -> ProblemTags:
    if use_llm:
        return classify_llm(rec, model=model)
    return classify_heuristic(rec)
