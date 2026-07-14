"""General-document parsing (Marker + pdftotext hybrid) for non-exam documents.

Unlike :mod:`src.bronze.hybrid_bronze` (which segments typed exam PDFs into
numbered problems), this module parses *arbitrary* documents — research papers,
notes, articles — into clean Markdown. It keeps the same hybrid philosophy used
for the problem corpus: run Marker for layout/structure/figures, then reconcile
the body text against the high-fidelity pdftotext text layer.
"""
from src.docs.general_doc import (
    GeneralDocResult,
    clean_general_markdown,
    convert_general_doc,
    marker_degraded,
    merge_general_hybrid,
)

__all__ = (
    "GeneralDocResult",
    "clean_general_markdown",
    "convert_general_doc",
    "marker_degraded",
    "merge_general_hybrid",
)
