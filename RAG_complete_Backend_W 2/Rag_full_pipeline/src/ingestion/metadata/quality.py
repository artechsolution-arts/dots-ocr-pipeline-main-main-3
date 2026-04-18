"""OCR quality scoring — AI-8."""
from __future__ import annotations

import re

_CLEAN_RE   = re.compile(r"[a-zA-Z0-9₹.,;:\-/() ]")
_GARBAGE_RE = re.compile(r"[^\x00-\x7F]")
_WORD_RE    = re.compile(r"^[a-zA-Z]{3,}$")


def compute_quality_score(text: str) -> float:
    """Heuristic OCR quality score in [0, 1].

    Retrieval behaviour (enforced downstream, not here):
      < 0.3   → exclude
      0.3–0.6 → deprioritize
      > 0.6   → normal
    """
    if not text or len(text) < 10:
        return 0.0

    total = len(text)
    alpha_ratio    = len(_CLEAN_RE.findall(text)) / total
    garbage_ratio  = len(_GARBAGE_RE.findall(text)) / total
    garbage_penalty = min(garbage_ratio * 2, 0.5)

    words = text.split()
    real_words = [w for w in words if _WORD_RE.match(w)]
    word_ratio = len(real_words) / max(len(words), 1)

    score = (alpha_ratio * 0.4) + (word_ratio * 0.4) - garbage_penalty + 0.2
    return round(max(0.0, min(1.0, score)), 3)
