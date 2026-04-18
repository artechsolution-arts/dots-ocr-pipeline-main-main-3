"""Chunk enrichment header — AI-7.

Builds a single-line header that pins every chunk to its source document
so embeddings capture document context, not just the raw paragraph.
Fields still unknown in Phase 1 (party_name, total_amount) are simply
dropped; the header re-gains them after AI-5 extraction runs.
"""
from __future__ import annotations

from typing import Mapping, Optional


def build_chunk_header(
    file_name: str,
    fname_meta: Optional[Mapping[str, str]] = None,
    extraction: Optional[Mapping[str, object]] = None,
) -> str:
    """Return a ``[...]`` header string. Caller prepends it to chunk text.

    ``fname_meta``  — dict from ``parse_filename_metadata``.
    ``extraction``  — dict from the LLM entity extractor (Phase 2);
                      may include ``party_name`` and ``total_amount``.
    """
    fm = fname_meta or {}
    ex = extraction or {}

    parts: list[str] = [file_name]

    if fm.get("doc_type"):
        parts.append(fm["doc_type"])

    month_fy = " ".join(p for p in (fm.get("doc_month"), fm.get("fiscal_year")) if p)
    if month_fy:
        parts.append(month_fy)

    if fm.get("doc_unit"):
        parts.append(fm["doc_unit"])

    party = ex.get("party_name")
    if party:
        parts.append(f"Vendor: {party}")

    total = ex.get("total_amount")
    if total is not None:
        try:
            parts.append(f"Total: ₹{float(total):,.0f}")
        except (TypeError, ValueError):
            pass

    return "[" + " | ".join(parts) + "]"
