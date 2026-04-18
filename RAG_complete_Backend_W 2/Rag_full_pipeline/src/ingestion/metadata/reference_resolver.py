"""Cross-document reference resolution — AI-3 second pass.

At ingestion time we record whatever ``ref_doc_number`` the LLM pulled
off an invoice / GRN into ``document_references(source_doc_id,
ref_doc_number, ref_type)`` with ``ref_doc_id`` left NULL. This module
runs separately to fuzz-match those strings against
``documents.doc_number`` and backfill the FK, turning "invoice mentions
PO" strings into navigable graph edges.

Safe to run repeatedly — only rows with NULL ``ref_doc_id`` are
considered, and each match is idempotent.
"""
from __future__ import annotations

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


def _key(s: str) -> str:
    """Canonical match key: uppercase, drop all non-alphanumerics."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def resolve_references(rbac) -> Tuple[int, int]:
    """For every unresolved reference, try to match against existing
    ``documents.doc_number`` (either exactly or via a normalized key).
    Returns ``(scanned, resolved)``."""
    conn = rbac._get_conn()
    scanned = resolved = 0
    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT id, doc_number FROM documents "
            "WHERE doc_number IS NOT NULL AND doc_number <> ''"
        )
        by_exact = {}
        by_key = {}
        for doc_id, doc_number in cur.fetchall():
            by_exact.setdefault(doc_number.strip(), str(doc_id))
            by_key.setdefault(_key(doc_number), str(doc_id))

        cur.execute(
            "SELECT id, source_doc_id, ref_doc_number "
            "FROM document_references WHERE ref_doc_id IS NULL"
        )
        pending = cur.fetchall()
        scanned = len(pending)

        for ref_id, source_id, ref_number in pending:
            target = by_exact.get((ref_number or "").strip())
            if not target:
                target = by_key.get(_key(ref_number))
            if not target or target == str(source_id):
                continue
            cur.execute(
                "UPDATE document_references SET ref_doc_id=%s WHERE id=%s",
                (target, ref_id),
            )
            resolved += 1

        cur.close()
        logger.info("[Refs] scanned=%d resolved=%d", scanned, resolved)
        return scanned, resolved
    finally:
        rbac._put_conn(conn)
