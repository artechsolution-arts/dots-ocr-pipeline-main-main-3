"""Backfill tool for existing documents — AI-9.

Run from the repo root with the same env as the pipeline (Postgres
creds). Each pass is independent, idempotent, and skips documents that
already have the relevant data so you can re-run safely.

Usage
-----
    python -m src.ingestion.backfill --filename           # instant
    python -m src.ingestion.backfill --quality            # instant
    python -m src.ingestion.backfill --extract            # Ollama per doc
    python -m src.ingestion.backfill --refs               # post-extract pass
    python -m src.ingestion.backfill --enrich-chunks      # heavy — re-embeds
    python -m src.ingestion.backfill --all                # filename+quality+extract+refs
    python -m src.ingestion.backfill --limit 20 --extract # smoke test

``--enrich-chunks`` is intentionally not in ``--all``: it rewrites every
existing chunk's text with the AI-7 header and re-embeds, which is
expensive. Opt in explicitly when you want it.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import List, Tuple

from src.config import cfg
from src.database.postgres_db import RBACManager, get_pg_pool
from src.ingestion.metadata.enrichment import build_chunk_header
from src.ingestion.metadata.entity_extractor import extract_entities
from src.ingestion.metadata.filename_parser import parse_filename_metadata
from src.ingestion.metadata.quality import compute_quality_score
from src.ingestion.metadata.reference_resolver import resolve_references

logger = logging.getLogger("backfill")

# Cap the text we feed the LLM so a single bloated chunk can't run the
# extractor to its full context limit.
MAX_TEXT_CHARS = 8000


# ─────────────────────────────── Passes ──────────────────────────────────

def pass_filename(rbac: RBACManager, limit: int) -> int:
    """Parse filename → update documents.{doc_month, doc_unit, doc_type,
    fiscal_year, serial_no}. Skips docs that already have doc_type set."""
    conn = rbac._get_conn()
    updated = 0
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, file_name FROM documents "
            "WHERE doc_type IS NULL AND fiscal_year IS NULL "
            "ORDER BY file_name "
            + (f"LIMIT {int(limit)}" if limit else "")
        )
        rows = cur.fetchall()
        for doc_id, file_name in rows:
            meta = parse_filename_metadata(file_name)
            if not meta:
                continue
            cur.execute(
                "UPDATE documents SET doc_month=%s, doc_unit=%s, "
                "doc_type=%s, fiscal_year=%s, serial_no=%s WHERE id=%s",
                (meta.get("doc_month"), meta.get("doc_unit"),
                 meta.get("doc_type"), meta.get("fiscal_year"),
                 meta.get("serial_no"), doc_id),
            )
            updated += 1
        cur.close()
        logger.info("[filename] scanned=%d updated=%d", len(rows), updated)
        return updated
    finally:
        rbac._put_conn(conn)


def pass_quality(rbac: RBACManager, limit: int) -> int:
    """Compute per-chunk quality_score. Also rolls up a doc-level average
    into documents.ocr_quality if that column is still NULL."""
    conn = rbac._get_conn()
    scored = 0
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, chunk_text FROM chunks "
            "WHERE quality_score IS NULL OR quality_score = 1.0 "
            + (f"LIMIT {int(limit)}" if limit else "")
        )
        for chunk_id, text in cur.fetchall():
            raw = text.split("\n\n", 1)[-1] if text.startswith("[") else text
            score = compute_quality_score(raw)
            cur.execute("UPDATE chunks SET quality_score=%s WHERE id=%s",
                        (score, chunk_id))
            scored += 1

        # Roll up doc-level score where missing.
        cur.execute(
            "UPDATE documents d SET ocr_quality = sub.avg "
            "FROM (SELECT document_id, AVG(quality_score)::float AS avg "
            "      FROM chunks GROUP BY document_id) sub "
            "WHERE sub.document_id = d.id AND d.ocr_quality IS NULL"
        )
        cur.close()
        logger.info("[quality] chunks_scored=%d", scored)
        return scored
    finally:
        rbac._put_conn(conn)


def _fetch_doc_text(rbac: RBACManager, doc_id: str) -> str:
    conn = rbac._get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_text FROM chunks WHERE document_id=%s "
            "ORDER BY chunk_index LIMIT 20", (doc_id,)
        )
        pieces: List[str] = []
        total = 0
        for (t,) in cur.fetchall():
            if t.startswith("["):           # strip any enrichment header
                t = t.split("\n\n", 1)[-1] if "\n\n" in t else t
            pieces.append(t)
            total += len(t)
            if total >= MAX_TEXT_CHARS:
                break
        cur.close()
        return "\n\n".join(pieces)[:MAX_TEXT_CHARS]
    finally:
        rbac._put_conn(conn)


def pass_extract(rbac: RBACManager, limit: int) -> Tuple[int, int]:
    """Run LLM entity extraction on each document that lacks party_name.
    Also records any ref_doc_number in document_references for later
    resolution. Returns ``(scanned, updated)``."""
    conn = rbac._get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, file_name, doc_type FROM documents "
            "WHERE party_name IS NULL AND total_amount IS NULL "
            "ORDER BY file_name "
            + (f"LIMIT {int(limit)}" if limit else "")
        )
        docs = cur.fetchall()
        cur.close()
    finally:
        rbac._put_conn(conn)

    updated = 0
    from src.ingestion.pipeline.stage_pipeline import _guess_ref_type

    for i, (doc_id, file_name, doc_type) in enumerate(docs, 1):
        text = _fetch_doc_text(rbac, str(doc_id))
        if not text.strip():
            continue
        t0 = time.time()
        result = extract_entities(text)
        dt = time.time() - t0
        if not result:
            logger.info("  [%d/%d] %s — no fields extracted (%.1fs)",
                         i, len(docs), file_name, dt)
            continue
        try:
            rbac.update_document_extraction(str(doc_id), result)
            updated += 1
        except Exception as e:
            logger.warning("  [%d/%d] %s — persist failed: %s",
                           i, len(docs), file_name, e)
            continue
        if result.get("ref_doc_number"):
            try:
                rbac.add_document_reference(
                    str(doc_id), result["ref_doc_number"],
                    _guess_ref_type(doc_type))
            except Exception as e:
                logger.warning("  [%d/%d] %s — ref insert failed: %s",
                               i, len(docs), file_name, e)
        logger.info("  [%d/%d] %s — %d fields (%.1fs)",
                    i, len(docs), file_name, len(result), dt)

    logger.info("[extract] scanned=%d updated=%d", len(docs), updated)
    return len(docs), updated


def pass_refs(rbac: RBACManager) -> Tuple[int, int]:
    return resolve_references(rbac)


def pass_enrich_chunks(rbac: RBACManager, limit: int) -> int:
    """Rewrite chunk_text with AI-7 header and re-embed. Heavy — runs the
    embedder on every updated chunk. Skips chunks whose text already
    starts with ``[`` (they've already been enriched)."""
    from src.ingestion.embedding.embedder import MxbaiEmbedder
    embedder = MxbaiEmbedder()

    conn = rbac._get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT d.id, d.file_name, d.department_id, d.doc_month, d.doc_unit, "
            "       d.doc_type, d.fiscal_year, d.party_name, d.total_amount "
            "FROM documents d "
            + (f"ORDER BY d.file_name LIMIT {int(limit)}" if limit else "ORDER BY d.file_name")
        )
        docs = cur.fetchall()
        cur.close()
    finally:
        rbac._put_conn(conn)

    total_rewritten = 0
    for doc_id, file_name, dept_id, dm, du, dt, fy, party, total in docs:
        fm = {"doc_month": dm, "doc_unit": du, "doc_type": dt, "fiscal_year": fy}
        fm = {k: v for k, v in fm.items() if v}
        ex = {}
        if party:
            ex["party_name"] = party
        if total is not None:
            ex["total_amount"] = float(total)
        header = build_chunk_header(file_name, fm, ex)

        conn = rbac._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, chunk_text FROM chunks WHERE document_id=%s "
                "ORDER BY chunk_index", (str(doc_id),)
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            rbac._put_conn(conn)

        # A chunk is considered already enriched only if its first line is
        # literally ``[<file_name>...]``. This avoids the trap where OCR
        # output starts with raw layout JSON like ``[{"bbox": ...}]`` and
        # gets mistaken for an enrichment header.
        fname_header_prefix = f"[{file_name}"
        to_update = []
        for chunk_id, text in rows:
            first_line = text.split("\n", 1)[0]
            if (first_line.startswith(fname_header_prefix)
                    and first_line.endswith("]")):
                continue
            to_update.append((chunk_id, f"{header}\n\n{text}"))

        if not to_update:
            continue

        vectors = embedder.embed_batch([t for _, t in to_update])
        conn = rbac._get_conn()
        try:
            cur = conn.cursor()
            for (chunk_id, new_text), vec in zip(to_update, vectors):
                cur.execute("UPDATE chunks SET chunk_text=%s WHERE id=%s",
                            (new_text, chunk_id))
                vec_str = RBACManager._vec_str(vec)
                cur.execute(
                    "UPDATE embeddings SET embedding=%s::vector WHERE chunk_id=%s",
                    (vec_str, chunk_id),
                )
            cur.close()
        finally:
            rbac._put_conn(conn)

        total_rewritten += len(to_update)
        logger.info("  %s — rewrote %d chunks", file_name, len(to_update))

    logger.info("[enrich] chunks_rewritten=%d", total_rewritten)
    return total_rewritten


# ─────────────────────────────── Entry ───────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill existing documents "
                                             "per INGESTION_PIPELINE_SPEC.md.")
    ap.add_argument("--filename", action="store_true",
                    help="Parse filenames → documents metadata columns")
    ap.add_argument("--quality", action="store_true",
                    help="Compute per-chunk quality_score + doc ocr_quality")
    ap.add_argument("--extract", action="store_true",
                    help="Run LLM entity extraction on docs missing party_name")
    ap.add_argument("--refs", action="store_true",
                    help="Resolve document_references.ref_doc_id")
    ap.add_argument("--enrich-chunks", action="store_true",
                    help="Rewrite chunk_text with AI-7 header + re-embed "
                         "(slow, writes to embeddings)")
    ap.add_argument("--all", action="store_true",
                    help="filename + quality + extract + refs  "
                         "(NOT enrich-chunks — opt in separately)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap the number of rows per pass (0 = unlimited)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not any([args.filename, args.quality, args.extract, args.refs,
                args.enrich_chunks, args.all]):
        ap.error("pick at least one pass, or --all")

    pool = get_pg_pool(minconn=1, maxconn=4)
    rbac = RBACManager(pool)

    if args.all or args.filename:
        logger.info("──── pass: filename ────")
        pass_filename(rbac, args.limit)
    if args.all or args.quality:
        logger.info("──── pass: quality ────")
        pass_quality(rbac, args.limit)
    if args.all or args.extract:
        logger.info("──── pass: extract (Ollama: %s) ────", cfg.llm_model)
        pass_extract(rbac, args.limit)
    if args.all or args.refs:
        logger.info("──── pass: refs ────")
        pass_refs(rbac)
    if args.enrich_chunks:
        logger.info("──── pass: enrich-chunks (re-embed) ────")
        pass_enrich_chunks(rbac, args.limit)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
