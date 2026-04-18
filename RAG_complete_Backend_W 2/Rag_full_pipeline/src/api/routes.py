"""
Ingestion API Routes
====================
Single-purpose API for the on-prem RAG ingestion pipeline.

Endpoints
---------
POST /ingest                          Upload PDFs → OCR → Chunk → Embed → pgvector + SeaweedFS
GET  /ingest/progress/{session_id}    SSE stream of per-file ingestion progress
GET  /health                          Service health (postgres, redis, rabbitmq, seaweedfs)
GET  /storage/jobs/{job_id}/files     List SeaweedFS objects for a job
DELETE /storage/jobs/{job_id}/files   Remove SeaweedFS artefacts for a job
"""

import json
import time
import uuid
import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import (
    FALLBACK_DEPT_ID, FALLBACK_USER_ID,
    MQ_QUEUE_DEAD, MQ_QUEUE_LARGE, MQ_QUEUE_NORMAL, MQ_QUEUE_PRIORITY,
    UPLOAD_DIR, cfg,
)
from src.models.schemas import BatchSession, FileProgress, JobPayload

logger = logging.getLogger(__name__)


def _safe_basename(name: str) -> str:
    """Strip any directory component from an untrusted filename. The
    browser shouldn't be able to write outside ``UPLOAD_DIR`` by
    sending ``../../evil.pdf`` — we collapse the path to its basename
    and reject empty results."""
    base = os.path.basename(name or "").replace("\\", "/").split("/")[-1]
    if not base or base in {".", ".."}:
        raise HTTPException(400, "invalid filename")
    return base


def create_router(rsm, ids, pipeline, mq_conn):
    router = APIRouter()

    # ── Health Check ──────────────────────────────────────────────────────────

    @router.get("/health")
    async def health():
        """
        Returns the live status of all infrastructure components.
        Your query system can poll this before sending ingestion requests.
        """
        seaweedfs_ok = False
        if pipeline.storage:
            try:
                seaweedfs_ok = await pipeline.storage.health() is not None
            except Exception:
                seaweedfs_ok = False

        status = {
            "status": "ok",
            "postgres": True,   # If we reached this point postgres is up
            "redis":    bool(rsm and rsm.ping()),
            "rabbitmq": bool(mq_conn and mq_conn.is_open),
            "seaweedfs": seaweedfs_ok,
        }
        # Overall ok only if core services are up
        if not status["redis"] or not status["rabbitmq"]:
            status["status"] = "degraded"
        return JSONResponse(status)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    @router.post("/ingest")
    async def ingest(
        files: List[UploadFile] = File(...),
        dept_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
    ):
        """
        Upload one or more PDF files for ingestion.

        The pipeline runs asynchronously:
          1. DotsOCR  — VLM layout detection + text extraction
          2. Chunking — Markdown-aware, token-limited chunks
          3. Embedding — mxbai-embed-large-v1 (1024-dim)
          4. Storage  — chunks + vectors → PostgreSQL/pgvector
                        raw PDF + markdown → SeaweedFS

        Parameters
        ----------
        files    : PDF file(s) to ingest
        dept_id  : Department UUID to scope vectors under (optional).
                   Defaults to the system default department.
                   Your query system should filter by the same dept_id.
        user_id  : Uploader UUID (optional). Defaults to system user.

        Returns
        -------
        {
          "session_id": "uuid",          -- poll /ingest/progress/{session_id}
          "dept_id": "uuid",             -- use this in your query system for vector filtering
          "files": [
            {"file_id": "uuid", "filename": "doc.pdf", "size_kb": 123.4}
          ]
        }
        """
        if not rsm or not rsm.ping():
            raise HTTPException(503, "Redis is offline — cannot accept ingestion jobs")

        # Resolve dept/user — fall back to seeded system defaults from config.
        resolved_dept = dept_id or ids.get("dept_default") or FALLBACK_DEPT_ID
        resolved_user = user_id or ids.get("user_default") or FALLBACK_USER_ID
        session_id = str(uuid.uuid4())
        session = BatchSession(
            session_id=session_id,
            total=len(files),
            user_id=str(resolved_user),
            dept_id=str(resolved_dept),
            upload_type="user",
        )
        rsm.create_session(session)

        ingested_files = []

        for f in files:
            safe_name = _safe_basename(f.filename)
            if not safe_name.lower().endswith(".pdf"):
                raise HTTPException(400, f"'{safe_name}' is not a PDF")
            # Assign this now so downstream refs (logs, DB rows, job payload)
            # all use the sanitized name rather than the user-supplied one.
            f_filename_safe = safe_name

            contents = await f.read()

            # Magic byte check — reject non-PDF content regardless of filename
            if not contents.startswith(b"%PDF-"):
                raise HTTPException(
                    400,
                    f"'{safe_name}' does not appear to be a valid PDF file",
                )

            file_id = str(uuid.uuid4())
            fpath = UPLOAD_DIR / f"{file_id}_{safe_name}"
            fpath.write_bytes(contents)

            # Register upload in PostgreSQL (FK integrity before publishing job)
            upload_id = None
            try:
                upload_id = pipeline.rbac.register_user_upload(
                    user_id=str(resolved_user),
                    dept_id=str(resolved_dept),
                    file_name=f_filename_safe,
                    file_path=str(fpath),
                    file_size_bytes=len(contents),
                    chat_id=None,
                    upload_scope="dept",
                )
            except Exception as e:
                logger.warning(
                    "Upload registration failed — file=%s file_id=%s: %s",
                    f_filename_safe, file_id, e,
                )

            fp = FileProgress(
                file_id=file_id,
                session_id=session_id,
                filename=f_filename_safe,
                size_kb=len(contents) / 1024,
                started_at=time.time(),
            )
            rsm.register_file(session_id, fp)

            job = JobPayload(
                session_id=session_id,
                file_id=file_id,
                filename=f_filename_safe,
                file_path=str(fpath),
                file_size_kb=len(contents) / 1024,
                user_id=str(resolved_user),
                dept_id=str(resolved_dept),
                upload_type="user",
                upload_id=upload_id,
            )

            from src.database.rabbitmq_broker import publish_job
            publish_job(job)

            ingested_files.append({
                "file_id":  file_id,
                "filename": f_filename_safe,
                "size_kb":  round(len(contents) / 1024, 1),
            })

            logger.info(
                "Job queued — file=%s file_id=%s session=%s dept=%s",
                f_filename_safe, file_id, session_id, resolved_dept,
            )

        return JSONResponse({
            "session_id": session_id,
            "dept_id":    str(resolved_dept),
            "files":      ingested_files,
        })

    # ── Progress (SSE) ────────────────────────────────────────────────────────

    @router.get("/ingest/progress/{session_id}")
    async def ingest_progress(session_id: str):
        """
        Server-Sent Events stream for real-time ingestion progress.

        Each event has the shape:
          data: {"type": "file_progress", "data": {
            "file_id": "uuid",
            "filename": "doc.pdf",
            "stage": "ocr" | "chunking" | "embedding" | "storing" | "done" | "error",
            "pct": 0-100,
            "chunks": 42,
            "doc_id": "uuid",    -- available once stored; use for PG queries
            "error": null
          }}

        The stream closes when all files in the session reach a terminal stage
        (done / error / skipped).
        """
        if not rsm or not rsm.ping():
            raise HTTPException(503, "Redis is offline")

        async def _event_stream():
            # Immediately emit current state for any already-progressed files
            summary = rsm.session_summary(session_id)
            if summary:
                for f in summary.get("files", []):
                    yield f"data: {json.dumps({'type': 'file_progress', 'data': f})}\n\n"

            # stop_event is set when the client disconnects, signalling the
            # Redis subscription thread to exit and release its connection.
            stop_event = threading.Event()
            q    = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _subscribe():
                try:
                    for event in rsm.subscribe_session(session_id,
                                                       stop_event=stop_event):
                        loop.call_soon_threadsafe(q.put_nowait, event)
                except Exception as e:
                    logger.warning("SSE subscribe error: %s", e)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            thread = threading.Thread(target=_subscribe, daemon=True)
            thread.start()

            try:
                while True:
                    event = await q.get()
                    if event is None:
                        break
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                # Client disconnected or session complete — release Redis connection
                stop_event.set()

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # ── SeaweedFS Storage Routes ───────────────────────────────────────────────

    @router.get("/storage/health")
    async def storage_health():
        if not pipeline.storage:
            return JSONResponse({"seaweedfs": "not configured"})
        return await pipeline.storage.health()

    @router.get("/storage/jobs/{job_id}/files")
    async def list_job_files(job_id: str):
        """List all SeaweedFS objects for a given job (raw PDF, extracted markdown)."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        return await pipeline.storage.list_job_files(job_id)

    @router.delete("/storage/jobs/{job_id}/files")
    async def delete_job_files(job_id: str):
        """Remove all SeaweedFS artefacts for a completed or failed job."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        deleted = await pipeline.storage.delete_job_artefacts(job_id)
        return {"deleted_count": deleted}

    @router.get("/storage/jobs/{job_id}/pdf-url")
    async def get_pdf_url(job_id: str, filename: str):
        """Return the SeaweedFS filer URL for the raw PDF of a job."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        return {"url": pipeline.storage.pdf_url(job_id, filename)}

    # ── Metrics & Admin ───────────────────────────────────────────────────────

    @router.get("/metrics")
    async def metrics():
        """Lightweight JSON snapshot for operators/dashboards.
        Queue depths from RabbitMQ, postgres doc/chunk counts, and stage
        pipeline internals when available."""
        metrics: dict = {"queues": {}, "db": {}, "stage": {}}

        # RabbitMQ queue depths — passive queue_declare returns the count
        try:
            import pika
            if mq_conn and mq_conn.is_open:
                ch = mq_conn.channel()
                for q in (MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL,
                          MQ_QUEUE_LARGE, MQ_QUEUE_DEAD):
                    try:
                        res = ch.queue_declare(q, passive=True)
                        metrics["queues"][q] = res.method.message_count
                    except Exception as e:
                        metrics["queues"][q] = f"error: {e}"
                ch.close()
        except Exception as e:
            metrics["queues"]["_error"] = str(e)

        # Postgres counts — cheap aggregate over indexed tables
        try:
            if pipeline.rbac:
                conn = pipeline.rbac._get_conn()
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT "
                                "(SELECT COUNT(*) FROM documents),"
                                "(SELECT COUNT(*) FROM chunks),"
                                "(SELECT COUNT(*) FROM embeddings),"
                                "(SELECT COUNT(*) FROM document_line_items),"
                                "(SELECT COUNT(*) FROM document_references)")
                    docs, chunks, embs, li, refs = cur.fetchone()
                    metrics["db"] = {
                        "documents":  docs,
                        "chunks":     chunks,
                        "embeddings": embs,
                        "line_items": li,
                        "references": refs,
                    }
                    cur.close()
                finally:
                    pipeline.rbac._put_conn(conn)
        except Exception as e:
            metrics["db"]["_error"] = str(e)

        # Stage pipeline internals — best-effort; private attrs, fine for ops
        try:
            sp = getattr(pipeline, "stage_pipeline", None) or getattr(
                pipeline, "_pipeline", None)
            if sp is not None:
                for q_name in ("_page_q", "_markdown_q", "_assembled_q",
                               "_chunk_q", "_store_q"):
                    q = getattr(sp, q_name, None)
                    if q is not None and hasattr(q, "qsize"):
                        metrics["stage"][q_name.lstrip("_")] = q.qsize()
        except Exception as e:
            metrics["stage"]["_error"] = str(e)

        return JSONResponse(metrics)

    @router.get("/admin/dlq")
    async def dlq_inspect(limit: int = 20):
        """Peek at the first ``limit`` messages on the dead-letter queue
        without consuming them. Handy for triaging failures."""
        if not (mq_conn and mq_conn.is_open):
            raise HTTPException(503, "RabbitMQ offline")
        messages = []
        ch = mq_conn.channel()
        try:
            for _ in range(max(1, min(limit, 100))):
                method, props, body = ch.basic_get(MQ_QUEUE_DEAD,
                                                    auto_ack=False)
                if method is None:
                    break
                try:
                    payload = json.loads(body.decode())
                except Exception:
                    payload = {"raw_body_len": len(body)}
                messages.append({
                    "delivery_tag": method.delivery_tag,
                    "headers":      dict(props.headers or {}),
                    "payload":      payload,
                })
                ch.basic_nack(method.delivery_tag, requeue=True)
        finally:
            ch.close()
        return {"count": len(messages), "messages": messages}

    @router.post("/admin/dlq/purge")
    async def dlq_purge():
        """Discard every message on the dead-letter queue. Irreversible —
        only call after triage."""
        if not (mq_conn and mq_conn.is_open):
            raise HTTPException(503, "RabbitMQ offline")
        ch = mq_conn.channel()
        try:
            res = ch.queue_purge(MQ_QUEUE_DEAD)
            purged = getattr(res.method, "message_count", None)
        finally:
            ch.close()
        return {"purged": purged}

    return router
