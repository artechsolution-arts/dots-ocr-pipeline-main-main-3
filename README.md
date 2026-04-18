# Virchow Ingestion Pipeline

On-prem document ingestion backend for enterprise RAG. Consumes PDFs, extracts text with a VLM OCR, pulls structured entities with a local LLM, chunks + embeds with mxbai, and stores everything in PostgreSQL + pgvector. Built for macOS on Apple Silicon (M-series) so the VLM and embedder run natively on the Metal GPU.

This repository is the **backend ingestion service only**. The retrieval / chat UI is a separate project that reads from the shared Postgres.


## Prerequisites

| Requirement | Why |
|---|---|
| macOS 13+ or Linux with Docker | Redis + RabbitMQ run in Docker; everything else is native |
| **Python 3.10 – 3.12** | PyTorch / transformers don't yet support 3.13+ (see `start.sh`) |
| **Docker Desktop** | For Redis + RabbitMQ (NOT for the OCR worker — it needs Metal/CUDA) |
| **Ollama** with `qwen2.5:14b-instruct` pulled | Entity extraction (AI-5). Run `ollama pull qwen2.5:14b-instruct` |
| **PostgreSQL 15+ with pgvector** | Vector store. Can be local Docker, managed cloud, or a separate host |
| **SeaweedFS** (or any S3-compatible store) | Raw PDF + processed markdown. Optional — falls back to local disk |
| ~15 GB free disk | DotsOCR weights (~7 GB) + venv + model cache |

---

## Quick start

```bash
# 1. Clone + enter
git clone <this-repo> virchow-ingestion && cd virchow-ingestion

# 2. Configure
cp "RAG_complete_Backend_W 2/Rag_full_pipeline/.env.example" \
   "RAG_complete_Backend_W 2/Rag_full_pipeline/.env"
# Edit .env — at minimum set PG_PASSWORD

# 3. Start (handles venv, deps, weights download, docker services)
chmod +x start.sh stop.sh
./start.sh
```

First run takes ~10 minutes — it creates the venv, installs deps, and downloads 7 GB of DotsOCR weights from HuggingFace. Subsequent runs start in ~30 seconds.

You'll see:

```
╔════════════════════════════════════════════════════╗
║   Pipeline is running!                             ║
║  API        : http://localhost:8000                ║
║  Health     : http://localhost:8000/health         ║
║  RabbitMQ   : http://localhost:15672 (guest/guest) ║
╚════════════════════════════════════════════════════╝
```

Test with:

```bash
curl http://localhost:8000/health
curl -F 'files=@test.pdf' http://localhost:8000/ingest
```

Stop with `./stop.sh` (or `Ctrl+C` in the terminal running `start.sh` — the trap calls `stop.sh`).


## Project structure

```
.
├── start.sh / stop.sh                # single-command lifecycle
├── docker-compose.yml                # Redis + RabbitMQ
├── weights/DotsOCR/                  # downloaded on first run
├── dots_ocr/                         # DotsOCR VLM (vendored)
├── logs/                             # api.log, worker.log
├── INGESTION_PIPELINE_SPEC.md        # design spec
└── RAG_complete_Backend_W 2/Rag_full_pipeline/
    ├── .env.example                  # copy to .env, fill in
    ├── main.py                       # FastAPI + stage pipeline lifespan
    ├── src/
    │   ├── api/routes.py             # /ingest, /health, /metrics, /admin/*
    │   ├── config.py                 # env-driven RAGConfig
    │   ├── database/
    │   │   ├── postgres_db.py        # schema + RBACManager + vector_search
    │   │   ├── redis_db.py
    │   │   └── rabbitmq_broker.py
    │   ├── ingestion/
    │   │   ├── pipeline/stage_pipeline.py    # the 19-thread stage pipeline
    │   │   ├── chunking/chunker.py           # markdown-aware, token-limited
    │   │   ├── embedding/embedder.py         # mxbai on MPS
    │   │   ├── ocr/                          # DotsOCR wrappers
    │   │   ├── metadata/                     # filename, entities, tables, quality, enrichment
    │   │   └── backfill.py                   # re-process existing DB
    │   ├── services/rag_pipeline.py          # top-level pipeline coordinator
    │   ├── storage/                          # SeaweedFS client
    │   └── worker/pool.py                    # RabbitMQ feeders
    └── tests/test_metadata.py        # unit tests for pure-logic modules
```

