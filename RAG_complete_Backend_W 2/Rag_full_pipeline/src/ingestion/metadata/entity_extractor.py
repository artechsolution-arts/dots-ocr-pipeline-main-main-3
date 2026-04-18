"""LLM entity extraction — AI-5.

Runs once per document at ingestion time. Uses a local Ollama model
(default ``qwen2.5:14b-instruct``) to pull financial entities from the
first N chars of the assembled document text and return a strictly
validated dict.

Design rules
------------
* **Fail-open.** Any network/parse/validation error returns ``{}`` and
  logs a warning — ingestion never fails because of the LLM.
* **Timeout-bound.** One HTTP call with a hard timeout (default 60s).
* **Per-field validation.** A field that fails validation is dropped,
  not stored. Bad data is worse than no data.
* **JSON mode.** Uses Ollama's ``format: "json"`` so we never have to
  hunt for a JSON block inside prose.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_EXTRACTION_MODEL", "qwen2.5:14b-instruct")
# Keep the timeout tight enough that a hung model doesn't stall a chunk
# worker for a full minute. Observed p95 of a successful extraction on
# qwen2.5:14b-instruct is ~6s; 25s gives ~4x headroom.
TIMEOUT_SEC  = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "25"))
MAX_TEXT_CHARS = int(os.environ.get("EXTRACTION_MAX_CHARS", "2000"))

_GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")

_SYSTEM_PROMPT = (
    "You are a structured-data extractor for Indian finance and procurement "
    "documents (Purchase Orders, Invoices, GRNs, Delivery Notes, Journal "
    "Vouchers). Reply with valid JSON only — no prose, no code fences. Every "
    "field you cannot confidently extract must be null."
)

_USER_TEMPLATE = """Extract the following fields from the document below.

Fields (use these exact keys):
  party_name       — vendor or customer name (string)
  party_gstin      — 15-char Indian GSTIN (string)
  doc_date         — document date as YYYY-MM-DD
  doc_number       — PO / invoice / GRN number (string)
  total_amount     — final payable incl. tax (number, no currency symbol)
  tax_amount       — total GST (number)
  net_amount       — amount before tax (number)
  payment_terms    — e.g. "Net 30 days" (string)
  ref_doc_number   — referenced PO / invoice number, if any (string)

Return a single JSON object with exactly those keys. Use null for unknowns.

Document text:
<<<
{text}
>>>"""


# ─────────────────────────────── Validation ──────────────────────────────

def _valid_amount(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f < 0 or f != f:          # reject negatives and NaN
        return None
    return round(f, 2)


def _valid_date(v: Any) -> Optional[date]:
    if not v or not isinstance(v, str):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(v.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _valid_gstin(v: Any) -> Optional[str]:
    if not v or not isinstance(v, str):
        return None
    s = v.strip().upper()
    return s if _GSTIN_RE.match(s) else None


def _valid_str(v: Any, max_len: int = 500) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    s = v.strip()
    if not s:
        return None
    return s[:max_len]


def _validate(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Apply per-field validation. Drop any field that fails — never store
    garbage. Returns a dict with only the valid keys present."""
    out: Dict[str, Any] = {}

    if (v := _valid_str(raw.get("party_name"))) is not None:
        out["party_name"] = v
    if (v := _valid_gstin(raw.get("party_gstin"))) is not None:
        out["party_gstin"] = v
    if (v := _valid_date(raw.get("doc_date"))) is not None:
        out["doc_date"] = v
    if (v := _valid_str(raw.get("doc_number"), 100)) is not None:
        out["doc_number"] = v
    if (v := _valid_amount(raw.get("total_amount"))) is not None:
        out["total_amount"] = v
    if (v := _valid_amount(raw.get("tax_amount"))) is not None:
        out["tax_amount"] = v
    if (v := _valid_amount(raw.get("net_amount"))) is not None:
        out["net_amount"] = v
    if (v := _valid_str(raw.get("payment_terms"), 200)) is not None:
        out["payment_terms"] = v
    # ref_doc_number must look like a real document number — the field on
    # Indian GRNs/invoices is often filled with "Verbal" / "Verbal Order"
    # meaning "no formal PO exists". Capturing those pollutes
    # document_references with strings that can never resolve. Require at
    # least one digit.
    if (v := _valid_str(raw.get("ref_doc_number"), 100)) is not None \
            and any(c.isdigit() for c in v):
        out["ref_doc_number"] = v

    return out


# ─────────────────────────────── Client ──────────────────────────────────

def extract_entities(text: str,
                     model: str = OLLAMA_MODEL,
                     timeout: float = TIMEOUT_SEC) -> Dict[str, Any]:
    """Run entity extraction against Ollama. Returns validated dict, or
    ``{}`` on any failure (logged)."""
    if not text or not text.strip():
        return {}

    prompt = _USER_TEMPLATE.format(text=text[:MAX_TEXT_CHARS])
    payload = {
        "model": model,
        "prompt": prompt,
        "system": _SYSTEM_PROMPT,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512,
        },
    }

    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        raw = body.get("response", "").strip()
        if not raw:
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            logger.warning("[Extract] Non-dict JSON from Ollama: %r", parsed)
            return {}
        return _validate(parsed)
    except requests.Timeout:
        logger.warning("[Extract] Ollama timeout (>%.0fs) — skipping", timeout)
    except requests.RequestException as e:
        logger.warning("[Extract] Ollama request failed: %s", e)
    except json.JSONDecodeError as e:
        logger.warning("[Extract] Invalid JSON from Ollama: %s", e)
    except Exception as e:  # last-resort guard — never break ingestion
        logger.warning("[Extract] Unexpected error: %s", e, exc_info=True)
    return {}
