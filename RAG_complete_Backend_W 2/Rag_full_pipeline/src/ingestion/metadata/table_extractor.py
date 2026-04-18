"""Table extraction for line-items — AI-6.

Uses pdfplumber against the raw PDF to find table rows and normalize
them into two shapes:

* ``document_line_items`` rows for the SQL side (numeric aggregation).
* Per-row chunk strings formatted as the spec prescribes, so retrieval
  can surface individual table rows without losing header context.

pdfplumber only works on PDFs with selectable text; scanned/image PDFs
return empty here, and the caller falls through to the OCR-based text
chunker. Heuristics are intentionally conservative — rows that look
like totals/headers/page-reprints (no numeric qty AND no numeric
amount) are dropped from the line-items list.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")
_HSN_RE = re.compile(r"^\d{4,8}$")

# Fuzzy header → schema column. Compared against lowercased, space-free.
_HEADER_MAP = {
    "description": ("description", "particulars", "product", "item",
                     "itemdescription", "productdescription", "goods",
                     "materialdescription", "goodsservices"),
    "hsn_sac":     ("hsn", "sac", "hsnsac", "hsncode", "hsnsaccode"),
    "quantity":    ("qty", "quantity", "qtynos"),
    "unit_of_measure": ("unit", "uom", "units", "unitmeasure"),
    "unit_rate":   ("rate", "unitrate", "price", "unitprice", "ratepermt",
                    "rs", "rateperqty"),
    "amount":      ("amount", "amt", "totalamount", "lineamount", "total",
                    "value", "taxablevalue"),
    "tax_rate":    ("taxrate", "gst", "gstrate", "gst%", "taxpct", "igst",
                    "cgst", "sgst"),
    "tax_amount":  ("taxamount", "gstamount", "taxamt", "gstamt"),
}


def _norm(s: Any) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def _to_num(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    m = _NUM_RE.search(s.replace("\u00a0", " "))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _classify_headers(header_row: List[Any]) -> Dict[int, str]:
    """Return ``{column_index: canonical_field_name}`` for each header
    cell that maps to a known field; unmapped columns are omitted."""
    out: Dict[int, str] = {}
    for i, cell in enumerate(header_row or []):
        key = _norm(cell)
        if not key:
            continue
        for field, aliases in _HEADER_MAP.items():
            if key in aliases or any(a in key for a in aliases):
                if i not in out:     # first-match wins per column
                    out[i] = field
                break
    return out


def _row_to_line_item(row: List[Any], header_map: Dict[int, str],
                     line_no: int) -> Optional[Dict[str, Any]]:
    """Project a raw row through ``header_map``. Drop rows with neither a
    numeric quantity nor a numeric amount — those are usually section
    separators, totals, or the header re-printed on a new page."""
    item: Dict[str, Any] = {"line_no": line_no}

    for idx, field in header_map.items():
        if idx >= len(row):
            continue
        raw = row[idx]
        if field in ("description", "unit_of_measure"):
            s = str(raw).strip() if raw is not None else ""
            if s:
                item[field] = s
        elif field == "hsn_sac":
            s = re.sub(r"\D", "", str(raw or ""))
            if s and _HSN_RE.match(s):
                item[field] = s
        else:
            n = _to_num(raw)
            if n is not None:
                item[field] = n

    has_qty    = "quantity" in item and item["quantity"] > 0
    has_amount = "amount"   in item and item["amount"]   > 0
    if not (has_qty or has_amount):
        return None
    if not item.get("description"):
        return None
    return item


def _format_row_chunk(doc_stem: str, page_num: int, line_no: int,
                      header_row: List[Any], row: List[Any]) -> str:
    """Spec-prescribed per-row chunk string (AI-6 step 2).

    Format::

        [Table: Line Items | Doc: <stem> | Page: <n>]
        Columns: H1 | H2 | H3 | ...
        Row <line_no>: V1 | V2 | V3 | ...
    """
    cols = [str(h).strip() for h in (header_row or []) if str(h or "").strip()]
    vals = [("" if c is None else str(c).strip()) for c in row]
    return (
        f"[Table: Line Items | Doc: {doc_stem} | Page: {page_num}]\n"
        f"Columns: {' | '.join(cols)}\n"
        f"Row {line_no}: {' | '.join(vals)}"
    )


def extract_tables(pdf_path: str,
                   max_rows: int = 500) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(line_items, row_chunks)``.

    ``line_items``  — dicts for ``RBACManager.add_line_items`` (numeric,
                       conservative).
    ``row_chunks``  — dicts ``{content, page, line_no}`` for every
                       non-empty data row (wider than line_items — we
                       keep rows even if they fail the numeric gate so
                       retrieval can still surface them as text).

    Empty lists on scanned PDFs or any error — never raises.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return [], []

    try:
        import pdfplumber
    except ImportError:
        logger.warning("[Tables] pdfplumber not installed — skipping")
        return [], []

    items: List[Dict[str, Any]] = []
    row_chunks: List[Dict[str, Any]] = []
    doc_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        with pdfplumber.open(pdf_path) as pdf:
            line_no = 0
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                except Exception as e:
                    logger.debug("[Tables] page extract failed: %s", e)
                    continue

                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    header_map = _classify_headers(tbl[0])
                    if "description" not in header_map.values():
                        continue
                    for row in tbl[1:]:
                        if not any(str(c or "").strip() for c in row):
                            continue
                        line_no += 1

                        row_chunks.append({
                            "content":  _format_row_chunk(
                                doc_stem, page_idx, line_no, tbl[0], row),
                            "page":     page_idx,
                            "line_no":  line_no,
                        })

                        item = _row_to_line_item(row, header_map, line_no)
                        if item:
                            items.append(item)

                        if len(row_chunks) >= max_rows:
                            return items, row_chunks
    except Exception as e:
        logger.warning("[Tables] extract failed for %s: %s",
                       os.path.basename(pdf_path), e)
    return items, row_chunks


def extract_line_items(pdf_path: str,
                       max_rows: int = 500) -> List[Dict[str, Any]]:
    """Back-compat wrapper for callers that only want line_items."""
    items, _ = extract_tables(pdf_path, max_rows=max_rows)
    return items
