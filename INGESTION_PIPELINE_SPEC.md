# Virchow — Ingestion Pipeline Specification
> Prepared for the ingestion pipeline team · April 2026

---

## Context: What Are We Building?

Virchow is an internal enterprise document intelligence system. The document corpus consists of financial and procurement transaction documents — Purchase Orders (PO), Sales Invoices, Purchase Invoices, Goods Receipt Notes (GRN), Delivery Notes (DN), and Journal Vouchers (JV) — organized by month, unit/department, and financial year.

### Document Naming Convention

Documents follow a structured filename format:

```
{MONTH}-{UNIT}-{DOCTYPE}-{FY_START}-{FY_END}-{SERIAL}.pdf

Examples:
  DEC-U2-PUR-24-25-40.pdf   → December, Unit 2, Purchase Order, FY 2024-25, Serial #40
  MAY-U3-RM-24-25-31.pdf    → May, Unit 3, Goods Receipt Note (GRN), FY 2024-25, Serial #31
  FEB-U2-DN-24-25-12.pdf    → February, Unit 2, Delivery Note, FY 2024-25, Serial #12
  DEC-U2-JV-19-20-5.pdf     → December, Unit 2, Journal Voucher, FY 2019-20, Serial #5
```

**Document type codes:**

| Code | Document Type |
|------|--------------|
| PUR  | Purchase Order |
| RM   | Goods Receipt Note (GRN) |
| DN   | Delivery Note |
| JV   | Journal Voucher |
| INV  | Sales Invoice |
| SO   | Sales Order |
| PI   | Purchase Invoice |
| CR   | Credit Note |
| DR   | Debit Note |

---

## Why the Ingestion Pipeline Needs an Overhaul

The current ingestion pipeline extracts raw text, splits it into chunks, and generates vector embeddings. That's it. No structure is extracted.

This means:

- **A new employee** asking "who are our vendors?" gets random text chunks — no synthesized answer.
- **C-suite** asking "total orders placed with ABC Traders in FY 2024-25" gets nothing — there's no aggregatable data.
- **Operations team** asking "what was the unit rate for cement in this GRN?" gets a garbled table chunk — no row-level data.

The retrieval system (already improved) can handle intent classification and query routing — but it can only work with what ingestion provides. The following action items transform the pipeline from "text extraction + embeddings" into a structured, queryable knowledge base.

---

## Current Database Schema (Retrieval Side — for Reference)

```sql
departments       -- id, name, description, is_active
users             -- id, email, name, department_id, role
chat              -- id, user_id, department_id, title
messages          -- id, chat_id, role, content
documents         -- id, title, file_name, department_id, uploaded_by, page_count
chunks            -- id, document_id, chunk_index, chunk_text, chunk_token_count, page_num
embeddings        -- id, chunk_id, department_id, embedding (vector), embedding_model
rag_retrieval_log -- id, chat_id, user_id, query_text, retrieved_chunk_ids, similarity_scores
```

Everything below describes additions and changes to this schema plus new pipeline steps.

---

## Action Items

---

### Action Item 1 — Enrich the `documents` Table

Add structured metadata columns populated at ingestion time. These are the foundation for all analytical and exploratory queries.

```sql
ALTER TABLE documents ADD COLUMN doc_month      TEXT;          -- 'December'
ALTER TABLE documents ADD COLUMN doc_unit       TEXT;          -- 'U2'
ALTER TABLE documents ADD COLUMN doc_type       TEXT;          -- 'Purchase Order'
ALTER TABLE documents ADD COLUMN fiscal_year    TEXT;          -- 'FY 2024-25'
ALTER TABLE documents ADD COLUMN serial_no      TEXT;          -- '40'
ALTER TABLE documents ADD COLUMN party_name     TEXT;          -- 'ABC Traders Pvt Ltd'
ALTER TABLE documents ADD COLUMN party_gstin    TEXT;          -- '27AABCA1234A1Z5'
ALTER TABLE documents ADD COLUMN doc_date       DATE;          -- 2024-12-15
ALTER TABLE documents ADD COLUMN doc_number     TEXT;          -- 'PO/U2/24-25/0040'
ALTER TABLE documents ADD COLUMN total_amount   NUMERIC(15,2); -- 456000.00
ALTER TABLE documents ADD COLUMN tax_amount     NUMERIC(15,2); -- 54000.00
ALTER TABLE documents ADD COLUMN net_amount     NUMERIC(15,2); -- 402000.00
ALTER TABLE documents ADD COLUMN payment_terms  TEXT;          -- 'Net 30 days'
ALTER TABLE documents ADD COLUMN ref_doc_number TEXT;          -- PO number cited in invoice/GRN
ALTER TABLE documents ADD COLUMN ocr_quality    FLOAT;         -- 0.0–1.0 quality score
```

**Indexes to add:**

```sql
CREATE INDEX idx_doc_party_name  ON documents USING gin(to_tsvector('english', coalesce(party_name, '')));
CREATE INDEX idx_doc_fiscal_year ON documents(fiscal_year);
CREATE INDEX idx_doc_type        ON documents(doc_type);
CREATE INDEX idx_doc_date        ON documents(doc_date);
CREATE INDEX idx_doc_dept_type   ON documents(department_id, doc_type);
```

---

### Action Item 2 — Add `document_line_items` Table

Line-item level data extracted from tables inside each document. Critical for SKU/product/quantity queries.

```sql
CREATE TABLE document_line_items (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    department_id   UUID NOT NULL REFERENCES departments(id),
    line_no         INTEGER,
    description     TEXT,           -- product or material name
    hsn_sac         TEXT,           -- HSN or SAC code
    quantity        NUMERIC(15,3),
    unit_of_measure TEXT,           -- KG, MT, NOS, PCS, LTR, etc.
    unit_rate       NUMERIC(15,2),
    amount          NUMERIC(15,2),
    tax_rate        NUMERIC(5,2),   -- GST percentage
    tax_amount      NUMERIC(15,2),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_line_items_doc    ON document_line_items(document_id);
CREATE INDEX idx_line_items_dept   ON document_line_items(department_id);
CREATE INDEX idx_line_items_hsn    ON document_line_items(hsn_sac);
CREATE INDEX idx_line_items_search ON document_line_items
    USING gin(to_tsvector('english', coalesce(description, '')));
```

**Example row:**
```
document_id: <uuid of MAY-U3-RM-24-25-31>
line_no:     3
description: Cement OPC 53 Grade
hsn_sac:     25011000
quantity:    500
unit:        MT
unit_rate:   380.00
amount:      190000.00
tax_rate:    18.00
tax_amount:  34200.00
```

---

### Action Item 3 — Add `document_references` Table

Links documents to each other via referenced document numbers (e.g. an invoice referencing a PO number).

```sql
CREATE TABLE document_references (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_doc_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ref_doc_number  TEXT NOT NULL,      -- the PO/invoice number string mentioned in source
    ref_doc_id      UUID REFERENCES documents(id) ON DELETE SET NULL, -- resolved FK if found
    ref_type        TEXT NOT NULL,      -- 'po_reference' | 'invoice_reference' | 'grn_reference'
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_doc_refs_source ON document_references(source_doc_id);
CREATE INDEX idx_doc_refs_refnum ON document_references(ref_doc_number);
```

**Resolution step:** After all documents are ingested, run a pass that matches `ref_doc_number` strings against `documents.doc_number` and populates `ref_doc_id`.

This enables queries like:
- "Did we receive what we ordered in PO-24-25-40?" → GRN references PO → compare quantities
- "Is this invoice backed by a purchase order?" → Invoice references PO → lookup chain

---

### Action Item 4 — Filename Metadata Parser (Zero ML, Implement First)

Parse the structured filename at upload time to populate `doc_month`, `doc_unit`, `doc_type`, `fiscal_year`, `serial_no`. No AI needed — pure regex/string splitting.

```python
MONTH_MAP = {
    "JAN": "January", "FEB": "February", "MAR": "March", "APR": "April",
    "MAY": "May", "JUN": "June", "JUL": "July", "AUG": "August",
    "SEP": "September", "OCT": "October", "NOV": "November", "DEC": "December",
}

DOC_TYPE_MAP = {
    "PUR": "Purchase Order",
    "RM":  "Goods Receipt Note",
    "DN":  "Delivery Note",
    "JV":  "Journal Voucher",
    "INV": "Sales Invoice",
    "SO":  "Sales Order",
    "PI":  "Purchase Invoice",
    "CR":  "Credit Note",
    "DR":  "Debit Note",
}

def parse_filename_metadata(filename: str) -> dict:
    """
    Parse DEC-U2-PUR-24-25-40.pdf → structured metadata dict.
    Returns only fields that can be confidently extracted.
    """
    import re
    name = re.sub(r'\.(pdf|xlsx?|docx?|csv|txt)$', '', filename, flags=re.IGNORECASE)
    parts = name.split('-')
    result = {}

    if len(parts) >= 1 and parts[0].upper() in MONTH_MAP:
        result["doc_month"] = MONTH_MAP[parts[0].upper()]

    if len(parts) >= 2 and re.match(r'^U\d+$', parts[1], re.IGNORECASE):
        result["doc_unit"] = parts[1].upper()

    if len(parts) >= 3 and parts[2].upper() in DOC_TYPE_MAP:
        result["doc_type"] = DOC_TYPE_MAP[parts[2].upper()]

    if len(parts) >= 5:
        try:
            result["fiscal_year"] = f"FY 20{parts[3]}-20{parts[4]}"
        except Exception:
            pass

    if len(parts) >= 6:
        result["serial_no"] = parts[5]

    return result
```

**This is the highest ROI action item — implement this first.** It gives every document month, unit, type, and financial year for free.

---

### Action Item 5 — LLM-Based Entity Extraction at Ingestion

After the raw text is extracted from each document, run a structured extraction prompt to pull financial entities. This runs **once at ingestion time** — latency does not matter.

**Extraction target fields:**

```
party_name      → vendor or customer name
party_gstin     → GSTIN (format: 2 digits + 10 alphanum + 1 digit + Z + 1 alphanum)
doc_date        → document date (DD/MM/YYYY or similar)
doc_number      → PO number / invoice number / GRN number
total_amount    → final payable amount including tax
tax_amount      → total GST amount
net_amount      → amount before tax
payment_terms   → e.g. "Net 30 days", "100% advance"
ref_doc_number  → referenced PO / invoice number (for GRNs and invoices)
```

**Recommended extraction approach:**

Use a small, fast local model (Ollama `qwen2.5:3b` or `mistral:7b` preferred over `0.5b` for extraction accuracy). Structure the prompt to return JSON:

```
Extract the following fields from this document. Return ONLY valid JSON.
If a field is not found, use null.

Fields: party_name, party_gstin, doc_date (YYYY-MM-DD), doc_number,
        total_amount (number only), tax_amount (number only),
        net_amount (number only), payment_terms, ref_doc_number

Document text:
{first_2000_chars_of_document}

JSON:
```

**Validation rules after extraction:**
- GSTIN: must match `^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$`
- Amounts: must be positive numbers
- doc_date: must be parseable as a date
- Reject and log any field that fails validation rather than storing bad data

---

### Action Item 6 — Table-Aware Chunking

The most impactful change for line-item accuracy. Current paragraph-based chunking garbles invoice tables into unreadable text.

**Recommended library:** `pdfplumber` (better table detection than PyMuPDF for structured PDFs)

**Strategy:**

1. **Detect tables** on each page using `pdfplumber.extract_tables()`
2. **For each table row**, create one chunk with column headers prepended:
   ```
   [Table: Line Items | Doc: MAY-U3-RM-24-25-31 | Page: 2]
   Columns: Sr | Description | HSN | Qty | Unit | Rate | Amount
   Row 3: 3 | Cement OPC 53 Grade | 25011000 | 500 | MT | 380.00 | 1,90,000.00
   ```
3. **Also insert** the row into `document_line_items` with parsed numeric values
4. **Non-table text** continues to use the existing paragraph/fixed-size chunking

**Fallback:** If `pdfplumber` cannot detect tables (scanned PDFs, image-based), fall back to current chunking and flag the document for manual review.

---

### Action Item 7 — Chunk Enrichment (Metadata Header Injection)

Every chunk — whether from a table row or a paragraph — should have the document's key metadata prepended as a header. This makes every chunk independently searchable and contextually anchored.

**Format:**

```
[{file_name} | {doc_type} | {doc_month} {fiscal_year} | {doc_unit} | Vendor: {party_name} | Total: ₹{total_amount}]

{original chunk text}
```

**Example:**

```
[MAY-U3-RM-24-25-31.pdf | Goods Receipt Note | May FY 2024-25 | U3 | Vendor: ABC Traders Pvt Ltd | Total: ₹2,24,200]

Received 500 MT of Cement OPC 53 Grade at ₹380/MT. Transport charges as per actuals.
Inspection passed. Store keeper signature obtained.
```

**Implementation note:** This header is prepended to `chunk_text` before embedding. The embedding should capture the document context — not just the raw paragraph text. Re-embed all enriched chunks.

---

### Action Item 8 — OCR Quality Scoring

Flag chunks with poor OCR output before they corrupt retrieval results.

**Quality score algorithm (store in `chunks.quality_score`):**

```python
import re

def compute_quality_score(text: str) -> float:
    if not text or len(text) < 10:
        return 0.0

    total_chars = len(text)

    # Ratio of alphanumeric + common punctuation to total characters
    clean_chars = len(re.findall(r'[a-zA-Z0-9₹.,;:\-/() ]', text))
    alpha_ratio = clean_chars / total_chars

    # Penalty for excessive special/garbage characters
    garbage = len(re.findall(r'[^\x00-\x7F]', text)) / total_chars  # non-ASCII ratio
    garbage_penalty = min(garbage * 2, 0.5)

    # Reward for recognizable word-like tokens
    words = text.split()
    real_words = [w for w in words if re.match(r'^[a-zA-Z]{3,}$', w)]
    word_ratio = len(real_words) / max(len(words), 1)

    score = (alpha_ratio * 0.4) + (word_ratio * 0.4) - garbage_penalty + 0.2
    return round(max(0.0, min(1.0, score)), 3)
```

**Retrieval behavior based on score:**

| Score | Action |
|-------|--------|
| < 0.3 | Exclude from retrieval entirely; flag document for re-OCR |
| 0.3–0.6 | Include but deprioritize (lower effective similarity) |
| > 0.6 | Normal retrieval |

**Schema change:**
```sql
ALTER TABLE chunks ADD COLUMN quality_score FLOAT DEFAULT 1.0;
CREATE INDEX idx_chunk_quality ON chunks(quality_score);
```

---

### Action Item 9 — Re-ingestion Pipeline for Existing Documents

~335 documents are already ingested without any of the above enrichment. A batch re-ingestion job is needed to backfill.

**Steps:**

```
For each existing document in the documents table:
  1. Fetch PDF from SeaweedFS using file_path
  2. Run filename metadata parser (Action Item 4) → update documents columns
  3. Run entity extraction (Action Item 5) → update party_name, party_gstin, amounts, etc.
  4. Re-extract tables (Action Item 6) → create new table-row chunks + populate line_items
  5. Enrich all existing chunks with metadata headers (Action Item 7) → update chunk_text
  6. Compute OCR quality scores (Action Item 8) → update chunks.quality_score
  7. Re-embed only NEW or CHANGED chunks
  8. Run document reference resolution (Action Item 3) → match ref_doc_number to doc_number
```

**Do NOT re-embed unchanged chunks** — vector embeddings for unmodified text are already correct and re-embedding wastes time and compute.

**Run in batches of 10–20 documents** with a delay to avoid overwhelming the embedding service and PostgreSQL.

**Logging:** Track `(document_id, step, status, error)` per document in a temp job log table so the pipeline can resume from failures without re-processing completed docs.

---

### Action Item 10 — Analytical Query Handler (Hybrid SQL + RAG)

Once structured data exists, add a query routing layer that handles analytical questions via SQL aggregation rather than vector search. The retrieval service already classifies queries as `INTENT_ANALYTICAL` — this handler is what fires when that intent is detected.

**Examples of queries this unlocks:**

```sql
-- "Total value of purchase orders placed with ABC Traders in FY 2024-25"
SELECT SUM(d.total_amount), COUNT(*)
FROM documents d
WHERE d.party_name ILIKE '%ABC Traders%'
  AND d.fiscal_year = 'FY 2024-25'
  AND d.doc_type = 'Purchase Order'
  AND d.department_id = $dept_id;

-- "Which products did we receive most in Unit 2 this financial year?"
SELECT li.description, SUM(li.quantity) AS total_qty, li.unit_of_measure
FROM document_line_items li
JOIN documents d ON d.id = li.document_id
WHERE d.doc_unit = 'U2'
  AND d.fiscal_year = 'FY 2024-25'
  AND d.doc_type = 'Goods Receipt Note'
  AND li.department_id = $dept_id
GROUP BY li.description, li.unit_of_measure
ORDER BY total_qty DESC
LIMIT 10;

-- "When did we first start transacting with Vendor X?"
SELECT MIN(d.doc_date), d.doc_type
FROM documents d
WHERE d.party_name ILIKE '%Vendor X%'
  AND d.department_id = $dept_id
GROUP BY d.doc_type;
```

**Integration with LLM:**
- Run the SQL query to get structured results
- Pass results as context to the LLM to generate a natural language answer
- Cite the source document names in the response

---

## Priority Order

| # | Action Item | Effort | Impact | Implement First? |
|---|-------------|--------|--------|-----------------|
| 1 | Filename metadata parser | Low | High | Yes — zero ML, immediate |
| 2 | Chunk enrichment with doc metadata | Low | High | Yes — improves all queries |
| 3 | Table-aware chunking | Medium | High | Yes — fixes line-item queries |
| 4 | Schema additions (documents, line_items, references) | Medium | High | Prerequisite for 5, 6, 10 |
| 5 | Entity extraction at ingestion | Medium | High | After schema is ready |
| 6 | Re-ingestion pipeline for existing 335 docs | High | High | After 1–5 are done |
| 7 | Document reference linking | Low | Medium | After re-ingestion |
| 8 | Analytical SQL query handler | Medium | High | After structured data exists |
| 9 | OCR quality scoring | Low | Medium | Can run in parallel |

---

## Interface Contract with the Retrieval Service

The retrieval service (`rag_pipeline/src/services/rag_pipeline.py`) already handles:
- Intent classification: `precision` | `exploratory` | `analytical`
- For `analytical` intent: will call an analytical handler (to be built per Action Item 10)
- For `exploratory` intent: runs global vector search with higher top_k and synthesis prompting
- For `precision` intent: scoped search by document name or active conversation context

**What the retrieval service expects from ingestion (after these changes):**

1. `chunks.chunk_text` — includes metadata header (Action Item 7)
2. `chunks.quality_score` — used to filter low-quality chunks before retrieval
3. `documents.*` metadata columns — used by analytical SQL handler
4. `document_line_items` — used by analytical SQL handler for product-level queries
5. `document_references` — used for cross-document linking queries

---

## Questions / Decisions Needed from the Team

1. **OCR source:** Are documents digitally generated PDFs (text-selectable) or scanned images? If scanned, an OCR step (Tesseract / AWS Textract / Azure Document Intelligence) is needed before text extraction.

2. **Table extraction accuracy:** `pdfplumber` works well for digital PDFs. For scanned PDFs with tables, a dedicated table extraction model (e.g. Microsoft Table Transformer) may be needed.

3. **Entity extraction model:** `qwen2.5:0.5b` (currently deployed) is too small for reliable JSON extraction. Recommend upgrading to at least `qwen2.5:3b` or `mistral:7b` for the ingestion extraction step only — the retrieval/chat model can remain at `0.5b` for speed.

4. **Re-ingestion timing:** Re-ingesting 335 documents with entity extraction will take time. Estimate 1–5 minutes per document depending on length and model speed. Plan accordingly — possibly run overnight.

5. **party_name normalization:** Vendor names appear in many forms ("ABC Traders", "ABC Traders Pvt Ltd", "ABC TRADERS PVT. LTD."). A normalization step (fuzzy deduplication) will be needed before the data is analytically useful.
