"""Filename metadata parser — AI-4.

Parses structured filenames into (doc_month, doc_unit, doc_type,
fiscal_year, serial_no).  Handles the four patterns actually present in
the corpus:

  1. ``DEC-U2-PUR-24-25-49.pdf``         MONTH-UNIT-DOCTYPE-FY-FY-SERIAL
  2. ``24-25-100025.pdf``                FY-FY-SERIAL   (only FY + serial)
  3. ``24-25-DC-U1-0014.pdf``            FY-FY-DOCTYPE-UNIT-SERIAL
  4. ``OCT-U-2-JV-24-25-4.pdf``          pattern 1 with a stray dash in UNIT

Any field that cannot be confidently extracted is omitted from the
result dict — callers should treat missing keys as NULL.  The parser
never raises.
"""
from __future__ import annotations

import os
import re
from typing import Dict

MONTH_MAP = {
    "JAN": "January", "FEB": "February", "MAR": "March", "APR": "April",
    "MAY": "May",     "JUN": "June",     "JUL": "July",  "AUG": "August",
    "SEP": "September", "OCT": "October", "NOV": "November", "DEC": "December",
}

# "DC" = Delivery Challan (common in Indian procurement docs, observed in
# corpus as ``24-25-DC-U1-0014.pdf``). All other codes per the spec.
DOC_TYPE_MAP = {
    "PUR": "Purchase Order",
    "RM":  "Goods Receipt Note",
    "DN":  "Delivery Note",
    "DC":  "Delivery Challan",
    "JV":  "Journal Voucher",
    "INV": "Sales Invoice",
    "SO":  "Sales Order",
    "PI":  "Purchase Invoice",
    "CR":  "Credit Note",
    "DR":  "Debit Note",
}

_EXT_RE = re.compile(r"\.(pdf|xlsx?|docx?|csv|txt)$", re.IGNORECASE)
_UNIT_RE = re.compile(r"^U\d+$", re.IGNORECASE)
_YY_RE = re.compile(r"^\d{2}$")


def _normalize(name: str) -> list[str]:
    """Strip extension, split on '-', and collapse ``U-<digit>`` back into
    ``U<digit>`` so pattern 4 (``OCT-U-2-JV-...``) matches pattern 1."""
    stem = _EXT_RE.sub("", os.path.basename(name))
    parts = stem.split("-")
    fixed: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if (p.upper() == "U"
                and i + 1 < len(parts)
                and parts[i + 1].isdigit()):
            fixed.append(f"U{parts[i + 1]}")
            i += 2
            continue
        fixed.append(p)
        i += 1
    return fixed


def parse_filename_metadata(filename: str) -> Dict[str, str]:
    """Return whichever of (doc_month, doc_unit, doc_type, fiscal_year,
    serial_no) the filename structure supports. Unmatched patterns yield
    ``{}`` rather than an error."""
    if not filename:
        return {}

    parts = _normalize(filename)
    if not parts:
        return {}

    result: Dict[str, str] = {}
    upper = [p.upper() for p in parts]

    # Pattern 1 / 4: MONTH-UNIT-DOCTYPE-YY-YY-SERIAL
    if (len(parts) >= 6
            and upper[0] in MONTH_MAP
            and _UNIT_RE.match(parts[1])
            and upper[2] in DOC_TYPE_MAP
            and _YY_RE.match(parts[3])
            and _YY_RE.match(parts[4])):
        result["doc_month"]   = MONTH_MAP[upper[0]]
        result["doc_unit"]    = parts[1].upper()
        result["doc_type"]    = DOC_TYPE_MAP[upper[2]]
        result["fiscal_year"] = f"FY 20{parts[3]}-20{parts[4]}"
        result["serial_no"]   = parts[5]
        return result

    # Pattern 3: YY-YY-DOCTYPE-UNIT-SERIAL
    if (len(parts) >= 5
            and _YY_RE.match(parts[0])
            and _YY_RE.match(parts[1])
            and upper[2] in DOC_TYPE_MAP
            and _UNIT_RE.match(parts[3])):
        result["fiscal_year"] = f"FY 20{parts[0]}-20{parts[1]}"
        result["doc_type"]    = DOC_TYPE_MAP[upper[2]]
        result["doc_unit"]    = parts[3].upper()
        result["serial_no"]   = parts[4]
        return result

    # Pattern 2: YY-YY-SERIAL  (serial only, no month/unit/doctype)
    if (len(parts) >= 3
            and _YY_RE.match(parts[0])
            and _YY_RE.match(parts[1])
            and parts[2].isdigit()):
        result["fiscal_year"] = f"FY 20{parts[0]}-20{parts[1]}"
        result["serial_no"]   = parts[2]
        return result

    return result
