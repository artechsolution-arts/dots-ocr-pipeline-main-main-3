"""Pure-logic unit tests for the ingestion-spec modules.

Runnable with ``python -m pytest tests/test_metadata.py`` from the
Rag_full_pipeline directory, or simply ``python tests/test_metadata.py``
— the module doubles as a runnable script if pytest is unavailable.

Deliberately avoids hitting Ollama, Postgres, or pdfplumber-on-disk —
those are integration tests and would make CI flaky.
"""
from __future__ import annotations

import datetime
import os
import sys
import unittest

# Allow running as a script without installing the project.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.metadata.enrichment import build_chunk_header
from src.ingestion.metadata.entity_extractor import _validate
from src.ingestion.metadata.filename_parser import parse_filename_metadata
from src.ingestion.metadata.quality import compute_quality_score
from src.ingestion.metadata.table_extractor import (
    _classify_headers, _format_row_chunk, _row_to_line_item, _to_num,
)


class FilenameParserTests(unittest.TestCase):
    def test_pattern_1_canonical(self):
        r = parse_filename_metadata("DEC-U2-PUR-24-25-49.pdf")
        self.assertEqual(r["doc_month"], "December")
        self.assertEqual(r["doc_unit"], "U2")
        self.assertEqual(r["doc_type"], "Purchase Order")
        self.assertEqual(r["fiscal_year"], "FY 2024-2025")
        self.assertEqual(r["serial_no"], "49")

    def test_pattern_2_fy_and_serial_only(self):
        r = parse_filename_metadata("24-25-100025.pdf")
        self.assertEqual(r, {"fiscal_year": "FY 2024-2025",
                             "serial_no": "100025"})

    def test_pattern_3_dc_delivery_challan(self):
        r = parse_filename_metadata("24-25-DC-U1-0014.pdf")
        self.assertEqual(r["doc_type"], "Delivery Challan")
        self.assertEqual(r["doc_unit"], "U1")
        self.assertEqual(r["fiscal_year"], "FY 2024-2025")
        self.assertEqual(r["serial_no"], "0014")
        self.assertNotIn("doc_month", r)

    def test_pattern_4_unit_with_stray_dash(self):
        r = parse_filename_metadata("OCT-U-2-JV-24-25-4.pdf")
        self.assertEqual(r["doc_unit"], "U2")
        self.assertEqual(r["doc_month"], "October")
        self.assertEqual(r["doc_type"], "Journal Voucher")

    def test_unknown_formats_return_empty(self):
        self.assertEqual(parse_filename_metadata("LEV-BL-25001-CHEM.pdf"), {})
        self.assertEqual(parse_filename_metadata("24-2-5DC-U2-0006.pdf"), {})
        self.assertEqual(parse_filename_metadata(""), {})

    def test_case_insensitive(self):
        r = parse_filename_metadata("dec-u2-pur-24-25-49.PDF")
        self.assertEqual(r["doc_month"], "December")
        self.assertEqual(r["doc_unit"], "U2")


class QualityScoreTests(unittest.TestCase):
    def test_empty_or_short_text_is_zero(self):
        self.assertEqual(compute_quality_score(""), 0.0)
        self.assertEqual(compute_quality_score("abc"), 0.0)

    def test_clean_english_scores_high(self):
        good = ("Purchase Order PO-U2-24-25-049 dated 12/12/2024 for "
                "500 MT cement at Rs 380 per MT total 190000")
        self.assertGreater(compute_quality_score(good), 0.6)

    def test_garbage_scores_low(self):
        junk = "~~~ !@#$%^ ≈≈ ≈ ∑∆ ^^^ \x00\x01 ¿?¡¡"
        self.assertLess(compute_quality_score(junk), 0.3)


class EnrichmentHeaderTests(unittest.TestCase):
    def test_full_metadata_includes_all_fields(self):
        h = build_chunk_header(
            "MAY-U3-RM-24-25-31.pdf",
            {"doc_type": "Goods Receipt Note", "doc_month": "May",
             "fiscal_year": "FY 2024-2025", "doc_unit": "U3"},
            {"party_name": "ABC Traders Pvt Ltd", "total_amount": 224200},
        )
        self.assertIn("Goods Receipt Note", h)
        self.assertIn("May FY 2024-2025", h)
        self.assertIn("U3", h)
        self.assertIn("Vendor: ABC Traders Pvt Ltd", h)
        self.assertIn("Total: ₹224,200", h)

    def test_missing_fields_are_skipped_not_printed_as_none(self):
        h = build_chunk_header("24-25-100025.pdf",
                                {"fiscal_year": "FY 2024-2025",
                                 "serial_no": "100025"}, {})
        self.assertNotIn("None", h)
        self.assertNotIn("Vendor:", h)
        self.assertNotIn("Total:", h)

    def test_bad_total_amount_does_not_crash(self):
        h = build_chunk_header("f.pdf", {}, {"total_amount": "not-a-number"})
        self.assertNotIn("Total:", h)


class EntityExtractorValidationTests(unittest.TestCase):
    """We only test the pure validator; the HTTP client is exercised in
    an integration test against a live Ollama."""

    def test_valid_gstin_passes(self):
        r = _validate({"party_gstin": "27AABCA1234A1Z5"})
        self.assertEqual(r["party_gstin"], "27AABCA1234A1Z5")

    def test_invalid_gstin_dropped(self):
        self.assertEqual(_validate({"party_gstin": "NOTREAL"}), {})
        self.assertEqual(_validate({"party_gstin": "27aabca1234a1z"}), {})

    def test_date_parsing_multiple_formats(self):
        for s in ("2024-12-15", "15/12/2024", "15-12-2024", "15.12.2024"):
            r = _validate({"doc_date": s})
            self.assertEqual(r["doc_date"], datetime.date(2024, 12, 15))

    def test_negative_amount_dropped(self):
        self.assertEqual(_validate({"total_amount": -5}), {})

    def test_string_amount_coerced(self):
        r = _validate({"total_amount": "1234.50"})
        self.assertEqual(r["total_amount"], 1234.50)

    def test_nonsense_keys_ignored(self):
        r = _validate({"party_name": "X",
                       "launch_code": "abc",
                       "total_amount": 100})
        self.assertEqual(set(r.keys()), {"party_name", "total_amount"})

    def test_ref_doc_number_must_have_digits(self):
        # "Verbal" / "Verbal Order" on Indian GRNs means "no PO" — don't
        # pollute document_references with strings that cannot resolve.
        self.assertEqual(_validate({"ref_doc_number": "Verbal"}), {})
        self.assertEqual(_validate({"ref_doc_number": "VERBAL ORDER"}), {})
        r = _validate({"ref_doc_number": "PO/U2/24-25/0049"})
        self.assertEqual(r["ref_doc_number"], "PO/U2/24-25/0049")

    def test_strings_trimmed_and_capped(self):
        r = _validate({"party_name": "  ACME  "})
        self.assertEqual(r["party_name"], "ACME")
        r2 = _validate({"payment_terms": "x" * 1000})
        self.assertLessEqual(len(r2["payment_terms"]), 200)


class TableExtractorTests(unittest.TestCase):
    def test_to_num_handles_commas_and_currency(self):
        self.assertEqual(_to_num("1,234.50"), 1234.50)
        self.assertEqual(_to_num("₹ 500"), 500.0)
        self.assertIsNone(_to_num(""))
        self.assertIsNone(_to_num(None))

    def test_classify_headers_matches_common_aliases(self):
        hdr = ["Sr", "Description", "HSN Code", "Qty", "Unit",
               "Rate", "Amount", "GST%"]
        m = _classify_headers(hdr)
        self.assertEqual(m[1], "description")
        self.assertEqual(m[2], "hsn_sac")
        self.assertEqual(m[3], "quantity")
        self.assertEqual(m[4], "unit_of_measure")
        self.assertEqual(m[5], "unit_rate")
        self.assertEqual(m[6], "amount")
        self.assertEqual(m[7], "tax_rate")

    def test_row_with_qty_and_description_kept(self):
        hdr_map = {0: "description", 1: "quantity", 2: "amount"}
        item = _row_to_line_item(["Cement OPC 53", "500", "190000"], hdr_map, 3)
        self.assertEqual(item["description"], "Cement OPC 53")
        self.assertEqual(item["quantity"], 500.0)
        self.assertEqual(item["amount"], 190000.0)
        self.assertEqual(item["line_no"], 3)

    def test_row_with_no_numbers_dropped(self):
        hdr_map = {0: "description", 1: "quantity"}
        self.assertIsNone(_row_to_line_item(["Total", ""], hdr_map, 1))

    def test_row_without_description_dropped(self):
        hdr_map = {0: "description", 1: "quantity", 2: "amount"}
        self.assertIsNone(_row_to_line_item(["", "1", "100"], hdr_map, 1))

    def test_format_row_chunk_matches_spec(self):
        out = _format_row_chunk(
            "MAY-U3-RM-24-25-31", 2, 3,
            ["Sr", "Description", "HSN", "Qty", "Unit", "Rate", "Amount"],
            ["3", "Cement OPC 53 Grade", "25011000", "500", "MT",
             "380.00", "1,90,000.00"],
        )
        self.assertIn("[Table: Line Items | Doc: MAY-U3-RM-24-25-31 | Page: 2]", out)
        self.assertIn("Columns: Sr | Description | HSN | Qty | Unit | Rate | Amount", out)
        self.assertTrue(out.rstrip().endswith("1,90,000.00"))
        self.assertIn("Row 3: 3 | Cement OPC 53 Grade", out)


if __name__ == "__main__":
    unittest.main()
