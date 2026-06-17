"""
Cross-Reference Validator — Catches numerical inconsistencies between
text mentions and table data within the same DRHP.
"""
import re
from src.config import AMOUNT_PATTERNS


class CrossReferenceValidator:
    """Detects numerical mismatches between running text and financial tables."""

    def validate(self, parsed_doc) -> list:
        """
        Extract financial figures from text and tables, flag mismatches.

        Args:
            parsed_doc: ParsedDocument object

        Returns:
            List of finding dicts (one per mismatch found)
        """
        # Step 1: Extract financial figures from running text
        text_figures = self._extract_text_figures(parsed_doc)

        # Step 2: Extract figures from tables
        table_figures = self._extract_table_figures(parsed_doc)

        # Step 3: Cross-reference — find same metric with different values
        mismatches = self._find_mismatches(text_figures, table_figures)

        # Step 4: Create findings
        findings = []
        for mm in mismatches:
            findings.append({
                "rule_id": f"XREF_{mm['metric_hash']}",
                "rule_title": f"Numerical Inconsistency: {mm['label']}",
                "regulation_ref": "Schedule VI, Part A — Internal Consistency",
                "source_document": "Internal Cross-Reference Check",
                "status": "NON_COMPLIANT",
                "severity": "HIGH",
                "confidence": 0.90,
                "check_type": "CROSS_REFERENCE",
                "evidence": {
                    "document_excerpt": (
                        f"Text (p.{mm['text_page']}): \"{mm['text_context'][:150]}\"\n"
                        f"Table (p.{mm['table_page']}): {mm['table_value']}"
                    ),
                    "page_numbers": [mm["text_page"], mm["table_page"]],
                    "rule_text": "Financial figures in text must be consistent with financial tables.",
                },
                "explanation": (
                    f"The document mentions '{mm['label']}' as {mm['text_raw']} "
                    f"on page {mm['text_page']}, but the financial table on page "
                    f"{mm['table_page']} shows {mm['table_value']}. "
                    f"Discrepancy: {mm['discrepancy']:.2f} {mm['unit']}."
                ),
                "recommendation": (
                    f"Verify the correct figure for '{mm['label']}' and ensure "
                    f"consistency between text and tables."
                ),
            })

        if findings:
            print(f"  🔢 Found {len(findings)} numerical inconsistencies")
        else:
            print(f"  ✅ No numerical inconsistencies detected")

        return findings

    def _extract_text_figures(self, parsed_doc) -> list:
        """Extract all financial figures from running text."""
        figures = []

        for page in parsed_doc.pages:
            text = page.text
            for pattern in AMOUNT_PATTERNS:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for m in matches:
                    value = self._parse_amount(m.group(1), m.group(2))
                    if value is None:
                        continue

                    # Get surrounding context for label extraction
                    start = max(0, m.start() - 120)
                    end = min(len(text), m.end() + 50)
                    context = text[start:end].strip()

                    label = self._extract_label(context)

                    figures.append({
                        "value": value,
                        "raw": m.group(0),
                        "unit": m.group(2).lower(),
                        "page": page.page_num,
                        "context": context,
                        "label": label,
                    })

        return figures

    def _extract_table_figures(self, parsed_doc) -> list:
        """Extract financial figures from tables."""
        figures = []

        for page in parsed_doc.pages:
            for table in page.tables:
                if not table.raw or len(table.raw) < 2:
                    continue

                headers = table.raw[0] if table.raw[0] else []

                for row_idx, row in enumerate(table.raw[1:], 1):
                    if not row:
                        continue

                    row_label = str(row[0]).strip() if row else ""

                    for col_idx, cell in enumerate(row):
                        cell_str = str(cell).strip() if cell else ""

                        # Try to parse as financial value
                        value = self._parse_table_cell(cell_str)
                        if value is not None:
                            col_header = str(headers[col_idx]).strip() if col_idx < len(headers) else ""
                            figures.append({
                                "value": value,
                                "raw": cell_str,
                                "page": page.page_num,
                                "row_label": row_label,
                                "col_header": col_header,
                                "label": f"{row_label} {col_header}".strip(),
                            })

        return figures

    def _find_mismatches(self, text_figures, table_figures, tolerance=0.05) -> list:
        """Find mismatches between text and table figures."""
        mismatches = []

        for tf in text_figures:
            if not tf["label"] or len(tf["label"]) < 3:
                continue

            for tbf in table_figures:
                if not self._labels_match(tf["label"], tbf["label"]):
                    continue

                # Check if values are different (beyond tolerance)
                if tf["value"] > 0 and tbf["value"] > 0:
                    # ORDER OF MAGNITUDE GUARD: skip if values differ by >100x
                    # This prevents false alarms like face value (₹10) vs
                    # issue size (₹1,500 million) which are different metrics
                    magnitude_ratio = max(tf["value"], tbf["value"]) / min(tf["value"], tbf["value"])
                    if magnitude_ratio > 100:
                        continue

                    ratio = abs(tf["value"] - tbf["value"]) / max(tf["value"], tbf["value"])
                    if ratio > tolerance:
                        import hashlib
                        metric_hash = hashlib.md5(
                            f"{tf['label']}_{tf['page']}_{tbf['page']}".encode()
                        ).hexdigest()[:8]

                        mismatches.append({
                            "label": tf["label"],
                            "text_raw": tf["raw"],
                            "text_value": tf["value"],
                            "text_page": tf["page"],
                            "text_context": tf["context"],
                            "table_value": tbf["raw"],
                            "table_page": tbf["page"],
                            "discrepancy": abs(tf["value"] - tbf["value"]),
                            "unit": tf.get("unit", ""),
                            "metric_hash": metric_hash,
                        })

        # Deduplicate (keep first occurrence of each metric)
        seen = set()
        unique = []
        for mm in mismatches:
            key = (mm["label"], mm["text_page"], mm["table_page"])
            if key not in seen:
                seen.add(key)
                unique.append(mm)

        return unique[:10]  # Limit to 10 most significant

    def _parse_amount(self, number_str: str, unit_str: str):
        """Parse a financial amount from text."""
        try:
            number = float(number_str.replace(",", ""))
            unit = unit_str.lower().strip()

            if unit in ("crore", "cr"):
                return number
            elif unit in ("lakh", "lac"):
                return number / 100  # Convert to crores for comparison
            elif unit == "million":
                return number * 0.1  # Rough conversion
            elif unit == "billion":
                return number * 100  # Rough conversion
            return number
        except (ValueError, AttributeError):
            return None

    def _parse_table_cell(self, cell: str):
        """Try to parse a table cell as a financial number."""
        if not cell:
            return None

        # Remove common formatting
        cleaned = cell.replace(",", "").replace("(", "-").replace(")", "")
        cleaned = re.sub(r"[₹$€£\s]", "", cleaned)
        cleaned = cleaned.strip()

        try:
            value = float(cleaned)
            if abs(value) > 0.01:  # Ignore zero/near-zero
                return abs(value)
        except ValueError:
            pass

        return None

    def _extract_label(self, context: str) -> str:
        """Extract a financial metric label from surrounding context."""
        # Common financial terms to look for
        terms = [
            "revenue", "turnover", "sales", "income", "profit", "loss",
            "ebitda", "net worth", "total assets", "total liabilities",
            "share capital", "reserves", "borrowings", "net profit",
            "gross profit", "operating profit", "pat", "pbt",
            "eps", "book value", "face value", "issue price",
            "offer size", "fresh issue", "offer for sale",
        ]

        context_lower = context.lower()
        for term in terms:
            if term in context_lower:
                return term

        # Fallback: take first few words before the number
        words = context.split()[:5]
        return " ".join(w for w in words if w.isalpha())[:50]

    def _labels_match(self, label1: str, label2: str) -> bool:
        """Check if two labels refer to the same metric."""
        l1 = set(label1.lower().split())
        l2 = set(label2.lower().split())

        # Remove common stopwords
        stopwords = {"of", "the", "and", "in", "for", "from", "to", "as", "at", "on", "total"}
        l1 = l1 - stopwords
        l2 = l2 - stopwords

        if not l1 or not l2:
            return False

        # Check overlap
        overlap = l1 & l2
        return len(overlap) >= min(len(l1), len(l2)) * 0.5
