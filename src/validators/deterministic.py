"""
Deterministic Validator — Rule-based checks using regex and pattern matching.
These produce 100% confidence findings (binary pass/fail).
"""
import re
from src.config import (
    SECTION_PATTERNS, MANDATORY_SECTIONS,
    CIN_PATTERN, ISIN_PATTERN, PAN_PATTERN,
)


class DeterministicValidator:
    """Runs deterministic (regex/pattern) compliance checks."""

    def validate(self, rule: dict, parsed_doc) -> dict:
        """
        Run a deterministic check based on the rule.

        Args:
            rule: Compliance rule dict
            parsed_doc: ParsedDocument object

        Returns:
            Finding dict
        """
        rule_id = rule.get("rule_id", "UNKNOWN")

        # Route to specific check based on rule characteristics
        required_elements = rule.get("required_elements", [])
        keywords = [k.lower() for k in rule.get("search_keywords", [])]

        # Check 1: Mandatory sections presence
        if any("section" in k for k in keywords) or "mandatory" in rule.get("title", "").lower():
            return self._check_mandatory_sections(rule, parsed_doc)

        # Check 2: CIN/ISIN/PAN on cover page
        if any(x in " ".join(keywords) for x in ["cin", "corporate identity", "isin", "pan"]):
            return self._check_identifiers(rule, parsed_doc)

        # Check 3: Generic required element presence
        if required_elements:
            return self._check_required_elements(rule, parsed_doc)

        # Default: search for keywords in document
        return self._keyword_presence_check(rule, parsed_doc)

    def _check_mandatory_sections(self, rule: dict, parsed_doc) -> dict:
        """Check if all mandatory SEBI Schedule VI sections are present."""
        detected = {s.name for s in parsed_doc.sections}
        missing = [s for s in MANDATORY_SECTIONS if s not in detected]

        if not missing:
            return self._make_finding(
                rule=rule,
                status="COMPLIANT",
                confidence=1.0,
                evidence=f"All {len(MANDATORY_SECTIONS)} mandatory sections detected: {sorted(detected)}",
                page_numbers=[1],
                explanation=f"Document contains all required Schedule VI sections.",
            )
        else:
            return self._make_finding(
                rule=rule,
                status="NON_COMPLIANT",
                confidence=1.0,
                evidence=f"Missing sections: {missing}. Found: {sorted(detected)}",
                page_numbers=[1],
                explanation=f"{len(missing)} mandatory section(s) not found: {', '.join(missing)}. "
                           f"Detected {len(detected)}/{len(MANDATORY_SECTIONS)} required sections.",
            )

    def _check_identifiers(self, rule: dict, parsed_doc) -> dict:
        """Check for CIN, ISIN, PAN on cover page."""
        cover_text = "\n".join(p.text for p in parsed_doc.pages[:5])
        findings_parts = []
        all_found = True

        # CIN check
        cin_match = re.search(CIN_PATTERN, cover_text)
        if cin_match:
            findings_parts.append(f"CIN found: {cin_match.group(0)}")
        else:
            findings_parts.append("CIN NOT found on cover page")
            all_found = False

        # ISIN check
        isin_match = re.search(ISIN_PATTERN, cover_text)
        if isin_match:
            findings_parts.append(f"ISIN found: {isin_match.group(0)}")

        # PAN check (for promoters)
        pan_matches = re.findall(PAN_PATTERN, cover_text)
        if pan_matches:
            findings_parts.append(f"PAN found: {len(pan_matches)} instance(s)")

        evidence = "; ".join(findings_parts)

        if all_found:
            return self._make_finding(
                rule=rule,
                status="COMPLIANT",
                confidence=1.0,
                evidence=evidence,
                page_numbers=[1, 2, 3],
                explanation="Required identifiers found on cover page.",
            )
        else:
            return self._make_finding(
                rule=rule,
                status="NON_COMPLIANT",
                confidence=1.0,
                evidence=evidence,
                page_numbers=[1, 2, 3],
                explanation="One or more required identifiers missing from cover page.",
            )

    def _check_required_elements(self, rule: dict, parsed_doc) -> dict:
        """Check if required elements are present in the document."""
        full_text = "\n".join(p.text for p in parsed_doc.pages)
        required = rule.get("required_elements", [])
        found = []
        missing = []

        for element in required:
            if re.search(re.escape(element), full_text, re.IGNORECASE):
                found.append(element)
            else:
                # Try fuzzy match with keywords
                keywords = element.lower().split()
                if all(kw in full_text.lower() for kw in keywords if len(kw) > 3):
                    found.append(element + " (partial match)")
                else:
                    missing.append(element)

        status = "COMPLIANT" if not missing else "NON_COMPLIANT"
        if missing and len(missing) < len(required) / 2:
            status = "NEEDS_REVIEW"

        return self._make_finding(
            rule=rule,
            status=status,
            confidence=1.0 if status != "NEEDS_REVIEW" else 0.7,
            evidence=f"Found: {found}. Missing: {missing}" if missing else f"All elements found: {found}",
            page_numbers=[1],
            explanation=f"{len(found)}/{len(required)} required elements found.",
        )

    def _keyword_presence_check(self, rule: dict, parsed_doc) -> dict:
        """Generic keyword presence check."""
        full_text = "\n".join(p.text for p in parsed_doc.pages).lower()
        keywords = rule.get("search_keywords", [])

        found_keywords = []
        found_pages = []

        for keyword in keywords:
            if keyword.lower() in full_text:
                found_keywords.append(keyword)
                # Find which pages contain it
                for p in parsed_doc.pages:
                    if keyword.lower() in p.text.lower():
                        found_pages.append(p.page_num)
                        break

        ratio = len(found_keywords) / max(len(keywords), 1)

        if ratio >= 0.7:
            status = "COMPLIANT"
        elif ratio >= 0.3:
            status = "NEEDS_REVIEW"
        else:
            status = "NON_COMPLIANT"

        return self._make_finding(
            rule=rule,
            status=status,
            confidence=0.85 if status == "COMPLIANT" else 0.7,
            evidence=f"Keywords found: {found_keywords}/{keywords}",
            page_numbers=found_pages[:5],
            explanation=f"{len(found_keywords)}/{len(keywords)} expected keywords found in document.",
        )

    def _make_finding(self, rule, status, confidence, evidence, page_numbers, explanation):
        """Create a standardized finding dict."""
        return {
            "rule_id": rule.get("rule_id", "UNKNOWN"),
            "rule_title": rule.get("title", "Unknown Rule"),
            "regulation_ref": rule.get("regulation_reference", "N/A"),
            "source_document": rule.get("source_document", "N/A"),
            "status": status,
            "confidence": confidence,
            "severity": rule.get("severity", "MEDIUM"),
            "check_type": "DETERMINISTIC",
            "evidence": {
                "document_excerpt": evidence,
                "page_numbers": page_numbers,
                "rule_text": rule.get("description", ""),
            },
            "explanation": explanation,
            "recommendation": rule.get("common_violations", ["Review manually"])[0]
                             if status != "COMPLIANT" else "No action needed.",
        }
