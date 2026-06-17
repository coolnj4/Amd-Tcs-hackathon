"""
Critic Agent — Self-reflection layer that reviews each finding to catch false positives.
A 'senior auditor' reviews the 'junior auditor's work.
"""
import json


CRITIC_SYSTEM = """You are a SENIOR SEBI compliance auditor reviewing a junior auditor's findings.
Your job is to verify findings are accurate and not false positives.
Be critical but fair. Only reject findings with clear reasoning.

IMPORTANT: If a finding is marked NEEDS_REVIEW but you can see evidence in the document 
that addresses the regulation, upgrade your verdict to VALID (if compliant) or keep it as 
VALID with a note (if non-compliant). The goal is to REDUCE ambiguity, not preserve it."""

CRITIC_PROMPT = """A junior auditor flagged this compliance finding during a DRHP review.
Your job is to verify whether it's correct or a false positive.

FINDING DETAILS:
- Rule ID: {rule_id}
- Rule: {rule_title}
- Regulation: {regulation_ref}
- Status: {status}
- Confidence: {confidence}
- Check Type: {check_type}

EVIDENCE FROM DOCUMENT:
{document_evidence}

JUNIOR AUDITOR'S EXPLANATION:
{explanation}

REGULATION TEXT:
{rule_text}

BROADER DOCUMENT CONTEXT:
{broader_context}

Critically evaluate:
1. Is the evidence actually relevant to this specific regulation?
2. Could the document be compliant in a way the junior auditor missed?
3. Is there ambiguity in the regulation that makes this judgment uncertain?
4. For DETERMINISTIC checks: is the pattern matching correct?
5. For SEMANTIC checks: is the LLM interpretation reasonable?
6. If the status is NEEDS_REVIEW: can you make a definitive COMPLIANT/NON_COMPLIANT judgment from the broader context?

IMPORTANT: If the junior auditor marked this as NEEDS_REVIEW but the broader document 
context shows relevant content, you should return VALID and note that the document does 
address this regulation. Reduce ambiguity wherever possible.

Respond ONLY in JSON:
{{
    "verdict": "VALID or FALSE_POSITIVE or NEEDS_MORE_EVIDENCE",
    "revised_status": "COMPLIANT or NON_COMPLIANT or NEEDS_REVIEW",
    "revised_confidence": 0.0 to 1.0,
    "reasoning": "2-3 sentences explaining your review decision",
    "missed_evidence": "Any evidence the junior auditor missed (or empty string)",
    "recommendation": "Keep finding | Discard finding | Retry with broader context"
}}"""


class CriticAgent:
    """Reviews findings from the validation engine to catch false positives."""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.stats = {"valid": 0, "false_positive": 0, "needs_more": 0, "errors": 0}

    def review_findings(self, findings: list, parsed_doc) -> dict:
        """
        Review all findings and classify as valid, false positive, or needs more evidence.

        Args:
            findings: List of finding dicts from validators
            parsed_doc: ParsedDocument for broader context

        Returns:
            Dict with 'verified', 'discarded', and 'retry' lists
        """
        verified = []
        discarded = []
        retry = []

        print(f"\n  🔍 Critic reviewing {len(findings)} findings...")

        for i, finding in enumerate(findings):
            status = finding.get("status", "")

            # Skip compliant findings (no need to verify passes)
            if status == "COMPLIANT":
                verified.append(finding)
                self.stats["valid"] += 1
                continue

            # Skip deterministic findings with 100% confidence (binary checks are reliable)
            if finding.get("check_type") == "DETERMINISTIC" and finding.get("confidence", 0) >= 0.99:
                verified.append(finding)
                self.stats["valid"] += 1
                continue

            # Review non-compliant and needs-review findings
            try:
                review_result = self._review_single(finding, parsed_doc)

                if review_result["verdict"] == "VALID":
                    finding["critic_confidence"] = review_result["revised_confidence"]
                    finding["critic_reasoning"] = review_result["reasoning"]
                    finding["critic_reviewed"] = True

                    # Allow critic to upgrade NEEDS_REVIEW → COMPLIANT/NON_COMPLIANT
                    revised_status = review_result.get("revised_status", "")
                    if revised_status in ("COMPLIANT", "NON_COMPLIANT") and finding.get("status") == "NEEDS_REVIEW":
                        finding["status"] = revised_status
                        finding["confidence"] = max(finding.get("confidence", 0.5),
                                                    review_result["revised_confidence"])

                    verified.append(finding)
                    self.stats["valid"] += 1
                    print(f"    ✅ [{finding['rule_id']}] Verified — {review_result['reasoning'][:60]}...")

                elif review_result["verdict"] == "FALSE_POSITIVE":
                    finding["discard_reason"] = review_result["reasoning"]
                    finding["critic_reviewed"] = True
                    discarded.append(finding)
                    self.stats["false_positive"] += 1
                    print(f"    🗑️  [{finding['rule_id']}] Discarded — {review_result['reasoning'][:60]}...")

                elif review_result["verdict"] == "NEEDS_MORE_EVIDENCE":
                    finding["critic_note"] = review_result["reasoning"]
                    finding["critic_reviewed"] = True
                    # Downgrade confidence rather than retry (saves time)
                    finding["confidence"] = min(finding.get("confidence", 0.5), 0.5)
                    finding["critic_confidence"] = review_result["revised_confidence"]

                    # Allow critic to override status even on NEEDS_MORE_EVIDENCE
                    revised_status = review_result.get("revised_status", "")
                    if revised_status in ("COMPLIANT", "NON_COMPLIANT"):
                        finding["status"] = revised_status
                    else:
                        finding["status"] = "NEEDS_REVIEW"

                    verified.append(finding)
                    self.stats["needs_more"] += 1
                    print(f"    ⚠️  [{finding['rule_id']}] Needs Review — {review_result['reasoning'][:60]}...")

            except Exception as e:
                print(f"    ❌ [{finding.get('rule_id', '?')}] Critic error: {e}")
                finding["critic_reviewed"] = False
                verified.append(finding)  # Keep finding if critic fails
                self.stats["errors"] += 1

        print(f"\n  📊 Critic Summary: "
              f"✅ {self.stats['valid']} valid, "
              f"🗑️ {self.stats['false_positive']} false positives caught, "
              f"⚠️ {self.stats['needs_more']} downgraded")

        return {
            "verified": verified,
            "discarded": discarded,
            "retry": retry,
        }

    def _review_single(self, finding: dict, parsed_doc) -> dict:
        """Review a single finding using the LLM critic."""
        # Get broader context from the document around the evidence pages
        evidence = finding.get("evidence", {})
        page_nums = evidence.get("page_numbers", [])

        broader_context = self._get_broader_context(parsed_doc, page_nums)

        response = self.llm.call_json(
            system=CRITIC_SYSTEM,
            user=CRITIC_PROMPT.format(
                rule_id=finding.get("rule_id", ""),
                rule_title=finding.get("rule_title", ""),
                regulation_ref=finding.get("regulation_ref", ""),
                status=finding.get("status", ""),
                confidence=finding.get("confidence", 0),
                check_type=finding.get("check_type", ""),
                document_evidence=evidence.get("document_excerpt", "N/A"),
                explanation=finding.get("explanation", ""),
                rule_text=evidence.get("rule_text", ""),
                broader_context=broader_context[:3000],
            ),
        )

        # Validate response
        verdict = response.get("verdict", "VALID")
        if verdict not in ["VALID", "FALSE_POSITIVE", "NEEDS_MORE_EVIDENCE"]:
            verdict = "VALID"  # Default to keeping finding

        return {
            "verdict": verdict,
            "revised_status": response.get("revised_status", ""),
            "revised_confidence": response.get("revised_confidence", finding.get("confidence", 0.5)),
            "reasoning": response.get("reasoning", "No reasoning provided"),
            "missed_evidence": response.get("missed_evidence", ""),
            "recommendation": response.get("recommendation", "Keep finding"),
        }

    def _get_broader_context(self, parsed_doc, page_nums, window: int = 2) -> str:
        """Get text from pages around the evidence location."""
        if isinstance(page_nums, str):
            # Try to parse page numbers from string
            import re
            nums = re.findall(r'\d+', str(page_nums))
            page_nums = [int(n) for n in nums[:5]]

        if not page_nums or not isinstance(page_nums, list):
            # Return first few pages as fallback
            return "\n\n".join(p.text[:500] for p in parsed_doc.pages[:3])

        context_pages = set()
        for pn in page_nums:
            for offset in range(-window, window + 1):
                context_pages.add(pn + offset)

        parts = []
        for page in parsed_doc.pages:
            if page.page_num in context_pages:
                parts.append(f"[Page {page.page_num}]\n{page.text[:800]}")

        return "\n\n".join(parts) if parts else "No context available."

    def get_stats(self) -> dict:
        """Return critic statistics."""
        return self.stats.copy()
