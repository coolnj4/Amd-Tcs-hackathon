"""
Semantic Validator — Uses LLM to assess compliance through natural language understanding.
The LLM receives the regulation text and document excerpt, then makes a judgment.

Includes retry logic: if first RAG retrieval is weak, retries with broader query
and falls back to direct section text scan.
"""
import json
import re


SEMANTIC_SYSTEM = """You are a SEBI compliance auditor. Your task is to determine whether a 
Draft Red Herring Prospectus (DRHP) complies with a specific SEBI regulation.

Be precise and evidence-based. Always cite specific text from the document as evidence.

IMPORTANT GUIDELINES:
- If you find ANY relevant content addressing the regulation, make a definitive judgment 
  (COMPLIANT or NON_COMPLIANT). Do NOT default to NEEDS_REVIEW.
- Use NEEDS_REVIEW ONLY when the document sections provided contain absolutely NO text 
  related to the regulation topic. This should be rare for a 400+ page DRHP.
- Partial disclosure is still disclosure — lean toward COMPLIANT with a note if the 
  document addresses the topic even if not perfectly.
- For section-presence checks: if the section exists with relevant content, it is COMPLIANT."""

SEMANTIC_PROMPT = """COMPLIANCE CHECK:

REGULATION: {regulation_ref}
RULE: {rule_title}
DESCRIPTION: {rule_description}

WHAT TO LOOK FOR:
{what_to_look_for}

VALIDATION QUESTION:
{validation_question}

SEBI REGULATION TEXT (from the compliance database):
---
{regulation_text}
---

RELEVANT DOCUMENT SECTIONS (from the DRHP being audited):
---
{document_text}
---

Analyze whether the document complies with this regulation. 

CRITICAL RULES FOR YOUR ASSESSMENT:
1. If the document text contains information relevant to this regulation, you MUST choose 
   COMPLIANT or NON_COMPLIANT — do NOT use NEEDS_REVIEW.
2. NEEDS_REVIEW is ONLY for cases where the retrieved text is completely irrelevant to the 
   regulation (wrong section retrieved).
3. For "section must exist" type checks: if the text above contains content from that 
   section, the document IS compliant.
4. Be specific about which page numbers or sections contain the evidence.

Respond ONLY in JSON:
{{
    "compliant": true or false or null,
    "confidence": 0.0 to 1.0,
    "evidence_found": "Exact quotes from the document that support your assessment (max 300 chars)",
    "evidence_missing": "What should be present but is not (if non-compliant, max 200 chars)",
    "explanation": "Detailed reasoning for your assessment (2-3 sentences)",
    "page_references": "Which pages or sections the evidence was found in",
    "status": "COMPLIANT or NON_COMPLIANT or NEEDS_REVIEW"
}}"""


class SemanticValidator:
    """Runs LLM-powered semantic compliance checks with retrieval retry."""

    def __init__(self, llm_client, embedding_manager):
        """
        Args:
            llm_client: LLMClient instance
            embedding_manager: EmbeddingManager instance for RAG retrieval
        """
        self.llm = llm_client
        self.embeddings = embedding_manager

    def validate(self, rule: dict, parsed_doc, compliance_collection,
                 document_collection) -> dict:
        """
        Run a semantic compliance check using RAG + LLM.
        Includes retry logic for weak retrievals.

        Args:
            rule: Compliance rule dict
            parsed_doc: ParsedDocument object
            compliance_collection: ChromaDB collection for regulations
            document_collection: ChromaDB collection for the DRHP

        Returns:
            Finding dict
        """
        rule_id = rule.get("rule_id", "UNKNOWN")
        validation_question = rule.get("validation_question", "Is this compliant?")

        # Step 1: Retrieve relevant regulation text from compliance RAG
        reg_query = f"{rule.get('regulation_reference', '')} {rule.get('title', '')}"
        reg_results = self.embeddings.query(
            compliance_collection,
            query_text=reg_query,
            top_k=3,
        )
        regulation_text = "\n\n".join([r["text"][:1500] for r in reg_results])

        # Step 2: Retrieve relevant document sections (with retry logic)
        document_text, avg_retrieval_score = self._retrieve_with_retry(
            rule, parsed_doc, document_collection
        )

        if not document_text.strip():
            return self._make_finding(
                rule=rule,
                status="NEEDS_REVIEW",
                confidence=0.3,
                llm_confidence=0.3,
                evidence_found="No relevant document sections found via retrieval.",
                evidence_missing="Could not locate relevant content in the DRHP.",
                explanation="RAG retrieval returned no relevant document sections for this rule.",
                page_references="N/A",
                retrieval_score=0.0,
            )

        # Step 3: Call LLM for semantic assessment
        try:
            response = self.llm.call_json(
                system=SEMANTIC_SYSTEM,
                user=SEMANTIC_PROMPT.format(
                    regulation_ref=rule.get("regulation_reference", ""),
                    rule_title=rule.get("title", ""),
                    rule_description=rule.get("description", ""),
                    what_to_look_for=rule.get("what_to_look_for", "General compliance"),
                    validation_question=validation_question,
                    regulation_text=regulation_text[:3000],
                    document_text=document_text[:5000],
                ),
            )

            status = response.get("status", "NEEDS_REVIEW")
            if status not in ["COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW"]:
                if response.get("compliant") is True:
                    status = "COMPLIANT"
                elif response.get("compliant") is False:
                    status = "NON_COMPLIANT"
                else:
                    status = "NEEDS_REVIEW"

            return self._make_finding(
                rule=rule,
                status=status,
                confidence=response.get("confidence", 0.5),
                llm_confidence=response.get("confidence", 0.5),
                evidence_found=response.get("evidence_found", ""),
                evidence_missing=response.get("evidence_missing", ""),
                explanation=response.get("explanation", "No explanation provided."),
                page_references=response.get("page_references", ""),
                retrieval_score=avg_retrieval_score,
            )

        except Exception as e:
            return self._make_finding(
                rule=rule,
                status="NEEDS_REVIEW",
                confidence=0.3,
                llm_confidence=0.0,
                evidence_found="",
                evidence_missing=str(e),
                explanation=f"LLM assessment failed: {e}",
                page_references="",
                retrieval_score=avg_retrieval_score,
            )

    def _retrieve_with_retry(self, rule, parsed_doc, document_collection):
        """
        Multi-strategy retrieval with fallback:
        1. Section-filtered RAG query using search_keywords
        2. Broader unfiltered RAG query using title + description
        3. Direct section text scan from parsed_doc.sections

        Returns:
            (document_text: str, avg_retrieval_score: float)
        """
        doc_query = " ".join(rule.get("search_keywords", [rule.get("title", "")]))
        applicable_sections = rule.get("applicable_sections", [])

        # Handle cover_page injection
        cover_page_text = ""
        if "cover_page" in applicable_sections:
            cover_pages = parsed_doc.pages[:5] if hasattr(parsed_doc, 'pages') else []
            cover_page_text = "\n\n".join(
                f"[Page {p.page_num}]\n{p.text}" for p in cover_pages if p.text.strip()
            )
            applicable_sections = [s for s in applicable_sections if s != "cover_page"]

        # === Strategy 1: Section-filtered RAG ===
        doc_results = []
        if applicable_sections:
            for section in applicable_sections:
                section_results = self.embeddings.query(
                    document_collection,
                    query_text=doc_query,
                    top_k=3,
                    where={"section": section},
                )
                doc_results.extend(section_results)

        # === Strategy 2: Unfiltered RAG (if strategy 1 was weak) ===
        avg_distance = self._avg_distance(doc_results)
        if not doc_results or avg_distance > 0.55:
            # Broaden the query using title + description
            broad_query = f"{rule.get('title', '')} {rule.get('description', '')}"
            broad_results = self.embeddings.query(
                document_collection,
                query_text=broad_query,
                top_k=8,
            )
            if not doc_results or self._avg_distance(broad_results) < avg_distance:
                doc_results = broad_results

        # === Strategy 3: Direct section text scan (if RAG still weak) ===
        avg_distance = self._avg_distance(doc_results)
        if avg_distance > 0.6 and applicable_sections and hasattr(parsed_doc, 'sections'):
            section_text_parts = []
            for section in parsed_doc.sections:
                if section.name in applicable_sections and section.text:
                    # Take first 3000 chars of each matching section
                    section_text_parts.append(
                        f"[Section: {section.name}, Pages {section.start_page}-{section.end_page}]\n"
                        f"{section.text[:3000]}"
                    )
            if section_text_parts:
                direct_text = "\n\n".join(section_text_parts)
                # Combine with whatever RAG found
                rag_text = "\n\n".join([r["text"][:1500] for r in doc_results[:3]])
                document_text = f"=== DIRECT SECTION SCAN ===\n{direct_text[:4000]}\n\n=== RAG RESULTS ===\n{rag_text}"

                if cover_page_text:
                    document_text = f"=== COVER PAGE ===\n{cover_page_text[:3000]}\n\n{document_text}"

                return document_text, max(0.5, 1.0 - avg_distance)

        # Build final document text from RAG results
        document_text = "\n\n".join([r["text"][:1500] for r in doc_results[:6]])

        if cover_page_text:
            document_text = (
                f"=== COVER PAGE (First 5 Pages) ===\n{cover_page_text[:4000]}\n\n"
                f"=== RELEVANT SECTIONS ===\n{document_text}"
            )

        avg_retrieval_score = 1.0 - self._avg_distance(doc_results)
        return document_text, max(0.0, avg_retrieval_score)

    def _avg_distance(self, results: list) -> float:
        """Compute average distance from RAG results."""
        if not results:
            return 1.0
        return sum(r.get("distance", 0.5) for r in results) / len(results)

    def _make_finding(self, rule, status, confidence, llm_confidence,
                      evidence_found, evidence_missing, explanation,
                      page_references, retrieval_score):
        """Create a standardized finding dict."""
        return {
            "rule_id": rule.get("rule_id", "UNKNOWN"),
            "rule_title": rule.get("title", "Unknown Rule"),
            "regulation_ref": rule.get("regulation_reference", "N/A"),
            "source_document": rule.get("source_document", "N/A"),
            "status": status,
            "confidence": confidence,
            "severity": rule.get("severity", "MEDIUM"),
            "check_type": "SEMANTIC",
            "evidence": {
                "document_excerpt": evidence_found,
                "evidence_missing": evidence_missing,
                "page_numbers": page_references,
                "rule_text": rule.get("description", ""),
            },
            "explanation": explanation,
            "recommendation": rule.get("common_violations", ["Review disclosure adequacy"])[0]
                             if status != "COMPLIANT" else "No action needed.",
            # Extra fields for confidence scoring
            "retrieval_score": retrieval_score,
            "llm_confidence": llm_confidence,
        }
