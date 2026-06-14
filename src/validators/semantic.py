"""
Semantic Validator — Uses LLM to assess compliance through natural language understanding.
The LLM receives the regulation text and document excerpt, then makes a judgment.
"""
import json


SEMANTIC_SYSTEM = """You are a SEBI compliance auditor. Your task is to determine whether a 
Draft Red Herring Prospectus (DRHP) complies with a specific SEBI regulation.

Be precise and evidence-based. Always cite specific text from the document as evidence.
If you cannot determine compliance from the available text, say so clearly."""

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
Respond ONLY in JSON:
{{
    "compliant": true or false or null,
    "confidence": 0.0 to 1.0,
    "evidence_found": "Exact quotes from the document that support your assessment (max 300 chars)",
    "evidence_missing": "What should be present but is not (if non-compliant, max 200 chars)",
    "explanation": "Detailed reasoning for your assessment (2-3 sentences)",
    "page_references": "Which pages or sections the evidence was found in",
    "status": "COMPLIANT or NON_COMPLIANT or NEEDS_REVIEW"
}}

Use "null" for compliant and NEEDS_REVIEW for status if you genuinely cannot determine compliance from the available text."""


class SemanticValidator:
    """Runs LLM-powered semantic compliance checks."""

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

        # Step 2: Retrieve relevant document sections from document RAG
        doc_query = " ".join(rule.get("search_keywords", [rule.get("title", "")]))
        applicable_sections = rule.get("applicable_sections", [])

        # Try section-filtered retrieval first
        doc_results = []
        if applicable_sections:
            for section in applicable_sections:
                section_results = self.embeddings.query(
                    document_collection,
                    query_text=doc_query,
                    top_k=2,
                    where={"section": section},
                )
                doc_results.extend(section_results)

        # If no filtered results, do unfiltered retrieval
        if not doc_results:
            doc_results = self.embeddings.query(
                document_collection,
                query_text=doc_query,
                top_k=5,
            )

        document_text = "\n\n".join([r["text"][:1500] for r in doc_results[:5]])

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
        avg_retrieval_score = 1.0 - (
            sum(r.get("distance", 0.5) for r in doc_results[:5]) / max(len(doc_results[:5]), 1)
        )

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
                    document_text=document_text[:4000],
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
