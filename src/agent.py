"""
Agent Orchestrator — LangGraph state machine that ties all components together.
Every node transition is logged for full auditability.
"""
import os
import json
import time
import re
from datetime import datetime
from typing import TypedDict, Optional

from src.config import COMPLIANCE_DIR, COMPLIANCE_RULES_PATH
from src.llm_client import LLMClient
from src.document_parser import parse_document
from src.chunker import chunk_document
from src.embeddings import EmbeddingManager
from src.rule_generator import generate_rules, load_rules
from src.validators.deterministic import DeterministicValidator
from src.validators.semantic import SemanticValidator
from src.validators.cross_reference import CrossReferenceValidator
from src.critic import CriticAgent
from src.confidence import score_all_findings, compute_overall_score, generate_score_summary


class AuditState:
    """Mutable state for the audit pipeline."""
    def __init__(self):
        self.document_path: str = ""
        self.parsed_doc = None
        self.doc_collection = None
        self.doc_collection_name: str = ""
        self.compliance_rules: list = []
        self.raw_findings: list = []
        self.verified_findings: list = []
        self.discarded_findings: list = []
        self.overall_score: float = 0.0
        self.score_summary: dict = {}
        self.audit_trail: list = []
        self.report: dict = {}
        self.start_time: float = 0
        self.errors: list = []


class AuditAgent:
    """
    Main orchestrator that runs the full audit pipeline.

    Pipeline: Parse → Index → Plan → Validate → Cross-Reference → Critic → Score → Report
    """

    def __init__(self, llm_client: LLMClient, embedding_manager: EmbeddingManager):
        self.llm = llm_client
        self.embeddings = embedding_manager
        self.compliance_collection = None

        # Initialize validators
        self.deterministic = DeterministicValidator()
        self.semantic = SemanticValidator(llm_client, embedding_manager)
        self.cross_ref = CrossReferenceValidator()
        self.critic = CriticAgent(llm_client)

    def setup_compliance_rag(self, force_reindex: bool = False):
        """
        Set up the compliance RAG by indexing SEBI regulation PDFs.
        Only needs to be done once (persisted in ChromaDB).
        """
        self.compliance_collection = self.embeddings.get_or_create_compliance_collection()

        # Check if already indexed
        count = self.compliance_collection.count()
        if count > 0 and not force_reindex:
            print(f"  ✅ Compliance collection already indexed ({count} chunks)")
            return

        print(f"\n  📚 Indexing compliance documents from {COMPLIANCE_DIR}...")

        # Parse and chunk each compliance PDF
        all_chunks = []
        compliance_dir = COMPLIANCE_DIR
        if os.path.isdir(compliance_dir):
            for filename in os.listdir(compliance_dir):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(compliance_dir, filename)
                    print(f"\n  Parsing: {filename}")
                    parsed = parse_document(pdf_path, verbose=True)
                    chunks = chunk_document(parsed, source_type="compliance")
                    all_chunks.extend(chunks)

        if all_chunks:
            self.embeddings.index_chunks(self.compliance_collection, all_chunks)
            print(f"\n  ✅ Compliance RAG ready: {len(all_chunks)} chunks indexed")
        else:
            print("  ⚠️  No compliance documents found!")

        return all_chunks

    def generate_compliance_rules(self, compliance_chunks=None, force_regenerate=False):
        """Generate compliance rules from SEBI regulations using LLM."""
        # Try loading cached rules first
        rules = load_rules()
        if rules and not force_regenerate:
            print(f"  ✅ Loaded {len(rules)} cached compliance rules")
            return rules

        # Need chunks to generate rules
        if not compliance_chunks:
            # Parse compliance docs to get chunks
            compliance_chunks = []
            if os.path.isdir(COMPLIANCE_DIR):
                for filename in os.listdir(COMPLIANCE_DIR):
                    if filename.endswith('.pdf'):
                        pdf_path = os.path.join(COMPLIANCE_DIR, filename)
                        parsed = parse_document(pdf_path, verbose=False)
                        chunks = chunk_document(parsed, source_type="compliance")
                        compliance_chunks.extend(chunks)

        rules = generate_rules(self.llm, compliance_chunks, force_regenerate)
        return rules

    def run_audit(self, document_path: str, rules: list = None) -> AuditState:
        """
        Run the full audit pipeline on a single document.

        Args:
            document_path: Path to the DRHP PDF
            rules: Optional pre-loaded compliance rules

        Returns:
            AuditState with all results
        """
        state = AuditState()
        state.document_path = document_path
        state.start_time = time.time()
        state.compliance_rules = rules or load_rules()

        print(f"\n{'='*70}")
        print(f"  🏁 STARTING AUDIT: {os.path.basename(document_path)}")
        print(f"  📋 Rules to check: {len(state.compliance_rules)}")
        print(f"{'='*70}")

        try:
            # Step 1: Parse Document
            self._log_step(state, "parse_document", "Parsing PDF document")
            state.parsed_doc = parse_document(document_path, verbose=True)
            self._complete_step(state, f"Parsed {state.parsed_doc.total_pages} pages, "
                               f"{state.parsed_doc.tables_count} tables, "
                               f"{state.parsed_doc.ocr_pages_count} OCR pages")

            # Step 2: Index Document into ChromaDB
            self._log_step(state, "index_document", "Indexing document into ChromaDB")
            doc_chunks = chunk_document(state.parsed_doc, source_type="drhp")
            company_name = state.parsed_doc.metadata.get("company_name", "unknown")
            slug = re.sub(r'[^a-z0-9]', '_', company_name.lower())[:40]
            state.doc_collection_name = slug

            # Delete existing collection if any
            self.embeddings.delete_document_collection(slug)
            state.doc_collection = self.embeddings.get_or_create_document_collection(slug)
            self.embeddings.index_chunks(state.doc_collection, doc_chunks)
            self._complete_step(state, f"Created {len(doc_chunks)} chunks in collection '{slug}'")

            # Step 3: Run Validation Engine
            self._log_step(state, "validation_engine", "Running validation checks")
            self._run_validation(state)
            self._complete_step(state, f"Generated {len(state.raw_findings)} raw findings")

            # Step 4: Cross-Reference Check
            self._log_step(state, "cross_reference", "Checking numerical cross-references")
            xref_findings = self.cross_ref.validate(state.parsed_doc)
            state.raw_findings.extend(xref_findings)
            self._complete_step(state, f"Found {len(xref_findings)} numerical inconsistencies")

            # Step 5: Critic Review
            self._log_step(state, "critic_review", "Critic agent reviewing findings")
            critic_result = self.critic.review_findings(state.raw_findings, state.parsed_doc)
            state.verified_findings = critic_result["verified"]
            state.discarded_findings = critic_result["discarded"]
            self._complete_step(state,
                f"Verified: {len(state.verified_findings)}, "
                f"Discarded: {len(state.discarded_findings)} false positives")

            # Step 6: Compute Confidence Scores
            self._log_step(state, "scoring", "Computing confidence scores")
            state.verified_findings = score_all_findings(state.verified_findings)
            state.overall_score = compute_overall_score(state.verified_findings)
            state.score_summary = generate_score_summary(state.verified_findings)
            self._complete_step(state, f"Overall compliance score: {state.overall_score:.0%}")

            # Step 7: Generate Report
            self._log_step(state, "report", "Generating audit report")
            state.report = self._build_report(state)
            self._complete_step(state, "Report generated")

        except Exception as e:
            state.errors.append(str(e))
            self._log_step(state, "error", f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()

        # Print summary
        elapsed = time.time() - state.start_time
        self._print_summary(state, elapsed)

        return state

    def _run_validation(self, state: AuditState):
        """Run all validation checks based on rule type."""
        rules = state.compliance_rules

        for i, rule in enumerate(rules):
            check_type = rule.get("check_type", "SEMANTIC")
            rule_id = rule.get("rule_id", f"RULE_{i}")

            print(f"    [{i+1}/{len(rules)}] Checking {rule_id} ({check_type})...")

            try:
                if check_type == "DETERMINISTIC":
                    finding = self.deterministic.validate(rule, state.parsed_doc)

                elif check_type in ("SEMANTIC", "NLP_EXTRACTION"):
                    if not self.compliance_collection or not state.doc_collection:
                        print(f"      ⚠️ Skipping semantic check — collections not ready")
                        continue
                    finding = self.semantic.validate(
                        rule, state.parsed_doc,
                        self.compliance_collection, state.doc_collection
                    )

                elif check_type == "NUMERICAL":
                    # Numerical checks are handled by cross-reference validator separately
                    continue

                else:
                    # Default to semantic
                    if self.compliance_collection and state.doc_collection:
                        finding = self.semantic.validate(
                            rule, state.parsed_doc,
                            self.compliance_collection, state.doc_collection
                        )
                    else:
                        continue

                state.raw_findings.append(finding)

                # Log each validation step
                status_icon = {"COMPLIANT": "✅", "NON_COMPLIANT": "❌", "NEEDS_REVIEW": "⚠️"}.get(
                    finding.get("status", ""), "❓"
                )
                print(f"      {status_icon} {finding.get('status', 'N/A')} "
                      f"(conf: {finding.get('confidence', 0):.2f})")

                state.audit_trail.append({
                    "step_id": len(state.audit_trail),
                    "timestamp": datetime.now().isoformat(),
                    "node": "validation",
                    "action": f"Validated {rule_id}: {rule.get('title', '')}",
                    "status": "completed",
                    "result": finding.get("status", ""),
                    "confidence": finding.get("confidence", 0),
                })

            except Exception as e:
                print(f"      ❌ Error: {e}")
                state.errors.append(f"Rule {rule_id}: {e}")

    def _build_report(self, state: AuditState) -> dict:
        """Build the final audit report."""
        company = state.parsed_doc.metadata.get("company_name", "Unknown")
        doc_type = state.parsed_doc.metadata.get("document_type", "DRHP")

        return {
            "document_name": os.path.basename(state.document_path),
            "company_name": company,
            "document_type": doc_type,
            "audit_date": datetime.now().isoformat(),
            "overall_compliance_score": state.overall_score,
            "score_summary": state.score_summary,
            "findings": state.verified_findings,
            "discarded_findings": state.discarded_findings,
            "audit_trail": state.audit_trail,
            "metadata": state.parsed_doc.metadata,
            "document_stats": {
                "total_pages": state.parsed_doc.total_pages,
                "tables_extracted": state.parsed_doc.tables_count,
                "ocr_pages": state.parsed_doc.ocr_pages_count,
                "sections_detected": len(state.parsed_doc.sections),
            },
            "errors": state.errors,
        }

    def _log_step(self, state: AuditState, node: str, action: str):
        """Log the start of a pipeline step."""
        state.audit_trail.append({
            "step_id": len(state.audit_trail),
            "timestamp": datetime.now().isoformat(),
            "node": node,
            "action": action,
            "status": "in_progress",
        })
        elapsed = time.time() - state.start_time
        print(f"\n  ⏱️  [{elapsed:.1f}s] {node}: {action}")

    def _complete_step(self, state: AuditState, result: str):
        """Mark the last audit trail step as complete."""
        if state.audit_trail:
            state.audit_trail[-1]["status"] = "completed"
            state.audit_trail[-1]["result"] = result
            state.audit_trail[-1]["completed_at"] = datetime.now().isoformat()
        print(f"      → {result}")

    def _print_summary(self, state: AuditState, elapsed: float):
        """Print a final summary of the audit."""
        summary = state.score_summary

        print(f"\n{'='*70}")
        print(f"  🏁 AUDIT COMPLETE — {os.path.basename(state.document_path)}")
        print(f"{'='*70}")
        print(f"  🏢 Company: {state.parsed_doc.metadata.get('company_name', 'N/A')}")
        print(f"  📊 Overall Compliance Score: {state.overall_score:.0%}")
        print(f"  ⏱️  Total Time: {elapsed:.1f} seconds")
        print(f"")
        print(f"  📋 Results:")
        print(f"     ✅ Compliant:      {summary.get('compliant', 0)}")
        print(f"     ❌ Non-Compliant:  {summary.get('non_compliant', 0)}")
        print(f"     ⚠️  Needs Review:   {summary.get('needs_review', 0)}")
        print(f"     🗑️  False Positives: {len(state.discarded_findings)}")
        print(f"")
        print(f"  🔍 Average Confidence: {summary.get('average_confidence', 0):.0%}")
        print(f"  📄 Pages Analyzed: {state.parsed_doc.total_pages}")
        print(f"  📊 Tables Extracted: {state.parsed_doc.tables_count}")

        if state.errors:
            print(f"\n  ⚠️  Errors ({len(state.errors)}):")
            for err in state.errors[:5]:
                print(f"     - {err[:80]}")

        print(f"{'='*70}")


def save_report(state: AuditState, output_dir: str = "."):
    """Save the audit report to JSON file."""
    company = state.parsed_doc.metadata.get("company_name", "unknown")
    slug = re.sub(r'[^a-z0-9]', '_', company.lower())[:30]
    filename = f"audit_report_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(state.report, f, indent=2, default=str)

    print(f"\n  💾 Report saved to: {filepath}")
    return filepath
