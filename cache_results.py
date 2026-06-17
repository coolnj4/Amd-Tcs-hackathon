"""
Cache Results — Serialize audit results for instant Streamlit loading.
Run this after completing audits in the notebook to save results as JSON.

Usage (in notebook):
    from cache_results import cache_audit_state
    cache_audit_state(audit_state, metrics_data, output_dir="shared")
"""
import os
import json
from datetime import datetime


def cache_audit_state(state, metrics: dict = None, output_dir: str = "shared"):
    """
    Cache an AuditState object as JSON for the Streamlit dashboard.

    Args:
        state: AuditState object from agent.run_audit()
        metrics: Optional dict from PipelineMetrics.to_dict()
        output_dir: Directory to save cache files
    """
    os.makedirs(output_dir, exist_ok=True)

    company = state.parsed_doc.metadata.get("company_name", "Unknown")
    slug = "".join(c if c.isalnum() else "_" for c in company.lower())[:40]

    # Build the cache payload
    cache = {
        "cached_at": datetime.now().isoformat(),
        "company_name": company,
        "document_name": os.path.basename(state.document_path),
        "document_type": state.parsed_doc.metadata.get("document_type", "DRHP"),
        "overall_score": state.overall_score,
        "score_summary": state.score_summary,
        "findings": state.verified_findings,
        "discarded_findings": state.discarded_findings,
        "audit_trail": state.audit_trail,
        "report": state.report,
        "document_stats": {
            "total_pages": state.parsed_doc.total_pages,
            "tables_extracted": state.parsed_doc.tables_count,
            "ocr_pages": state.parsed_doc.ocr_pages_count,
            "sections_detected": len(state.parsed_doc.sections),
            "chunks_created": len(state.raw_findings),
        },
        "errors": state.errors,
    }

    if metrics:
        cache["metrics"] = metrics

    filepath = os.path.join(output_dir, f"audit_cache_{slug}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, default=str)

    print(f"  💾 Cached audit results to: {filepath}")
    return filepath


def load_cached_audits(cache_dir: str = "shared") -> list:
    """
    Load all cached audit results from a directory.

    Returns:
        List of dicts, one per cached audit
    """
    results = []
    if not os.path.isdir(cache_dir):
        return results

    for filename in sorted(os.listdir(cache_dir)):
        if filename.startswith("audit_cache_") and filename.endswith(".json"):
            filepath = os.path.join(cache_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_cache_file"] = filepath
                results.append(data)
            except Exception as e:
                print(f"  ⚠️ Failed to load {filename}: {e}")

    return results


def create_demo_cache(output_dir: str = "shared"):
    """
    Create demo cache from existing report JSON files.
    Use this if you already have audit_report_*.json files.
    """
    reports_dir = os.path.join(os.environ.get("AUDIT_BASE_DIR", "."), "reports")
    if not os.path.isdir(reports_dir):
        print(f"  ⚠️ No reports directory found at {reports_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(reports_dir):
        if filename.startswith("audit_report_") and filename.endswith(".json"):
            filepath = os.path.join(reports_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    report = json.load(f)

                company = report.get("company_name", "Unknown")
                slug = "".join(c if c.isalnum() else "_" for c in company.lower())[:40]

                cache = {
                    "cached_at": datetime.now().isoformat(),
                    "company_name": company,
                    "document_name": report.get("document_name", ""),
                    "document_type": report.get("document_type", "DRHP"),
                    "overall_score": report.get("overall_compliance_score", 0),
                    "score_summary": report.get("score_summary", {}),
                    "findings": report.get("findings", []),
                    "discarded_findings": report.get("discarded_findings", []),
                    "audit_trail": report.get("audit_trail", []),
                    "report": report,
                    "document_stats": report.get("document_stats", {}),
                    "errors": report.get("errors", []),
                }

                out_path = os.path.join(output_dir, f"audit_cache_{slug}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, default=str)

                print(f"  ✅ Created cache: {out_path}")

            except Exception as e:
                print(f"  ❌ Failed to process {filename}: {e}")
