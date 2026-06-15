"""
Report Generator — Creates PDF audit reports from findings.
"""
import os
import json
from datetime import datetime

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False
    print("[WARN] fpdf2 not installed. PDF report generation disabled.")


class AuditReportPDF(FPDF if HAS_FPDF else object):
    """Custom PDF report for compliance audit results."""

    def __init__(self):
        if not HAS_FPDF:
            raise ImportError("fpdf2 required: pip install fpdf2")
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'AI-Driven Audit & Compliance Report', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 8)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')


def generate_pdf_report(report: dict, output_path: str) -> str:
    """
    Generate a PDF audit report from the report dict.

    Args:
        report: Report dict from AuditAgent
        output_path: Path to save the PDF

    Returns:
        Path to the generated PDF
    """
    if not HAS_FPDF:
        print("[WARN] Cannot generate PDF — fpdf2 not installed")
        return ""

    pdf = AuditReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ─── Executive Summary ─────────────────────
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Executive Summary', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font('Helvetica', '', 10)
    company = report.get('company_name', 'Unknown')
    doc_name = report.get('document_name', 'Unknown')
    score = report.get('overall_compliance_score', 0)
    summary = report.get('score_summary', {})

    pdf.cell(0, 6, f'Company: {company}', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f'Document: {doc_name}', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f'Audit Date: {report.get("audit_date", "N/A")}', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Score
    pdf.set_font('Helvetica', 'B', 16)
    score_color = (46, 125, 50) if score >= 0.7 else (255, 152, 0) if score >= 0.5 else (211, 47, 47)
    pdf.set_text_color(*score_color)
    pdf.cell(0, 10, f'Overall Compliance Score: {score:.0%}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Summary stats
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Rules Checked: {summary.get("total_rules_checked", 0)}', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f'Compliant: {summary.get("compliant", 0)}', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f'Non-Compliant: {summary.get("non_compliant", 0)}', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f'Needs Review: {summary.get("needs_review", 0)}', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f'Average Confidence: {summary.get("average_confidence", 0):.0%}', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ─── Findings Detail ─────────────────────
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Detailed Findings', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    findings = report.get('findings', [])

    # Non-compliant first, then needs-review, then compliant
    sort_order = {"NON_COMPLIANT": 0, "NEEDS_REVIEW": 1, "COMPLIANT": 2}
    sorted_findings = sorted(findings, key=lambda f: sort_order.get(f.get("status", ""), 3))

    for i, finding in enumerate(sorted_findings, 1):
        status = finding.get('status', 'UNKNOWN')
        status_icon = {'COMPLIANT': '[PASS]', 'NON_COMPLIANT': '[FAIL]', 'NEEDS_REVIEW': '[REVIEW]'}.get(status, '[?]')

        # Status color
        if status == "NON_COMPLIANT":
            pdf.set_text_color(211, 47, 47)
        elif status == "NEEDS_REVIEW":
            pdf.set_text_color(255, 152, 0)
        else:
            pdf.set_text_color(46, 125, 50)

        pdf.set_font('Helvetica', 'B', 10)
        title = f"{i}. {status_icon} {finding.get('rule_title', 'Unknown')}"
        pdf.cell(0, 7, _safe(title[:90]), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 5, f"   Rule: {finding.get('rule_id', '')} | Regulation: {finding.get('regulation_ref', '')}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"   Severity: {finding.get('severity', '')} | Confidence: {finding.get('confidence', 0):.0%} | "
                       f"Type: {finding.get('check_type', '')}",
                 new_x="LMARGIN", new_y="NEXT")

        # Explanation
        explanation = finding.get('explanation', '')
        if explanation:
            pdf.set_font('Helvetica', 'I', 9)
            try:
                pdf.multi_cell(0, 5, _safe(f" {explanation[:300]}"))
            except Exception:
                pdf.cell(0, 5, _safe(explanation[:80]), new_x="LMARGIN", new_y="NEXT")

        # Evidence
        evidence = finding.get('evidence', {})
        excerpt = evidence.get('document_excerpt', '')
        if excerpt and status != "COMPLIANT":
            pdf.set_font('Helvetica', '', 8)
            try:
                pdf.multi_cell(0, 4, _safe(f" Evidence: {excerpt[:200]}"))
            except Exception:
                pdf.cell(0, 4, _safe(f" Evidence: {excerpt[:70]}"), new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)

    # ─── Discarded Findings ─────────────────────
    discarded = report.get('discarded_findings', [])
    if discarded:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, f'False Positives Caught by Critic ({len(discarded)})', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        pdf.set_font('Helvetica', '', 9)
        for d in discarded:
            pdf.cell(0, 5, _safe(f"- {d.get('rule_id', '')}: {d.get('rule_title', '')}"),
                     new_x="LMARGIN", new_y="NEXT")
            reason = d.get('discard_reason', 'No reason provided')
            pdf.set_font('Helvetica', 'I', 8)
            try:
                pdf.multi_cell(0, 4, _safe(f" {reason[:200]}"))
            except Exception:
                pdf.cell(0, 4, _safe(reason[:80]), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Helvetica', '', 9)
            pdf.ln(2)

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    pdf.output(output_path)
    print(f"  📄 PDF report saved to: {output_path}")
    return output_path


def _safe(text: str) -> str:
    """Make text safe for FPDF (handle encoding, control chars, long strings)."""
    import re
    # Remove control characters except newline/tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Replace tabs with spaces
    text = text.replace('\t', '  ')
    # Break very long unbreakable words (>60 chars) by inserting spaces
    text = re.sub(r'(\S{60})', r'\1 ', text)
    # Encode to latin-1 safely
    return text.encode('latin-1', errors='replace').decode('latin-1')


def print_findings_table(findings: list):
    """Print findings as a formatted table to console."""
    print(f"\n{'='*90}")
    print(f"  {'#':>3}  {'Status':<14}  {'Conf':>5}  {'Severity':<9}  {'Type':<14}  {'Rule'}")
    print(f"  {'-'*3}  {'-'*14}  {'-'*5}  {'-'*9}  {'-'*14}  {'-'*30}")

    sort_order = {"NON_COMPLIANT": 0, "NEEDS_REVIEW": 1, "COMPLIANT": 2}
    sorted_findings = sorted(findings, key=lambda f: sort_order.get(f.get("status", ""), 3))

    for i, f in enumerate(sorted_findings, 1):
        status = f.get("status", "?")
        icon = {"COMPLIANT": "✅", "NON_COMPLIANT": "❌", "NEEDS_REVIEW": "⚠️"}.get(status, "❓")
        conf = f.get("confidence", 0)
        severity = f.get("severity", "?")
        check_type = f.get("check_type", "?")
        title = f.get("rule_title", "?")[:35]

        print(f"  {i:3d}  {icon} {status:<12}  {conf:5.0%}  {severity:<9}  {check_type:<14}  {title}")

    print(f"{'='*90}")
