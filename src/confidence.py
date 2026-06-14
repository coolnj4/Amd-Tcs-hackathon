"""
Confidence Scorer — Computes multi-signal confidence scores for each finding.
"""
from src.config import CONFIDENCE_WEIGHTS, SEVERITY_WEIGHTS


def compute_finding_confidence(finding: dict) -> float:
    """
    Compute final confidence score for a finding using multi-signal weighting.

    Signals:
    - retrieval_score: How similar was the retrieved content (from RAG)
    - evidence_density: How much evidence was found
    - llm_confidence: LLM's self-reported confidence
    - critic_confidence: Critic agent's revised confidence
    - cross_validation: Did multiple check methods agree

    Args:
        finding: Finding dict

    Returns:
        Confidence score (0.0 to 1.0)
    """
    # Deterministic checks are always 1.0 confidence
    if finding.get("check_type") == "DETERMINISTIC":
        return finding.get("confidence", 1.0)

    # Cross-reference checks are high confidence by nature
    if finding.get("check_type") == "CROSS_REFERENCE":
        return finding.get("confidence", 0.90)

    # Multi-signal scoring for semantic and other checks
    retrieval = finding.get("retrieval_score", 0.5)
    evidence = _compute_evidence_density(finding)
    llm_conf = finding.get("llm_confidence", finding.get("confidence", 0.5))
    critic_conf = finding.get("critic_confidence", llm_conf)
    cross_val = 1.0 if finding.get("cross_validated") else 0.6

    w = CONFIDENCE_WEIGHTS
    score = (
        w["retrieval"] * retrieval +
        w["evidence"] * evidence +
        w["llm"] * llm_conf +
        w["critic"] * critic_conf +
        w["cross_validation"] * cross_val
    )

    return round(max(0.0, min(1.0, score)), 2)


def _compute_evidence_density(finding: dict) -> float:
    """Estimate evidence density from finding."""
    evidence = finding.get("evidence", {})

    score = 0.0

    # Has document excerpt?
    excerpt = evidence.get("document_excerpt", "")
    if excerpt and len(excerpt) > 50:
        score += 0.4
    elif excerpt:
        score += 0.2

    # Has page numbers?
    pages = evidence.get("page_numbers", [])
    if pages:
        score += 0.3

    # Has rule text?
    rule_text = evidence.get("rule_text", "")
    if rule_text and len(rule_text) > 20:
        score += 0.3

    return min(1.0, score)


def compute_overall_score(findings: list) -> float:
    """
    Compute overall document compliance score.

    Uses severity-weighted scoring:
    - CRITICAL rules have 3x weight
    - HIGH rules have 2x weight
    - MEDIUM rules have 1x weight
    - LOW rules have 0.5x weight

    Args:
        findings: List of finding dicts

    Returns:
        Overall compliance score (0.0 to 1.0)
    """
    if not findings:
        return 0.0

    compliant_weight = 0.0
    total_weight = 0.0

    for f in findings:
        severity = f.get("severity", "MEDIUM")
        weight = SEVERITY_WEIGHTS.get(severity, 1.0)
        total_weight += weight

        if f.get("status") == "COMPLIANT":
            compliant_weight += weight
        elif f.get("status") == "NEEDS_REVIEW":
            compliant_weight += weight * 0.5  # Partial credit

    if total_weight == 0:
        return 0.0

    return round(compliant_weight / total_weight, 2)


def score_all_findings(findings: list) -> list:
    """
    Compute and update confidence scores for all findings.

    Args:
        findings: List of finding dicts (modified in place)

    Returns:
        Same list with updated confidence scores
    """
    for finding in findings:
        finding["confidence"] = compute_finding_confidence(finding)

    return findings


def generate_score_summary(findings: list) -> dict:
    """
    Generate a summary of scores and statistics.

    Args:
        findings: List of finding dicts

    Returns:
        Summary dict with counts, scores, and breakdowns
    """
    total = len(findings)
    compliant = sum(1 for f in findings if f.get("status") == "COMPLIANT")
    non_compliant = sum(1 for f in findings if f.get("status") == "NON_COMPLIANT")
    needs_review = sum(1 for f in findings if f.get("status") == "NEEDS_REVIEW")

    overall_score = compute_overall_score(findings)
    avg_confidence = sum(f.get("confidence", 0) for f in findings) / max(total, 1)

    # By severity
    by_severity = {}
    for f in findings:
        sev = f.get("severity", "UNKNOWN")
        if sev not in by_severity:
            by_severity[sev] = {"compliant": 0, "non_compliant": 0, "needs_review": 0}
        status = f.get("status", "UNKNOWN")
        if status == "COMPLIANT":
            by_severity[sev]["compliant"] += 1
        elif status == "NON_COMPLIANT":
            by_severity[sev]["non_compliant"] += 1
        else:
            by_severity[sev]["needs_review"] += 1

    # By check type
    by_type = {}
    for f in findings:
        ct = f.get("check_type", "UNKNOWN")
        by_type[ct] = by_type.get(ct, 0) + 1

    return {
        "overall_compliance_score": overall_score,
        "total_rules_checked": total,
        "compliant": compliant,
        "non_compliant": non_compliant,
        "needs_review": needs_review,
        "average_confidence": round(avg_confidence, 2),
        "by_severity": by_severity,
        "by_check_type": by_type,
    }
