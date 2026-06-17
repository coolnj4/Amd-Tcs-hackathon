"""
Evaluation Script — Computes Precision, Recall, F1 from cached audit results
against ground truth labels.

Usage:
    python evaluate.py                              # Auto-detect cached results
    python evaluate.py --cache-dir shared            # Specify cache directory
    python evaluate.py --ground-truth ground_truth.json  # Custom ground truth path
"""
import os
import json
import argparse
from collections import defaultdict


def load_ground_truth(path: str = "ground_truth.json") -> dict:
    """Load ground truth labels."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_results(cache_dir: str = "shared") -> list:
    """Load all cached audit results."""
    results = []
    if not os.path.isdir(cache_dir):
        print(f"  Warning: Cache directory '{cache_dir}' not found")
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
                print(f"  Warning: Failed to load {filename}: {e}")

    return results


def match_audit_to_gt(audit_data: dict, gt_documents: dict) -> tuple:
    """Match an audit result to its ground truth document."""
    company = audit_data.get("company_name", "").upper()
    
    for gt_key, gt_doc in gt_documents.items():
        gt_company = gt_doc.get("company_name", "").upper()
        # Fuzzy match: check if key words overlap
        if gt_company in company or company in gt_company:
            return gt_key, gt_doc
        # Try matching by filename
        doc_name = audit_data.get("document_name", "").lower()
        gt_file = gt_doc.get("document_file", "").lower()
        if gt_file and gt_file in doc_name:
            return gt_key, gt_doc
    
    return None, None


def evaluate(audit_data: dict, gt_doc: dict) -> dict:
    """
    Evaluate a single audit against ground truth.
    
    Returns dict with per-rule results and aggregate metrics.
    """
    findings = audit_data.get("findings", [])
    gt_labels = gt_doc.get("labels", {})
    
    # Build lookup: rule_id -> finding
    findings_by_id = {}
    for f in findings:
        findings_by_id[f.get("rule_id", "")] = f
    
    results = {
        "per_rule": [],
        "total_gt_rules": len(gt_labels),
        "matched": 0,
        "correct": 0,
        "incorrect": 0,
        "missing": 0,
    }
    
    tp = 0  # True positives (correctly identified status)
    fp = 0  # False positives (wrong status)
    fn = 0  # False negatives (rule in GT but not in audit)
    
    # Breakdown by check type
    by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    
    for rule_id, gt_label in gt_labels.items():
        expected = gt_label["expected_status"]
        check_type = gt_label.get("check_type", "UNKNOWN")
        finding = findings_by_id.get(rule_id)
        
        rule_result = {
            "rule_id": rule_id,
            "rule_title": gt_label.get("rule_title", ""),
            "expected": expected,
            "check_type": check_type,
        }
        
        if finding:
            actual = finding.get("status", "UNKNOWN")
            rule_result["actual"] = actual
            rule_result["confidence"] = finding.get("confidence", 0)
            results["matched"] += 1
            
            # For evaluation: NEEDS_REVIEW counts as incorrect
            # unless the expected is also ambiguous
            if actual == expected:
                rule_result["match"] = True
                results["correct"] += 1
                tp += 1
                by_type[check_type]["tp"] += 1
            elif actual == "NEEDS_REVIEW" and expected == "COMPLIANT":
                # Partial miss — system couldn't determine
                rule_result["match"] = False
                rule_result["note"] = "System returned NEEDS_REVIEW instead of COMPLIANT"
                results["incorrect"] += 1
                fp += 1
                by_type[check_type]["fp"] += 1
            else:
                rule_result["match"] = False
                results["incorrect"] += 1
                fp += 1
                by_type[check_type]["fp"] += 1
        else:
            rule_result["actual"] = "NOT_FOUND"
            rule_result["match"] = False
            results["missing"] += 1
            fn += 1
            by_type[check_type]["fn"] += 1
        
        by_type[check_type]["total"] += 1
        results["per_rule"].append(rule_result)
    
    # Compute metrics
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    
    results["precision"] = round(precision, 3)
    results["recall"] = round(recall, 3)
    results["f1"] = round(f1, 3)
    results["tp"] = tp
    results["fp"] = fp
    results["fn"] = fn
    
    # Per check-type breakdown
    results["by_check_type"] = {}
    for ct, counts in by_type.items():
        ct_tp = counts["tp"]
        ct_fp = counts["fp"]
        ct_fn = counts["fn"]
        ct_p = ct_tp / max(ct_tp + ct_fp, 1)
        ct_r = ct_tp / max(ct_tp + ct_fn, 1)
        ct_f1 = 2 * ct_p * ct_r / max(ct_p + ct_r, 0.001)
        results["by_check_type"][ct] = {
            "precision": round(ct_p, 3),
            "recall": round(ct_r, 3),
            "f1": round(ct_f1, 3),
            "total": counts["total"],
            "correct": ct_tp,
        }
    
    return results


def print_report(all_results: list):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("  EVALUATION REPORT — AI Audit & Compliance Validator")
    print("=" * 70)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    all_type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    
    for result in all_results:
        company = result.get("company", "Unknown")
        print(f"\n  Document: {company}")
        print(f"  {'-' * 50}")
        print(f"  GT Rules:  {result['total_gt_rules']}")
        print(f"  Matched:   {result['matched']}")
        print(f"  Correct:   {result['correct']}")
        print(f"  Incorrect: {result['incorrect']}")
        print(f"  Missing:   {result['missing']}")
        print(f"  Precision: {result['precision']:.1%}")
        print(f"  Recall:    {result['recall']:.1%}")
        print(f"  F1 Score:  {result['f1']:.1%}")
        
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        
        # Per check type
        for ct, stats in result.get("by_check_type", {}).items():
            print(f"    {ct}: P={stats['precision']:.0%} R={stats['recall']:.0%} "
                  f"F1={stats['f1']:.0%} ({stats['correct']}/{stats['total']})")
            all_type_stats[ct]["tp"] += stats.get("correct", 0)
            all_type_stats[ct]["total"] += stats.get("total", 0)
        
        # Show mismatches
        mismatches = [r for r in result["per_rule"] if not r.get("match", False)]
        if mismatches:
            print(f"\n  Mismatches:")
            for m in mismatches:
                print(f"    {m['rule_id']}: expected={m['expected']}, "
                      f"got={m.get('actual', 'N/A')} "
                      f"{'(' + m.get('note', '') + ')' if m.get('note') else ''}")
    
    # Aggregate
    agg_p = total_tp / max(total_tp + total_fp, 1)
    agg_r = total_tp / max(total_tp + total_fn, 1)
    agg_f1 = 2 * agg_p * agg_r / max(agg_p + agg_r, 0.001)
    
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE METRICS (across all documents)")
    print(f"{'=' * 70}")
    print(f"  Precision:  {agg_p:.1%}")
    print(f"  Recall:     {agg_r:.1%}")
    print(f"  F1 Score:   {agg_f1:.1%}")
    
    for ct in sorted(all_type_stats.keys()):
        stats = all_type_stats[ct]
        acc = stats["tp"] / max(stats["total"], 1)
        print(f"  {ct}: {acc:.0%} accuracy ({stats['tp']}/{stats['total']})")
    
    print(f"{'=' * 70}\n")
    
    return {
        "precision": round(agg_p, 3),
        "recall": round(agg_r, 3),
        "f1": round(agg_f1, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate audit results against ground truth")
    parser.add_argument("--cache-dir", default="shared", help="Directory with cached audit results")
    parser.add_argument("--ground-truth", default="ground_truth.json", help="Ground truth file")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()
    
    # Load data
    gt = load_ground_truth(args.ground_truth)
    gt_docs = gt.get("documents", {})
    
    cached = load_cached_results(args.cache_dir)
    
    if not cached:
        # Try reports directory as fallback
        reports_dir = os.path.join(
            os.environ.get("AUDIT_BASE_DIR", "."), "reports"
        )
        if os.path.isdir(reports_dir):
            print(f"  Trying reports directory: {reports_dir}")
            for filename in sorted(os.listdir(reports_dir)):
                if filename.startswith("audit_report_") and filename.endswith(".json"):
                    filepath = os.path.join(reports_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    cached.append({
                        "company_name": report.get("company_name", "Unknown"),
                        "document_name": report.get("document_name", ""),
                        "findings": report.get("findings", []),
                    })
    
    if not cached:
        print("ERROR: No cached audit results found.")
        print("Run the notebook first to generate audit results.")
        return
    
    print(f"\nFound {len(cached)} audit result(s)")
    print(f"Ground truth has {len(gt_docs)} document(s)")
    
    all_results = []
    for audit_data in cached:
        gt_key, gt_doc = match_audit_to_gt(audit_data, gt_docs)
        if gt_doc:
            print(f"\n  Matched: {audit_data.get('company_name', 'Unknown')} -> {gt_key}")
            result = evaluate(audit_data, gt_doc)
            result["company"] = audit_data.get("company_name", "Unknown")
            all_results.append(result)
        else:
            print(f"  No ground truth match for: {audit_data.get('company_name', 'Unknown')}")
    
    if all_results:
        aggregate = print_report(all_results)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"aggregate": aggregate, "per_document": all_results}, f, indent=2, default=str)
            print(f"  Saved evaluation results to: {args.output}")


if __name__ == "__main__":
    main()
