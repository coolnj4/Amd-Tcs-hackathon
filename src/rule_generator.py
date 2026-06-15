"""
Rule Generator — Uses LLM to auto-generate compliance rules from SEBI regulation PDFs.
This is the key differentiator: the system reads regulations and creates its own checklist.
"""
import json
import os
from tqdm import tqdm
from src.config import COMPLIANCE_RULES_PATH


RULE_EXTRACTION_SYSTEM = "You are a SEBI regulatory compliance expert specializing in IPO and capital markets regulations."

RULE_EXTRACTION_PROMPT = """Read the following SEBI regulation text and extract a structured compliance validation rule 
that can be checked against a Draft Red Herring Prospectus (DRHP).

REGULATION TEXT:
{regulation_text}

SOURCE: {source_file}
REGULATION REFERENCE: {regulation_ref}

Output a JSON object with these fields:
{{
    "rule_id": "A unique ID like ICDR_REG_26_1 or LODR_REG_15",
    "title": "Short descriptive title (max 10 words)",
    "regulation_reference": "Exact regulation number (e.g., Regulation 26(1))",
    "source_document": "Which SEBI circular this comes from",
    "description": "What this regulation requires, in plain English (2-3 sentences)",
    "check_type": "One of: DETERMINISTIC, SEMANTIC, NUMERICAL, NLP_EXTRACTION",
    "severity": "One of: CRITICAL, HIGH, MEDIUM, LOW",
    "required_elements": ["Specific things that must be present in the DRHP"],
    "search_keywords": ["Keywords to search for relevant DRHP sections"],
    "validation_question": "A precise yes/no question to check compliance",
    "what_to_look_for": "Specific evidence to check in the document",
    "common_violations": ["1-2 common ways companies fail this rule"],
    "applicable_sections": ["Which DRHP sections this rule applies to, e.g. cover_page, risk_factors, capital_structure"]
}}

RULES:
- Only extract rules that can be VALIDATED by reading a DRHP document.
- Skip procedural rules about SEBI's internal processes or timelines.
- Skip rules about intermediaries (merchant bankers, registrars) unless they require disclosures in the DRHP.
- Be specific about what constitutes compliance vs non-compliance.
- Choose check_type based on what's needed:
  - DETERMINISTIC: Can be checked with regex/pattern matching (presence of CIN, ISIN, specific sections)
  - SEMANTIC: Requires understanding meaning (adequacy of disclosures, risk factor quality)
  - NUMERICAL: Involves checking financial figures (table totals, percentages)
  - NLP_EXTRACTION: Requires extracting specific entities or facts
- If this text doesn't contain a rule checkable from a DRHP, respond with: {{"skip": true, "reason": "brief reason"}}
"""


def _load_default_rules() -> list:
    """Load default SEBI compliance rules from the bundled JSON file."""
    rules_json_path = os.path.join(os.path.dirname(__file__), "default_rules.json")
    if os.path.exists(rules_json_path):
        with open(rules_json_path, 'r') as f:
            return json.load(f)
    print("[WARN] default_rules.json not found. Using empty rule set.")
    return []


DEFAULT_SEBI_RULES = _load_default_rules()


def generate_rules(llm_client, compliance_chunks: list, force_regenerate: bool = False) -> list:
    """
    Auto-generate compliance rules from SEBI regulation chunks using LLM.

    Args:
        llm_client: LLMClient instance
        compliance_chunks: List of Chunk objects from compliance documents
        force_regenerate: If True, regenerate even if cached rules exist

    Returns:
        List of rule dicts
    """
    # Check cache first
    if not force_regenerate and os.path.exists(COMPLIANCE_RULES_PATH):
        with open(COMPLIANCE_RULES_PATH, 'r') as f:
            rules = json.load(f)
        print(f"  ✅ Loaded {len(rules)} cached rules from {COMPLIANCE_RULES_PATH}")
        return rules

    # If force_regenerate is False, we use the default rules list as a fallback
    # so the user doesn't have to wait for hours on the first run of the demo
    if not force_regenerate:
        print(f"\n  ⚠️  No cached rules found. Initializing with default SEBI compliance checklist...")
        rules = DEFAULT_SEBI_RULES
        # Save to cache
        os.makedirs(os.path.dirname(COMPLIANCE_RULES_PATH), exist_ok=True)
        with open(COMPLIANCE_RULES_PATH, 'w') as f:
            json.dump(rules, f, indent=2)
        print(f"  ✅ Created {len(rules)} default SEBI compliance rules in cache at {COMPLIANCE_RULES_PATH}")
        return rules

    print(f"\n  🧠 Auto-generating compliance rules from {len(compliance_chunks)} regulation chunks...")
    print(f"  ⏳ This will take a few minutes (one LLM call per chunk)...\n")

    rules = []
    skipped = 0

    for chunk in tqdm(compliance_chunks, desc="  Extracting rules"):
        try:
            # Get source info from metadata
            source_file = chunk.metadata.get("source_file", "Unknown")
            regulation_ref = chunk.metadata.get("regulation_number", "N/A")

            # Skip very short chunks
            if len(chunk.text.strip()) < 100:
                skipped += 1
                continue

            response = llm_client.call_json(
                system=RULE_EXTRACTION_SYSTEM,
                user=RULE_EXTRACTION_PROMPT.format(
                    regulation_text=chunk.text[:3000],  # Limit input size
                    source_file=os.path.basename(source_file),
                    regulation_ref=regulation_ref,
                ),
            )

            if response.get("skip"):
                skipped += 1
                continue

            # Validate required fields
            required = ["rule_id", "title", "check_type", "severity", "validation_question"]
            if all(response.get(f) for f in required):
                # Ensure unique rule_id
                existing_ids = {r["rule_id"] for r in rules}
                if response["rule_id"] in existing_ids:
                    response["rule_id"] = f"{response['rule_id']}_{len(rules)}"

                rules.append(response)
            else:
                skipped += 1

        except Exception as e:
            print(f"  [WARN] Rule extraction failed for chunk: {e}")
            skipped += 1
            continue

    # Save to cache
    os.makedirs(os.path.dirname(COMPLIANCE_RULES_PATH), exist_ok=True)
    with open(COMPLIANCE_RULES_PATH, 'w') as f:
        json.dump(rules, f, indent=2)

    print(f"\n  ✅ Generated {len(rules)} compliance rules (skipped {skipped} non-applicable chunks)")
    print(f"  💾 Saved to {COMPLIANCE_RULES_PATH}")

    # Print summary by check type
    by_type = {}
    for r in rules:
        ct = r.get("check_type", "UNKNOWN")
        by_type[ct] = by_type.get(ct, 0) + 1
    print(f"  📊 By type: {by_type}")

    by_severity = {}
    for r in rules:
        s = r.get("severity", "UNKNOWN")
        by_severity[s] = by_severity.get(s, 0) + 1
    print(f"  📊 By severity: {by_severity}")

    return rules


def load_rules() -> list:
    """Load rules from cache file."""
    if os.path.exists(COMPLIANCE_RULES_PATH):
        with open(COMPLIANCE_RULES_PATH, 'r') as f:
            return json.load(f)
    return []


def print_rules_summary(rules: list):
    """Print a formatted summary of compliance rules."""
    print(f"\n{'='*70}")
    print(f"  COMPLIANCE RULES SUMMARY ({len(rules)} rules)")
    print(f"{'='*70}")

    for i, rule in enumerate(rules, 1):
        severity_icon = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }.get(rule.get("severity", ""), "⚪")

        check_icon = {
            "DETERMINISTIC": "🔧",
            "SEMANTIC": "🧠",
            "NUMERICAL": "🔢",
            "NLP_EXTRACTION": "📝",
        }.get(rule.get("check_type", ""), "❓")

        print(f"  {i:2d}. {severity_icon} {check_icon}  [{rule.get('rule_id', 'N/A')}]")
        print(f"      {rule.get('title', 'N/A')}")
        print(f"      {rule.get('regulation_reference', '')} | {rule.get('check_type', '')} | {rule.get('severity', '')}")
        print()
