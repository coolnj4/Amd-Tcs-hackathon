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


DEFAULT_SEBI_RULES = [
  {
    "rule_id": "SEBI_ICDR_CIN",
    "title": "Corporate Identity Number (CIN) Disclosure",
    "regulation_reference": "Regulation 26(1)",
    "source_document": "sebi_icdr_master_circular.pdf",
    "description": "The Draft Red Herring Prospectus (DRHP) must disclose the Corporate Identity Number (CIN) of the issuer company on the cover page.",
    "check_type": "DETERMINISTIC",
    "severity": "CRITICAL",
    "required_elements": ["CIN", "L21000", "U21000", "L\\d{5}[A-Z]{2}\\d{4}[A-Z]{3}\\d{6}"],
    "search_keywords": ["Corporate Identity Number", "CIN", "Company Registration Number"],
    "validation_question": "Is the Corporate Identity Number (CIN) of the company disclosed in the prospectus?",
    "what_to_look_for": "A 21-digit alphanumeric code starting with L or U on the cover page or general information section.",
    "common_violations": ["CIN is missing or has incorrect format"],
    "applicable_sections": ["cover_page", "general_info"]
  },
  {
    "rule_id": "SEBI_ICDR_RISK_FACTORS_COVER",
    "title": "Risk Factors Reference on Cover Page",
    "regulation_reference": "Schedule VI, Part A",
    "source_document": "sebi_icdr_master_circular.pdf",
    "description": "The cover page must contain a specific reference to the 'Risk Factors' section, directing investors to read the detailed risk disclosures.",
    "check_type": "SEMANTIC",
    "severity": "HIGH",
    "required_elements": ["risk factors", "detailed risks"],
    "search_keywords": ["risk factors", "read the risk factors", "refer to page"],
    "validation_question": "Does the cover page contain a clear reference directing readers to the detailed Risk Factors section?",
    "what_to_look_for": "A sentence on the first page pointing to 'Risk Factors' and referencing the page range or section.",
    "common_violations": ["Missing reference to risk factors on the cover page"],
    "applicable_sections": ["cover_page"]
  },
  {
    "rule_id": "SEBI_ICDR_OBJECTS_OF_ISSUE",
    "title": "Objects of the Issue Disclosure",
    "regulation_reference": "Schedule VI, Part A(2)",
    "source_document": "sebi_icdr_master_circular.pdf",
    "description": "The objects of the issue must specify the funding requirements, details of the projects, schedule of deployment, and means of finance.",
    "check_type": "SEMANTIC",
    "severity": "CRITICAL",
    "required_elements": ["Objects of the Offer", "Means of Finance", "Schedule of Deployment"],
    "search_keywords": ["Objects of the Offer", "Objects of the Issue", "Requirements of Funds", "Means of Finance"],
    "validation_question": "Are the objects of the issue clearly detailed with project funding requirements and a schedule of deployment?",
    "what_to_look_for": "Detailed breakdown of tables showing project cost, funding timeline, and deployment schedules in the Objects section.",
    "common_violations": ["Lack of detail on deployment schedules or unallocated general corporate purposes exceed 25% limit"],
    "applicable_sections": ["objects_of_issue"]
  },
  {
    "rule_id": "SEBI_LODR_BOARD_COMPOSITION",
    "title": "Independent Directors Board Ratio",
    "regulation_reference": "Regulation 17(1)",
    "source_document": "sebi_lodr_master_circular.pdf",
    "description": "The board of directors must have an optimum combination of executive and non-executive directors. At least 33% of the board must be independent if the chairperson is non-executive, and 50% if executive.",
    "check_type": "NUMERICAL",
    "severity": "CRITICAL",
    "required_elements": ["Independent Directors", "Non-Executive Directors", "Board of Directors"],
    "search_keywords": ["board composition", "independent directors", "non-executive", "executive directors"],
    "validation_question": "Does the Board of Directors meet the minimum required ratio of independent directors?",
    "what_to_look_for": "Table of Board members outlining their designations (Independent vs Executive vs Non-Executive).",
    "common_violations": ["Insufficient number of independent directors on the board"],
    "applicable_sections": ["our_management"]
  },
  {
    "rule_id": "SEBI_LODR_AUDIT_COMMITTEE",
    "title": "Audit Committee Composition",
    "regulation_reference": "Regulation 18(1)",
    "source_document": "sebi_lodr_master_circular.pdf",
    "description": "The audit committee must consist of at least three directors, with two-thirds of them being independent directors.",
    "check_type": "SEMANTIC",
    "severity": "CRITICAL",
    "required_elements": ["Audit Committee", "Independent members", "Minimum three"],
    "search_keywords": ["Audit Committee", "composition of audit committee", "committee members"],
    "validation_question": "Is the Audit Committee composed of at least three directors with two-thirds being independent?",
    "what_to_look_for": "Section detailing the composition of the Audit Committee and designations of its members.",
    "common_violations": ["Fewer than two-thirds independent directors on the Audit Committee"],
    "applicable_sections": ["our_management"]
  },
  {
    "rule_id": "SEBI_ICDR_DIVIDEND_POLICY",
    "title": "Dividend History and Policy Disclosure",
    "regulation_reference": "Schedule VI, Part A(9)",
    "source_document": "sebi_icdr_master_circular.pdf",
    "description": "The issuer company must disclose its dividend history for the last three financial years and state its dividend policy.",
    "check_type": "SEMANTIC",
    "severity": "MEDIUM",
    "required_elements": ["dividend policy", "dividend history", "dividend declared"],
    "search_keywords": ["Dividend Policy", "dividends declared", "dividend history"],
    "validation_question": "Does the prospectus disclose the dividend policy and dividend history for the past three years?",
    "what_to_look_for": "Dividend policy description and table of dividends declared in the Dividend Policy section.",
    "common_violations": ["Missing dividend history table or vague policy statement"],
    "applicable_sections": ["dividend_policy"]
  },
  {
    "rule_id": "SEBI_ICDR_BASIS_FOR_PRICE",
    "title": "Basis for Offer Price Quantitative Disclosures",
    "regulation_reference": "Schedule VI, Part A(5)",
    "source_document": "sebi_icdr_master_circular.pdf",
    "description": "The prospectus must disclose quantitative factors justifying the offer price, including EPS, P/E ratio, Return on Net Worth (RoNW), and Net Asset Value (NAV).",
    "check_type": "SEMANTIC",
    "severity": "HIGH",
    "required_elements": ["Earnings Per Share", "P/E Ratio", "Return on Net Worth", "Net Asset Value"],
    "search_keywords": ["Basis for Offer Price", "Quantitative Factors", "EPS", "RONW", "NAV", "P/E Ratio"],
    "validation_question": "Are quantitative factors such as EPS, RoNW, P/E ratio, and NAV disclosed and justified?",
    "what_to_look_for": "Tables showing key financial metrics and comparison with industry peers in the Basis for Offer Price section.",
    "common_violations": ["Missing peer comparison or incomplete quantitative disclosures"],
    "applicable_sections": ["basis_for_price"]
  },
  {
    "rule_id": "SEBI_ICDR_PROMOTER_DETAILS",
    "title": "Promoters and Group Companies Disclosure",
    "regulation_reference": "Schedule VI, Part A(8)",
    "source_document": "sebi_icdr_master_circular.pdf",
    "description": "The prospectus must disclose detailed profiles of the promoters, including their educational qualifications, experience, and details of promoter group entities.",
    "check_type": "SEMANTIC",
    "severity": "HIGH",
    "required_elements": ["Promoter details", "educational qualification", "experience", "Promoter Group"],
    "search_keywords": ["Our Promoters", "promoter profile", "qualification of promoters", "experience of promoters"],
    "validation_question": "Are promoter details (qualification, experience, group companies) fully disclosed?",
    "what_to_look_for": "Detailed profiles and tables of promoters and promoter group entities in the Our Promoters section.",
    "common_violations": ["Missing background details or qualifications of individual promoters"],
    "applicable_sections": ["our_promoters"]
  },
  {
    "rule_id": "SEBI_ICDR_LITIGATION_DISCLOSURE",
    "title": "Outstanding Litigations Disclosure",
    "regulation_reference": "Schedule VI, Part A(11)",
    "source_document": "sebi_icdr_master_circular.pdf",
    "description": "All outstanding litigations involving the issuer, promoters, directors, and group companies above the materiality threshold must be disclosed.",
    "check_type": "SEMANTIC",
    "severity": "CRITICAL",
    "required_elements": ["Outstanding Litigations", "Materiality Threshold", "Criminal proceedings", "Tax disputes"],
    "search_keywords": ["Outstanding Litigation", "Legal Proceedings", "Pending Litigation", "Materiality Policy"],
    "validation_question": "Are outstanding litigations and material legal proceedings disclosed?",
    "what_to_look_for": "Summarized table and details of criminal, tax, and civil cases in the Legal and Other Information section.",
    "common_violations": ["Under-reporting pending tax disputes or criminal cases against promoters/directors"],
    "applicable_sections": ["legal_info"]
  },
  {
    "rule_id": "SEBI_ICDR_RISK_FACTORS_INTERNAL",
    "title": "Prominent Internal Risk Disclosures",
    "regulation_reference": "Schedule VI, Part A(1)",
    "source_document": "sebi_icdr_master_circular.pdf",
    "description": "Risk factors must be prominent, clear, and divided into internal risk factors (specific to the company/industry) and external risk factors.",
    "check_type": "SEMANTIC",
    "severity": "HIGH",
    "required_elements": ["Internal Risk Factors", "External Risk Factors", "Risk Factors"],
    "search_keywords": ["Risk Factors", "Internal Risk", "External Risk"],
    "validation_question": "Are the risk factors categorized into internal and external risks in a clear hierarchy?",
    "what_to_look_for": "List of risk factors with distinct subheadings for Internal Risks and External Risks in the Risk Factors section.",
    "common_violations": ["No clear division between internal and external risk factors"],
    "applicable_sections": ["risk_factors"]
  }
]


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
