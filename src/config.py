"""
Configuration and constants for the Audit & Compliance Validator.
Adjust paths based on your environment (local vs AMD cloud).
"""
import os

# ============================================================
# PATHS — Change these based on your environment
# ============================================================
# Base directory containing the dataset
BASE_DIR = os.environ.get("AUDIT_BASE_DIR", ".")

# Where compliance PDFs live (SEBI regulations)
COMPLIANCE_DIR = os.path.join(BASE_DIR, "compliance_guidelines")

# Where IPO DRHP PDFs live
IPO_DIR = os.path.join(BASE_DIR, "ipo_documents")

# Persistent storage directory (survives notebook restart on AMD)
# On AMD cloud: use the shared folder path
# Locally: just use ./shared
SHARED_DIR = os.environ.get("AUDIT_SHARED_DIR", os.path.join(BASE_DIR, "shared"))

# ChromaDB persistent path
CHROMA_DB_PATH = os.path.join(SHARED_DIR, "chroma_db")

# Auto-generated compliance rules cache
COMPLIANCE_RULES_PATH = os.path.join(SHARED_DIR, "compliance_rules.json")

# Evaluation ground truth
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "ground_truth.json")

# ============================================================
# MODEL CONFIGURATION
# ============================================================
# vLLM server endpoint
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = "dummy"  # vLLM doesn't need a real key
VLLM_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

# Embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DEVICE = "cuda"  # Use "cpu" if GPU not available for embeddings
EMBEDDING_DIM = 1024  # bge-large-en-v1.5 output dimension

# OCR toggle — disable if PDFs are selectable text (no scanned images)
# PaddleOCR has compatibility issues on some environments (ROCm, etc.)
USE_OCR = os.environ.get("USE_OCR", "false").lower() == "true"

# ============================================================
# LLM PARAMETERS
# ============================================================
LLM_TEMPERATURE = 0.1       # Low temp for consistency
LLM_MAX_TOKENS = 2048       # Max output tokens per call
LLM_MAX_RETRIES = 3         # Retry failed LLM calls

# ============================================================
# CHUNKING PARAMETERS
# ============================================================
CHUNK_SIZE = 1000            # Target chunk size in tokens (approx chars / 4)
CHUNK_OVERLAP = 200          # Overlap between chunks in tokens
MAX_CHUNK_CHARS = 4000       # Max characters per chunk
OVERLAP_CHARS = 800          # Overlap in characters

# ============================================================
# RAG PARAMETERS
# ============================================================
COMPLIANCE_COLLECTION_NAME = "sebi_compliance"
DOCUMENT_COLLECTION_PREFIX = "drhp_"
RAG_TOP_K = 8                # Number of chunks to retrieve

# ============================================================
# VALIDATION PARAMETERS
# ============================================================
CONFIDENCE_WEIGHTS = {
    "retrieval": 0.15,
    "evidence": 0.15,
    "llm": 0.25,
    "critic": 0.25,
    "cross_validation": 0.20,
}

SEVERITY_WEIGHTS = {
    "CRITICAL": 3.0,
    "HIGH": 2.0,
    "MEDIUM": 1.0,
    "LOW": 0.5,
}

# ============================================================
# SECTION PATTERNS — SEBI Schedule VI Mandatory Sections
# ============================================================
SECTION_PATTERNS = {
    "risk_factors":          r"(?i)(?:^|\n)\s*(?:SECTION\s+\w+[\s:–\-]*)?RISK\s+FACTORS",
    "capital_structure":     r"(?i)(?:^|\n)\s*(?:SECTION\s+\w+[\s:–\-]*)?CAPITAL\s+STRUCTURE",
    "objects_of_issue":      r"(?i)(?:^|\n)\s*OBJECTS?\s+OF\s+(?:THE\s+)?(?:OFFER|ISSUE)",
    "basis_for_price":       r"(?i)(?:^|\n)\s*BASIS\s+FOR\s+(?:OFFER|ISSUE)\s+PRICE",
    "financial_info":        r"(?i)(?:^|\n)\s*FINANCIAL\s+(?:INFORMATION|STATEMENTS)",
    "legal_info":            r"(?i)(?:^|\n)\s*LEGAL\s+AND\s+OTHER",
    "our_management":        r"(?i)(?:^|\n)\s*OUR\s+MANAGEMENT",
    "our_promoters":         r"(?i)(?:^|\n)\s*OUR\s+PROMOTERS?",
    "dividend_policy":       r"(?i)(?:^|\n)\s*DIVIDEND\s+POLICY",
    "industry_overview":     r"(?i)(?:^|\n)\s*INDUSTRY\s+OVERVIEW",
    "our_business":          r"(?i)(?:^|\n)\s*OUR\s+BUSINESS",
    "tax_benefits":          r"(?i)(?:^|\n)\s*STATEMENT\s+OF\s+(?:POSSIBLE\s+)?TAX\s+BENEFITS",
    "general_info":          r"(?i)(?:^|\n)\s*GENERAL\s+INFORMATION",
    "related_party":         r"(?i)(?:^|\n)\s*RELATED\s+PARTY\s+TRANSACTIONS?",
    "offer_structure":       r"(?i)(?:^|\n)\s*(?:TERMS\s+OF\s+THE\s+)?OFFER(?:\s+STRUCTURE)?",
    "history_corporate":     r"(?i)(?:^|\n)\s*HISTORY\s+AND\s+(?:CERTAIN\s+)?CORPORATE",
    "regulatory_statutory":  r"(?i)(?:^|\n)\s*OTHER\s+REGULATORY\s+AND\s+STATUTORY",
    "description_equity":    r"(?i)(?:^|\n)\s*DESCRIPTION\s+OF\s+EQUITY\s+SHARES",
    "restrictions_foreign":  r"(?i)(?:^|\n)\s*RESTRICTIONS?\s+ON\s+FOREIGN",
    "definitions":           r"(?i)(?:^|\n)\s*DEFINITIONS?\s+AND\s+ABBREVIATIONS?",
    "table_of_contents":     r"(?i)(?:^|\n)\s*TABLE\s+OF\s+CONTENTS?",
    "summary":               r"(?i)(?:^|\n)\s*(?:OFFER\s+)?SUMMARY",
    "key_regulations":       r"(?i)(?:^|\n)\s*KEY\s+(?:INDUSTRY\s+)?REGULATIONS?",
}

# Required sections per SEBI Schedule VI
MANDATORY_SECTIONS = [
    "risk_factors", "capital_structure", "objects_of_issue",
    "basis_for_price", "financial_info", "legal_info",
    "our_management", "our_promoters", "dividend_policy",
    "industry_overview", "our_business", "tax_benefits",
    "general_info", "definitions", "table_of_contents",
]

# Financial amount patterns for cross-reference checking
AMOUNT_PATTERNS = [
    r"₹\s*([\d,]+\.?\d*)\s*(crore|lakh|million|billion|cr|lac)",
    r"Rs\.?\s*([\d,]+\.?\d*)\s*(crore|lakh|million|billion|cr|lac)",
    r"INR\s*([\d,]+\.?\d*)\s*(crore|lakh|million|billion|cr|lac)",
]

# CIN pattern (Corporate Identity Number)
CIN_PATTERN = r"[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}"

# ISIN pattern
ISIN_PATTERN = r"INE[A-Z0-9]{9}"

# PAN pattern
PAN_PATTERN = r"[A-Z]{5}\d{4}[A-Z]"


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs(SHARED_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(IPO_DIR, exist_ok=True)
    os.makedirs(COMPLIANCE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)


def print_config():
    """Print current configuration for verification."""
    print("=" * 60)
    print("AUDIT & COMPLIANCE VALIDATOR — CONFIGURATION")
    print("=" * 60)
    print(f"  Base Directory:     {BASE_DIR}")
    print(f"  Compliance PDFs:    {COMPLIANCE_DIR}")
    print(f"  IPO Documents:      {IPO_DIR}")
    print(f"  Shared Storage:     {SHARED_DIR}")
    print(f"  ChromaDB Path:      {CHROMA_DB_PATH}")
    print(f"  vLLM Endpoint:      {VLLM_BASE_URL}")
    print(f"  LLM Model:          {VLLM_MODEL_NAME}")
    print(f"  Embedding Model:    {EMBEDDING_MODEL_NAME}")
    print(f"  Embedding Device:   {EMBEDDING_DEVICE}")
    print("=" * 60)
