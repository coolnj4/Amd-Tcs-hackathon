"""
Document Parser — Extracts text, tables, and metadata from PDF documents.
Uses PyMuPDF for text, pdfplumber for tables, and PaddleOCR for scanned pages.
Optionally uses LLM agent to extract Table of Contents page numbers.
"""
import re
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from src.config import SECTION_PATTERNS, MANDATORY_SECTIONS


@dataclass
class TableData:
    """Extracted table from a PDF page."""
    raw: list                           # Raw table data (list of lists)
    page_num: int                       # 1-indexed page number
    markdown: str = ""                  # Markdown representation
    html: str = ""                      # HTML representation (from PPStructure)


@dataclass
class PageContent:
    """Parsed content of a single PDF page."""
    page_num: int                       # 1-indexed
    text: str                           # Extracted text
    tables: list = field(default_factory=list)  # List of TableData
    ocr_used: bool = False              # Whether OCR was needed


@dataclass
class DetectedSection:
    """A detected document section."""
    name: str                           # Section key (e.g., "risk_factors")
    title: str                          # Actual heading text found
    start_page: int                     # 1-indexed start page
    end_page: Optional[int] = None      # 1-indexed end page (None if last section)
    text: str = ""                      # Full section text


@dataclass
class ParsedDocument:
    """Complete parsed document."""
    file_path: str
    total_pages: int
    pages: list                         # List of PageContent
    sections: list                      # List of DetectedSection
    metadata: dict = field(default_factory=dict)  # Company name, date, etc.
    tables_count: int = 0
    ocr_pages_count: int = 0


def _init_paddle_ocr():
    """Lazily initialize PaddleOCR (heavy import)."""
    try:
        from paddleocr import PaddleOCR
        # Try initializing with standard arguments
        try:
            return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        except ValueError as e:
            print(f"[WARN] Failed standard PaddleOCR init: {e}. Retrying with use_angle_cls and lang...")
            try:
                return PaddleOCR(use_angle_cls=True, lang='en')
            except ValueError as e2:
                print(f"[WARN] Failed with use_angle_cls: {e2}. Retrying with lang only...")
                try:
                    return PaddleOCR(lang='en')
                except ValueError as e3:
                    print(f"[WARN] Failed with lang: {e3}. Retrying with no arguments...")
                    return PaddleOCR()
    except ImportError:
        print("[WARN] PaddleOCR not installed. OCR fallback disabled.")
        print("       Install with: pip install paddlepaddle paddleocr")
        return None


# Lazy global OCR engine
_ocr_engine = None


def get_ocr_engine():
    """Get or initialize the OCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = _init_paddle_ocr()
    return _ocr_engine


def table_to_markdown(table_data: list) -> str:
    """Convert a raw table (list of lists) to markdown format."""
    if not table_data or len(table_data) < 2:
        return ""

    # Clean None values
    cleaned = []
    for row in table_data:
        cleaned.append([str(cell).strip() if cell else "" for cell in row])

    # Build markdown table
    header = cleaned[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in cleaned[1:]:
        # Pad row if needed
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def extract_tables_from_page(pdf_path: str, page_num: int) -> list:
    """
    Extract tables from a specific page using pdfplumber.

    Args:
        pdf_path: Path to PDF file
        page_num: 0-indexed page number

    Returns:
        List of TableData objects
    """
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                raw_tables = page.extract_tables()
                for raw in raw_tables:
                    if raw and len(raw) >= 2:  # At least header + 1 row
                        md = table_to_markdown(raw)
                        tables.append(TableData(
                            raw=raw,
                            page_num=page_num + 1,
                            markdown=md,
                        ))
    except Exception as e:
        print(f"  [WARN] Table extraction failed on page {page_num + 1}: {e}")

    return tables


def ocr_page(page: fitz.Page) -> str:
    """
    Run OCR on a page using PaddleOCR.

    Args:
        page: PyMuPDF page object

    Returns:
        Extracted text string
    """
    engine = get_ocr_engine()
    if engine is None:
        return ""

    try:
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        # Convert RGBA to RGB if needed
        if pix.n == 4:
            img = img[:, :, :3]

        try:
            result = engine.ocr(img, cls=True)
        except TypeError:
            try:
                result = engine.ocr(img)
            except Exception as e:
                # Some newer versions might take different arguments or have a predict method
                # Let's try calling it directly if standard ocr fails
                result = engine(img)

        if result and result[0]:
            # Some versions return list of results, some return dict or custom objects
            try:
                lines = [line[1][0] for line in result[0] if line[1]]
                return "\n".join(lines)
            except Exception:
                # Fallback for alternative return formats (e.g. lists of strings or dicts)
                if isinstance(result, list):
                    if all(isinstance(x, str) for x in result):
                        return "\n".join(result)
                return str(result)
    except Exception as e:
        print(f"  [WARN] OCR failed: {e}")

    return ""


def detect_sections(pages: list) -> list:
    """
    Detect document sections by matching heading patterns.

    Args:
        pages: List of PageContent objects

    Returns:
        List of DetectedSection objects, sorted by start_page
    """
    sections = []
    full_text_by_page = {p.page_num: p.text for p in pages}

    for section_key, pattern in SECTION_PATTERNS.items():
        for page in pages:
            match = re.search(pattern, page.text)
            if match:
                # Extract the actual heading text (first 100 chars of match context)
                start = max(0, match.start() - 10)
                heading_text = page.text[start:match.end() + 50].strip()
                heading_text = heading_text.split("\n")[0].strip()  # First line only

                sections.append(DetectedSection(
                    name=section_key,
                    title=heading_text[:100],
                    start_page=page.page_num,
                ))
                break  # Only find first occurrence

    # Sort by start page
    sections.sort(key=lambda s: s.start_page)

    # Set end pages (each section ends where the next begins)
    for i in range(len(sections)):
        if i + 1 < len(sections):
            sections[i].end_page = sections[i + 1].start_page - 1
        else:
            sections[i].end_page = pages[-1].page_num if pages else None

    # Extract section text
    for section in sections:
        section_text_parts = []
        for page in pages:
            if section.start_page <= page.page_num <= (section.end_page or page.page_num):
                section_text_parts.append(page.text)
        section.text = "\n\n".join(section_text_parts)

    return sections


def extract_sections_with_llm(pages: list, llm_client, verbose: bool = True) -> list:
    """
    Extract document sections by calling LLM on the Table of Contents pages.
    """
    if verbose:
        print("  🧠 Analyzing Table of Contents using LLM Agent...")

    # Identify TOC pages (usually in first 12 pages)
    toc_text_parts = []
    toc_page_nums = []
    for p in pages[:12]:
        text_lower = p.text.lower()
        if "table of contents" in text_lower or "index" in text_lower or "contents" in text_lower or "particulars" in text_lower:
            toc_text_parts.append(p.text)
            toc_page_nums.append(p.page_num)

    # Fallback to pages 3-7 if no explicit keyword matches
    if not toc_text_parts:
        toc_text_parts = [p.text for p in pages[2:7] if p.text]
        toc_page_nums = [p.page_num for p in pages[2:7] if p.text]

    # Combine text with page indicators
    toc_text = ""
    for num, txt in zip(toc_page_nums, toc_text_parts):
        toc_text += f"\n\n--- Page {num} ---\n\n{txt}"

    if not toc_text.strip():
        return []

    TOC_EXTRACTION_SYSTEM = "You are a SEBI compliance expert specializing in IPO document analysis."
    
    prompt = f"""Analyze the following Table of Contents (TOC) pages of a Draft Red Herring Prospectus (DRHP). 
Identify the starting page number for each section.

Standard sections to look for:
{", ".join(MANDATORY_SECTIONS)}

Return a JSON object containing a list of sections, structured exactly like this:
{{
    "sections": [
        {{
            "name": "section_name_from_list",
            "title": "Exact Title as shown in TOC",
            "start_page": 45
        }}
    ]
}}

Rules:
1. ONLY include sections that are actually listed in the TOC text.
2. The start_page MUST be an integer representing the page number shown in the TOC.
3. Use the exact names from the list above for the "name" field.

TOC TEXT:
{toc_text}
"""
    
    try:
        response = llm_client.call_json(system=TOC_EXTRACTION_SYSTEM, user=prompt)
        extracted = response.get("sections", [])
        
        sections = []
        for item in extracted:
            name = item.get("name")
            title = item.get("title")
            start_page = item.get("start_page")
            
            # Verify and clean the data
            if name and name in MANDATORY_SECTIONS and title and isinstance(start_page, int):
                sections.append(DetectedSection(
                    name=name,
                    title=title,
                    start_page=start_page
                ))
                
        if not sections:
            return []

        # Sort sections by start page
        sections.sort(key=lambda s: s.start_page)
        
        # Calculate end pages
        for i in range(len(sections)):
            if i + 1 < len(sections):
                sections[i].end_page = sections[i + 1].start_page - 1
            else:
                sections[i].end_page = pages[-1].page_num if pages else None
                
        # Extract section text
        for section in sections:
            section_text_parts = []
            for page in pages:
                if section.start_page <= page.page_num <= (section.end_page or page.page_num):
                    section_text_parts.append(page.text)
            section.text = "\n\n".join(section_text_parts)
            
        return sections
    except Exception as e:
        if verbose:
            print(f"  [WARN] LLM TOC extraction failed: {e}")
        return []


def extract_cover_metadata(pages: list) -> dict:
    """
    Extract metadata from the cover page (first 3 pages).

    Args:
        pages: List of PageContent (first 3 pages)

    Returns:
        Dict with company_name, cin, registered_office, etc.
    """
    cover_text = "\n".join(p.text for p in pages[:3])
    metadata = {}

    # Try to extract company name (usually in large text on first page)
    # Heuristic: look for "LIMITED" or "LTD" preceded by company name
    name_match = re.search(
        r"([A-Z][A-Z\s&.\-']+(?:LIMITED|LTD|PRIVATE\s+LIMITED))",
        cover_text
    )
    if name_match:
        metadata["company_name"] = name_match.group(1).strip()

    # CIN
    from src.config import CIN_PATTERN
    cin_match = re.search(CIN_PATTERN, cover_text)
    if cin_match:
        metadata["cin"] = cin_match.group(0)

    # ISIN
    from src.config import ISIN_PATTERN
    isin_match = re.search(ISIN_PATTERN, cover_text)
    if isin_match:
        metadata["isin"] = isin_match.group(0)

    # Check for "DRAFT RED HERRING PROSPECTUS" or "RED HERRING PROSPECTUS"
    if re.search(r"(?i)DRAFT\s+RED\s+HERRING\s+PROSPECTUS", cover_text):
        metadata["document_type"] = "DRHP"
    elif re.search(r"(?i)RED\s+HERRING\s+PROSPECTUS", cover_text):
        metadata["document_type"] = "RHP"
    else:
        metadata["document_type"] = "UNKNOWN"

    # Try to extract date
    date_match = re.search(
        r"(?:dated|filed|date)\s*[:\s]*(\d{1,2}[\s/\-\.]+\w+[\s/\-\.]+\d{4})",
        cover_text, re.IGNORECASE
    )
    if date_match:
        metadata["filing_date"] = date_match.group(1).strip()

    return metadata


def parse_document(pdf_path: str, verbose: bool = True, llm_client=None) -> ParsedDocument:
    """
    Parse a complete PDF document.

    Args:
        pdf_path: Path to the PDF file
        verbose: Whether to show progress
        llm_client: Optional LLMClient to extract Table of Contents page boundaries

    Returns:
        ParsedDocument with all extracted content
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Parsing: {pdf_path}")
        print(f"{'='*60}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages = []
    total_tables = 0
    ocr_count = 0

    iterator = range(total_pages)
    if verbose:
        iterator = tqdm(iterator, desc="  Extracting pages", unit="page")

    for page_num in iterator:
        page = doc[page_num]

        # Extract text using PyMuPDF
        text = page.get_text("text")
        ocr_used = False

        # If text is too short, try OCR
        if len(text.strip()) < 50:
            ocr_text = ocr_page(page)
            if ocr_text:
                text = ocr_text
                ocr_used = True
                ocr_count += 1

        # Extract tables
        tables = extract_tables_from_page(pdf_path, page_num)
        total_tables += len(tables)

        pages.append(PageContent(
            page_num=page_num + 1,
            text=text,
            tables=tables,
            ocr_used=ocr_used,
        ))

    doc.close()

    # Detect sections (use LLM TOC extraction if llm_client is provided, else fallback to regex)
    sections = []
    if llm_client:
        try:
            sections = extract_sections_with_llm(pages, llm_client, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  [WARN] LLM TOC extraction failed: {e}. Falling back to regex...")
    
    if not sections:
        sections = detect_sections(pages)

    # Extract metadata from cover
    metadata = extract_cover_metadata(pages)

    parsed = ParsedDocument(
        file_path=pdf_path,
        total_pages=total_pages,
        pages=pages,
        sections=sections,
        metadata=metadata,
        tables_count=total_tables,
        ocr_pages_count=ocr_count,
    )

    if verbose:
        print(f"  ✅ Parsed {total_pages} pages, {total_tables} tables, {ocr_count} OCR pages")
        print(f"  📋 Detected {len(sections)} sections: {[s.name for s in sections]}")
        if metadata:
            print(f"  🏢 Company: {metadata.get('company_name', 'N/A')}")
            print(f"  📄 Type: {metadata.get('document_type', 'N/A')}")

    return parsed
