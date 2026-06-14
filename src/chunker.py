"""
Section-Aware Chunker — Splits documents into semantically meaningful chunks
preserving section boundaries and table integrity.
"""
import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from src.config import MAX_CHUNK_CHARS, OVERLAP_CHARS


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    # Metadata keys: section, page_numbers, chunk_type, source_file, company_name,
    #                regulation_number, chapter, subject (for compliance chunks)


def generate_chunk_id(text: str, prefix: str = "") -> str:
    """Generate a deterministic chunk ID from text content."""
    hash_val = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_val}" if prefix else hash_val


def split_text_with_overlap(text: str, max_chars: int = None, overlap: int = None) -> list:
    """
    Split text into overlapping chunks at sentence boundaries.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        overlap: Number of overlapping characters

    Returns:
        List of text strings
    """
    max_chars = max_chars or MAX_CHUNK_CHARS
    overlap = overlap or OVERLAP_CHARS

    if len(text) <= max_chars:
        return [text]

    # Split at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Calculate overlap: keep last few sentences
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_len += len(s)

            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_document(parsed_doc, source_type: str = "drhp") -> list:
    """
    Create section-aware chunks from a parsed document.

    For DRHP documents:
      - Primary split by detected section
      - Secondary split within large sections (with overlap)
      - Tables are separate chunks with markdown representation

    For compliance documents:
      - Split by regulation number
      - Schedules split by item

    Args:
        parsed_doc: ParsedDocument object
        source_type: "drhp" or "compliance"

    Returns:
        List of Chunk objects
    """
    chunks = []
    company_name = parsed_doc.metadata.get("company_name", "unknown")

    if source_type == "drhp":
        chunks = _chunk_drhp(parsed_doc, company_name)
    elif source_type == "compliance":
        chunks = _chunk_compliance(parsed_doc)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    print(f"  📦 Created {len(chunks)} chunks ({source_type})")
    return chunks


def _chunk_drhp(parsed_doc, company_name: str) -> list:
    """Chunk a DRHP document by sections."""
    chunks = []

    if parsed_doc.sections:
        # Chunk by section
        for section in parsed_doc.sections:
            section_text = section.text.strip()
            if not section_text:
                continue

            # Split large sections
            text_parts = split_text_with_overlap(section_text)

            for i, part in enumerate(text_parts):
                chunk_id = generate_chunk_id(part, f"drhp_{section.name}")
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=part,
                    metadata={
                        "section": section.name,
                        "section_title": section.title,
                        "page_numbers": f"{section.start_page}-{section.end_page}",
                        "chunk_type": "text",
                        "chunk_index": i,
                        "total_chunks_in_section": len(text_parts),
                        "source_file": parsed_doc.file_path,
                        "company_name": company_name,
                    }
                ))
    else:
        # Fallback: chunk by page groups if no sections detected
        full_text = "\n\n".join(p.text for p in parsed_doc.pages)
        text_parts = split_text_with_overlap(full_text)
        for i, part in enumerate(text_parts):
            chunk_id = generate_chunk_id(part, "drhp_page")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=part,
                metadata={
                    "section": "unknown",
                    "chunk_type": "text",
                    "chunk_index": i,
                    "source_file": parsed_doc.file_path,
                    "company_name": company_name,
                }
            ))

    # Add table chunks separately
    for page in parsed_doc.pages:
        for table in page.tables:
            table_text = table.markdown if table.markdown else str(table.raw)
            if len(table_text.strip()) < 20:
                continue

            # Find which section this table belongs to
            table_section = "unknown"
            for section in parsed_doc.sections:
                if section.start_page <= page.page_num <= (section.end_page or page.page_num):
                    table_section = section.name
                    break

            chunk_id = generate_chunk_id(table_text, f"table_p{page.page_num}")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=table_text,
                metadata={
                    "section": table_section,
                    "page_numbers": str(page.page_num),
                    "chunk_type": "table",
                    "source_file": parsed_doc.file_path,
                    "company_name": company_name,
                }
            ))

    return chunks


def _chunk_compliance(parsed_doc) -> list:
    """
    Chunk a compliance/regulation document by regulation numbers.

    Strategy: Look for patterns like "Regulation 26", "CHAPTER III",
    "Schedule VI" to split the document.
    """
    chunks = []
    full_text = "\n\n".join(p.text for p in parsed_doc.pages)

    # Try to split by regulation numbers
    # Pattern: "Regulation X" or "Reg. X" followed by content
    reg_pattern = r"(?:^|\n)\s*((?:Regulation|Reg\.?)\s+\d+[A-Z]?(?:\(\d+\))?)"
    reg_splits = list(re.finditer(reg_pattern, full_text, re.IGNORECASE))

    source_file = parsed_doc.file_path

    if reg_splits:
        for i, match in enumerate(reg_splits):
            start = match.start()
            end = reg_splits[i + 1].start() if i + 1 < len(reg_splits) else len(full_text)

            reg_text = full_text[start:end].strip()
            reg_number = match.group(1).strip()

            if len(reg_text) < 30:
                continue

            # Sub-chunk if too large
            parts = split_text_with_overlap(reg_text, max_chars=3000, overlap=500)
            for j, part in enumerate(parts):
                chunk_id = generate_chunk_id(part, f"reg_{reg_number.replace(' ', '_')}")
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=part,
                    metadata={
                        "regulation_number": reg_number,
                        "source_file": source_file,
                        "chunk_type": "regulation",
                        "chunk_index": j,
                        "subject": _extract_subject(part),
                    }
                ))
    else:
        # Fallback: chunk by fixed size with overlap
        parts = split_text_with_overlap(full_text, max_chars=2000, overlap=400)
        for i, part in enumerate(parts):
            chunk_id = generate_chunk_id(part, "compliance")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=part,
                metadata={
                    "source_file": source_file,
                    "chunk_type": "compliance_text",
                    "chunk_index": i,
                    "subject": _extract_subject(part),
                }
            ))

    # Also chunk by Schedule/Chapter headings
    schedule_pattern = r"(?:^|\n)\s*(SCHEDULE\s+[IVX]+[A-Z]?|CHAPTER\s+[IVX]+[A-Z]?)"
    schedule_splits = list(re.finditer(schedule_pattern, full_text, re.IGNORECASE))

    for i, match in enumerate(schedule_splits):
        start = match.start()
        end = schedule_splits[i + 1].start() if i + 1 < len(schedule_splits) else len(full_text)
        section_text = full_text[start:end].strip()
        section_name = match.group(1).strip()

        if len(section_text) < 50:
            continue

        parts = split_text_with_overlap(section_text, max_chars=3000, overlap=500)
        for j, part in enumerate(parts):
            chunk_id = generate_chunk_id(part, f"sched_{section_name.replace(' ', '_')}")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=part,
                metadata={
                    "regulation_number": section_name,
                    "source_file": source_file,
                    "chunk_type": "schedule",
                    "chunk_index": j,
                    "subject": _extract_subject(part),
                }
            ))

    return chunks


def _extract_subject(text: str, max_len: int = 80) -> str:
    """Extract a short subject description from the first line of text."""
    first_line = text.strip().split("\n")[0].strip()
    # Remove regulation number prefix
    first_line = re.sub(r"^(?:Regulation|Reg\.?)\s+\d+[A-Z]?\s*[.:\-–]?\s*", "", first_line)
    return first_line[:max_len]
