# 🏆 AI-Driven Audit & Compliance Validator

**TCS AMD Hackathon 2026** | AMD Instinct MI300X (192GB HBM) | ROCm + vLLM

An AI agent that validates IPO prospectuses (DRHP/RHP) against SEBI compliance regulations and produces auditable reports with confidence scores.

---

## Architecture

```
User uploads DRHP PDF
        ↓
Document Intake Layer (PyMuPDF + pdfplumber + PaddleOCR)
        ↓
Two-RAG System (ChromaDB + bge-large-en-v1.5)
  ├── Compliance Collection (SEBI regulations)
  └── Document Collection (uploaded DRHP)
        ↓
LLM Auto-generates compliance rules from SEBI PDFs
        ↓
5-Layer Validation Engine
  ├── Deterministic (regex/pattern matching)
  ├── NLP Extraction (entity checks)
  ├── Semantic (LLM-powered assessment)
  ├── Numerical (financial table verification)
  └── Cross-Reference (text vs table mismatch)
        ↓
Self-Reflection Critic Agent (catches false positives)
        ↓
Multi-Signal Confidence Scoring
        ↓
Audit Report (PDF + JSON) with full audit trail
```

## Key Features

| Feature | Description |
|---|---|
| **Auto Rule Generation** | LLM reads SEBI regulations and creates its own compliance checklist |
| **5-Layer Validation** | Deterministic + NLP + Semantic + Numerical + Cross-Reference |
| **Self-Reflection** | Critic agent catches false positives with reasoning |
| **Confidence Scoring** | 5-signal weighted formula (retrieval + evidence + LLM + critic + cross-val) |
| **Full Auditability** | Every step logged with timestamps in visual audit trail |
| **Cross-Reference** | Catches numerical mismatches between text and tables |

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Qwen/Qwen2.5-72B-Instruct via vLLM (bf16) |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector Store | ChromaDB (persistent) |
| PDF Parsing | PyMuPDF + pdfplumber + PaddleOCR |
| UI | Streamlit |
| Report Export | fpdf2 |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start vLLM Server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

### 3. Run the Demo Notebook
Open `demo_notebook.ipynb` in JupyterLab and run cells sequentially.

### 4. Data Setup
- Place SEBI regulation PDFs in `compliance_guidelines/`
- Place DRHP PDFs in `ipo_documents/`
- Run `download_dataset.py` to auto-download the dataset

## Project Structure

```
├── compliance_guidelines/      # SEBI regulation PDFs (3 files, ~30MB)
├── ipo_documents/              # IPO DRHP PDFs (100 files, ~1GB, gitignored)
├── src/                        # Core application code
│   ├── config.py               # Configuration & constants
│   ├── llm_client.py           # vLLM API wrapper
│   ├── document_parser.py      # PDF parsing + OCR
│   ├── chunker.py              # Section-aware chunking
│   ├── embeddings.py           # Embeddings + ChromaDB
│   ├── rule_generator.py       # Auto-generate compliance rules
│   ├── agent.py                # Pipeline orchestrator
│   ├── critic.py               # Self-reflection critic
│   ├── confidence.py           # Confidence scoring
│   ├── report_generator.py     # PDF report generation
│   └── validators/
│       ├── deterministic.py    # Regex/pattern checks
│       ├── semantic.py         # LLM-powered checks
│       └── cross_reference.py  # Numerical mismatch detection
├── demo_notebook.ipynb         # Complete demo notebook
├── download_dataset.py         # Dataset download script
├── metadata.csv                # Dataset index
└── requirements.txt            # Dependencies
```

## Dataset

- **100 recent IPO DRHPs** scraped from official sources
- **3 SEBI Master Circulars**: ICDR, LODR, SAST regulations
- Run `python download_dataset.py` to download (requires ~2-3GB)

## Hardware Requirements

- **GPU**: AMD Instinct MI300X (192GB HBM) or equivalent
- **RAM**: 64GB+ recommended
- **Storage**: 5GB for code + models, 3GB for dataset

## Team

Built for the TCS AMD Hackathon 2026
