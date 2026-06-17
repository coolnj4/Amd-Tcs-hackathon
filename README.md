# 🛡️ AI-Driven Audit & Compliance Validator

**TCS AMD Hackathon 2026** | AMD Instinct MI300X (192GB HBM) | ROCm + vLLM

An autonomous AI agent that validates IPO prospectuses (DRHP/RHP) against SEBI compliance regulations using a 5-layer validation pipeline with self-reflection, and produces auditable reports with multi-signal confidence scores.

---

## 🏗️ Architecture

```
User uploads DRHP PDF
        ↓
Document Intake Layer (PyMuPDF + pdfplumber + PaddleOCR)
        ↓
Two-RAG System (ChromaDB + bge-large-en-v1.5)
  ├── Compliance Collection (SEBI regulations — 700 chunks)
  └── Document Collection (uploaded DRHP — ~1,300 chunks per doc)
        ↓
103 SEBI Compliance Rules (ICDR + LODR + SAST)
        ↓
5-Layer Validation Engine
  ├── Layer 1: Deterministic (regex/pattern matching)
  ├── Layer 2: NLP Extraction (entity checks)
  ├── Layer 3: Semantic (LLM + RAG-powered assessment)
  ├── Layer 4: Numerical (financial cross-reference)
  └── Layer 5: Cross-Reference (text vs table mismatch)
        ↓
Self-Reflection Critic Agent (catches false positives)
        ↓
Multi-Signal Confidence Scoring (5 weighted signals)
        ↓
Audit Report (PDF + JSON) + Interactive Streamlit Dashboard
```

## ✨ Key Features

| Feature | Description |
|---|---|
| **103 SEBI Compliance Rules** | Comprehensive coverage across ICDR Schedule VI, LODR, and SAST regulations |
| **5-Layer Validation** | Deterministic + NLP + Semantic + Numerical + Cross-Reference checks |
| **Self-Reflection Critic** | Critic agent reviews all findings, catches false positives with reasoning |
| **Confidence Scoring** | 5-signal weighted formula (retrieval + evidence + LLM + critic + cross-validation) |
| **Full Auditability** | Every step logged with timestamps — 110+ step audit trail per document |
| **Cross-Reference** | Catches numerical mismatches between text and financial tables |
| **Interactive Dashboard** | Streamlit UI with charts, findings explorer, and pipeline metrics |
| **Magnitude-Aware Matching** | Prevents false alarms when comparing per-share vs aggregate amounts |

## 🔧 Tech Stack

| Component | Tool |
|---|---|
| LLM | Qwen/Qwen2.5-72B-Instruct via vLLM (bf16) |
| Embeddings | BAAI/bge-large-en-v1.5 (1024-dim) |
| Vector Store | ChromaDB (persistent) |
| PDF Parsing | PyMuPDF + pdfplumber + PaddleOCR |
| Orchestration | LangGraph state machine |
| Dashboard | Streamlit + Plotly |
| Report Export | fpdf2 (PDF) + JSON |
| GPU | AMD Instinct MI300X (192GB HBM3) |

## 🚀 Quick Start

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

### 4. Launch Streamlit Dashboard
```bash
# Install Streamlit (if not already)
pip install streamlit plotly

# Launch the dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The dashboard loads pre-cached audit results instantly. If no cached data exists,
it will look for report JSON files in `Amd-Tcs-hackathon/reports/`.

### 5. Data Setup
- Place SEBI regulation PDFs in `compliance_guidelines/` (or `Amd-Tcs-hackathon/compliance_guidelines/`)
- Place DRHP PDFs in `ipo_documents/` (or `Amd-Tcs-hackathon/ipo_documents/`)
- Run `download_dataset.py` to auto-download sample documents

## 📊 Performance Metrics

| Metric | Value |
|---|---|
| End-to-End Latency | ~19 min per 500-page DRHP |
| Parse Speed | ~1.9 pages/sec |
| Tables Extracted | 700+ per document |
| Rules Coverage | 103 SEBI rules (ICDR + LODR + SAST) |
| LLM Calls per Audit | ~200 (semantic + critic) |
| Estimated Tokens per Audit | ~400,000 |
| Critic False Positive Rate | 100% catch rate on cross-ref false alarms |
| GPU Memory | ~165 GB / 192 GB (vLLM 90% utilization) |

## 📁 Project Structure

```
├── app.py                      # Streamlit dashboard (interactive UI)
├── cache_results.py            # Pre-cache audit results for instant demo
├── demo_notebook.ipynb         # Complete demo notebook
├── download_dataset.py         # Dataset download script
├── requirements.txt            # Dependencies
├── compliance_guidelines/      # SEBI regulation PDFs (3 files)
├── ipo_documents/              # IPO DRHP PDFs (105 files, gitignored)
├── src/                        # Core application code
│   ├── config.py               # Configuration & constants
│   ├── llm_client.py           # vLLM API wrapper with token tracking
│   ├── document_parser.py      # PDF parsing + OCR
│   ├── chunker.py              # Section-aware chunking
│   ├── embeddings.py           # Embeddings + ChromaDB
│   ├── rule_generator.py       # Compliance rules management
│   ├── agent.py                # Pipeline orchestrator (LangGraph)
│   ├── critic.py               # Self-reflection critic agent
│   ├── confidence.py           # Multi-signal confidence scoring
│   ├── report_generator.py     # PDF report generation
│   ├── metrics.py              # Pipeline performance metrics
│   ├── default_rules.json      # 103 pre-built SEBI compliance rules
│   └── validators/
│       ├── deterministic.py    # Regex/pattern checks
│       ├── semantic.py         # LLM-powered RAG checks
│       └── cross_reference.py  # Numerical mismatch detection
└── shared/                     # Persistent storage (ChromaDB, caches)
    ├── chroma_db/              # ChromaDB vector index
    ├── compliance_rules.json   # Cached compliance rules
    └── audit_cache_*.json      # Pre-cached results for Streamlit
```

## 📈 Dataset

- **105 recent IPO DRHPs** scraped from official SEBI/BSE/NSE sources
- **3 SEBI Master Circulars**: ICDR, LODR, SAST regulations (~546 pages total)
- Run `python download_dataset.py` to download (requires ~2-3GB)

## 💻 Hardware Requirements

- **GPU**: AMD Instinct MI300X (192GB HBM3) or equivalent
- **RAM**: 64GB+ recommended
- **Storage**: 5GB for code + models, 3GB for dataset

## 👥 Team

Built for the **TCS AMD Hackathon 2026**
