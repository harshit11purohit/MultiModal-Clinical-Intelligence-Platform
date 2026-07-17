# 🩺 MultiModal Clinical Intelligence Platform

**A production-grade Retrieval-Augmented Generation system for clinical report analysis**, combining hybrid search, cross-encoder reranking, and LLM-based safety guardrails to deliver structured, evidence-backed medical insights.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-000000)](https://www.pinecone.io/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**[🔗 Live Demo][![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/harshit11purohit/Medical_Chatbot_using_Llama2-2)

---

## 🎯 What This Solves

Patients get lab reports full of jargon and no context. MedIQ lets a user **upload a medical report (PDF or scanned image)** and instantly get a structured clinical breakdown — abnormal parameters, possible causes, dietary/lifestyle advice, and red flags — grounded in a retrieval pipeline instead of raw LLM hallucination, with a safety layer that rejects unsafe or unverified answers before they reach the user.

## 🏗️ System Architecture

```
                    ┌──────────────────────┐
                    │   User Query / File   │
                    │  (chat msg or upload) │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴─────────────────┐
              │                                    │
     ┌────────▼────────┐                ┌──────────▼──────────┐
     │  PDF / OCR Layer │                │   Intent Router (LLM) │
     │  (PyPDF/Tesseract)│               │  Greeting/General/Report│
     └────────┬────────┘                └──────────┬──────────┘
              │                                    │
              ▼                                    ▼
     ┌─────────────────────────────────────────────────────┐
     │              Hybrid Ensemble Retriever                │
     │      BM25 (keyword)  +  Pinecone (semantic, k=5)       │
     │                  weighted 50 / 50                      │
     └───────────────────────┬───────────────────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │  Cross-Encoder Reranker   │
                 │  (BAAI/bge-reranker-base) │
                 │       top_n = 3           │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Structured Clinical Prompt│
                 │  (Groq · Llama-3.1-8B)    │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │   LLM Safety Guardrail    │
                 │   PASS / REJECT decision  │
                 └────────────┬────────────┘
                              ▼
                     Structured JSON Response
```

## ✨ Key Features

- **Hybrid Retrieval Engine** — Combines sparse (BM25) and dense (Pinecone vector) search via `EnsembleRetriever` so answers aren't blind-sided by either pure keyword or pure semantic gaps.
- **Cross-Encoder Reranking** — A `BAAI/bge-reranker-base` model re-scores the top candidates for precision before they ever reach the LLM prompt.
- **Unified Upload + Query Pipeline** — A single `/upload` call accepts a file *and* an optional question in the same request, prioritizing the freshly uploaded document as context ahead of the long-term vector index — no separate "index then ask" round trip.
- **OCR for Scanned Reports** — Non-PDF uploads (photos of lab reports) are run through Tesseract OCR before entering the same retrieval pipeline as digital PDFs.
- **LLM Safety Guardrail** — Every generated answer is passed through a second-pass Groq call that explicitly rejects self-harm content, illegal guidance, or hallucinated medical claims before the user sees it.
- **Persistent Hybrid Index** — BM25 corpus is cached to disk (`bm25_docs_cache.pkl`) and merged incrementally on every upload, avoiding full re-ingestion on restart.
- **Full Observability** — Integrated with **LangSmith** tracing for prompt/response inspection across the entire retrieval → rerank → generate → guardrail chain.
- **Deduplication-Aware Ingestion** — Prevents redundant re-embedding of the same file while still allowing fresh questions against an already-indexed document.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | Flask + Flask-CORS |
| **LLM Inference** | Groq (Llama-3.1-8B-Instant) |
| **Orchestration** | LangChain / LangChain-Classic |
| **Vector Store** | Pinecone |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **Reranking** | HuggingFace Cross-Encoder (BAAI/bge-reranker-base) |
| **Document Parsing** | PyPDFLoader, PyPDFDirectoryLoader |
| **OCR** | Tesseract (pytesseract) |
| **Observability** | LangSmith |
| **Frontend** | React + Vite, Tailwind CSS (clinical dashboard UI) |
| **Embeddings** | Sentence-Transformers / HuggingFace |

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service metadata / endpoint index |
| `/health` | GET | Health check — LLM status + indexed chunk count |
| `/chat` | POST | Ask a question against the existing knowledge base (routes between greeting / general / report intents) |
| `/upload` | POST | Upload a PDF or image (optionally with a `msg` field) — indexes the file and answers the question in one call |

**Example — upload + ask in one request:**
```bash
curl -X POST http://localhost:8080/upload \
  -F "file=@blood_report.pdf" \
  -F "msg=What do my abnormal values indicate?"
```

**Example response shape:**
```json
{
  "response": "### Executive Summary\n...\n### Abnormal Parameters\n...\n### Red Flags\n..."
}
```

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/harshit11purohit/multimodal-clinical-intelligence-platform.git
cd multimodal-clinical-intelligence-platform

# 2. Install
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in: PINECONE_API_KEY, GROQ_API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

# 4. Initialize vector index / cache
python setup_database.py

# 5. Run the API
python app.py
```

Server boots on `http://localhost:8080`.

**Frontend (React + Vite + Tailwind):**
```bash
cd frontend
npm install
npm run dev
```
Dashboard boots on `http://localhost:5173` (already whitelisted in the Flask CORS config).







---

<p align="center">Built by <b>Harshit Purohit</b> · <a href="mailto:harshitpurohit953@gmail.com">harshitpurohit953@gmail.com</a> · <a href="#">LinkedIn</a> · <a href="#">GitHub</a></p>
