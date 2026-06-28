# ⚖️ Court Case RAG System

A **Corrective RAG (CRAG)** pipeline for querying legal documents (court judgements, Constitution of India, etc.) using:
- 🔍 **Local Qwen embeddings** (GTE-Qwen2-1.5B) for dense vector search
- 🗄️ **PostgreSQL** for full-text and vector search (via pgvector)
- 🤖 **Gemini 2.5 Flash** for LLM-powered answer generation
- 📄 **FastAPI** backend + HTML/JS frontend

---

## 🛠️ Prerequisites

Make sure the following are installed on your system:

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 - 3.13 | Must match your venv |
| PostgreSQL | 14+ | With the `pgvector` extension |
| Git | Any | To clone the repo |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/shouryaMehrotra9/llm-model.git
cd llm-model
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:
- **Windows**: `.\venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

---

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Then install the embedding server dependencies:

```bash
pip install sentence-transformers fastapi uvicorn
```

---

### 4. Set Up PostgreSQL

Make sure PostgreSQL is running. Then create the database:

```sql
psql -U postgres
CREATE DATABASE court_cases;
\q
```

Optionally install `pgvector` for faster vector search (the app works without it too):
```sql
psql -U postgres -d court_cases
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

---

### 5. Configure Environment Variables

Copy the example env file and fill in your details:

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here

DATABASE_URL=postgresql://postgres@localhost:5432/court_cases
EMBEDDER_URL=http://localhost:8081/embed
EMBEDDER_DIM=1536
```

> **Get a free Gemini API key** at: https://aistudio.google.com/app/apikey

---

### 6. Start the Local Qwen Embedding Server

In a **separate terminal**, run:

```bash
python qwen_embed_server.py
```

This will:
- Automatically **download** the `Alibaba-NLP/gte-Qwen2-1.5B-instruct` model (~3 GB) from Hugging Face on first run.
- Start a local FastAPI server on **`http://localhost:8081`** serving the `/embed` endpoint.

> ⚠️ **First run takes 5-10 minutes** to download the model. Subsequent runs are instant.

> ⚠️ **Windows users**: If you hit a `rope_theta` or `get_usable_length` error, run the patch script:
> ```bash
> python scratch/patch_qwen.py
> ```
> Then restart the embedding server.

---

### 7. Start the Main Application Server

In a **different terminal**, run:

```bash
uvicorn main:app --host 127.0.0.1 --port 5000
```

---

### 8. Open the Application

Open your browser and navigate to:

👉 **http://127.0.0.1:5000**

---

## 📂 Adding Your Own Documents

1. Open the app in your browser.
2. Use the **Upload PDF** button to add your own court case documents.
3. Uploaded PDFs are automatically chunked and indexed into the database.

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
FastAPI Backend (port 5000)
    │
    ├─► Qwen Embedder (port 8081) ──► Dense Vector Search (PostgreSQL)
    │
    ├─► Full-Text Search (PostgreSQL tsvector)
    │
    ├─► RRF Fusion (combines dense + sparse results)
    │
    ├─► CRAG Grader (filters relevant chunks by similarity score)
    │       ├─ score ≥ 0.55 → Correct  ✅
    │       ├─ score ≥ 0.40 → Ambiguous ⚠️
    │       └─ score < 0.40 → Incorrect ❌
    │
    └─► Gemini 2.5 Flash (generates final answer)
```

---

## 🔑 Running Without Local Embeddings

If you don't want to run the Qwen embedding server, the system will automatically fall back to **Gemini's built-in embeddings** for document indexing (uses API credits).

To use this fallback mode, simply don't start `qwen_embed_server.py`.

---

## 🌐 Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Frontend | Vanilla HTML + CSS + JS |
| Database | PostgreSQL + pgvector |
| Embeddings | GTE-Qwen2-1.5B (local) |
| LLM | Gemini 2.5 Flash |
| RAG Strategy | Corrective RAG (CRAG) |
