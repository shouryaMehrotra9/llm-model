from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import shutil
import re
import glob
import logging
import requests
from typing import List
from contextlib import contextmanager

# Import LangChain components for PDF extraction and fallback splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import BM25 for sparse re-ranking
from rank_bm25 import BM25Okapi

# Import google-genai SDK
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import psycopg2 components for PostgreSQL and pgvector
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from pgvector.psycopg2 import register_vector

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("court_cases_rag")

app = FastAPI(title="Court Case RAG Q&A Backend")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables/configuration
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDER_URL = os.getenv("EMBEDDER_URL")
EMBEDDER_DIM = int(os.getenv("EMBEDDER_DIM", "2560"))
METADATA_FILE = "./metadata.json"
USER_PDFS_DIR = os.getenv("CASES_DIR", "./court_cases_db/8e6371b3-4bc5-4b8d-9f8c-5778a1986bd0")

pool = None
HAS_PGVECTOR = False

# Initialize Connection Pool
if DATABASE_URL:
    try:
        pool = SimpleConnectionPool(1, 20, dsn=DATABASE_URL)
        logger.info("PostgreSQL connection pool initialized successfully.")
    except Exception as ex:
        logger.error(f"Failed to initialize PostgreSQL pool (checking credentials or connection): {ex}")
else:
    logger.warning("DATABASE_URL is not set in environment variables.")

@contextmanager
def db_cursor():
    if not pool:
        raise ValueError("Database connection pool is not initialized. Please verify your DATABASE_URL in .env.")
    conn = pool.getconn()
    try:
        if HAS_PGVECTOR:
            try:
                register_vector(conn)
            except Exception as e:
                logger.debug(f"pgvector registration status: {e}")
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction error: {e}")
        raise e
    finally:
        pool.putconn(conn)

def embed_text(text: str) -> list[float] | None:
    if not EMBEDDER_URL:
        logger.warning("EMBEDDER_URL is not configured.")
        return None
    
    # Prevent self-deadlock if the URL points to our own FastAPI port
    if "localhost:5000" in EMBEDDER_URL or "127.0.0.1:5000" in EMBEDDER_URL:
        logger.warning("EMBEDDER_URL points to localhost:5000. Skipping embedding call to avoid deadlock.")
        return None
        
    try:
        r = requests.post(EMBEDDER_URL, json={"text": text}, timeout=2.0)
        r.raise_for_status()
        embedding = r.json().get("embedding")
        if isinstance(embedding, list) and len(embedding) == EMBEDDER_DIM:
            return embedding
        logger.warning(f"Unexpected embedding output format or dimension mismatch.")
        return None
    except Exception as e:
        logger.warning(f"Embedder error: {e}")
        return None

def extract_metadata(first_300_words: str, filename: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is not configured.")
        # Fallback
        return {
            "title": os.path.splitext(filename)[0].replace("_", " ").title(),
            "citation": None,
            "court": None,
            "decision_date": None
        }
        
    prompt = f"""
    You are a legal metadata extractor. Extract the title, citation, court, and decision date from the following text (first 300 words of a court judgment).
    
    TEXT:
    {first_300_words}
    
    Return a JSON object only. Format decision_date as YYYY-MM-DD. If a field cannot be found, use null.
    
    Fields:
    {{
      "title": "<case title>",
      "citation": "<citation>",
      "court": "<court name>",
      "decision_date": "<YYYY-MM-DD>"
    }}
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        meta = json.loads(response.text.strip())
        if not meta.get("title"):
            meta["title"] = os.path.splitext(filename)[0].replace("_", " ").title()
        return meta
    except Exception as e:
        logger.error(f"Error extracting metadata via Gemini: {e}")
        return {
            "title": os.path.splitext(filename)[0].replace("_", " ").title(),
            "citation": None,
            "court": None,
            "decision_date": None
        }

def call_gemini_chunker(full_text: str) -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
You are a legal document chunker. Split the following Indian court judgment into semantically meaningful chunks for a RAG system. The text is free-flowing prose — infer structure from content.

JUDGMENT TEXT:
{full_text}

Return a JSON array only. No preamble, no markdown fences.

Each element:
{{
  "chunk_index": <int starting at 0>,
  "section_role": "<head | facts | issues | reasoning | decision | other>",
  "content": "<verbatim chunk text>",
  "word_count": <int>,
  "citations_mentioned": ["<case citations>"],
  "statutes_mentioned": ["<acts/sections>"],
  "summary": "<one sentence>"
}}

ROLE GUIDE:
- head       → case name, court, bench, citation, date
- facts      → background facts, procedural history
- issues     → questions of law framed by the court
- reasoning  → court's analysis, precedent discussion
- decision   → operative holding, ratio, final order
- other      → anything else

CHUNKING RULES:
1. Each chunk self-contained and meaningful in isolation.
2. Target 150–300 words. Hard max 400 words.
3. Never split a sentence across chunks.
4. Preserve verbatim text in "content" — do not paraphrase.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )
    )
    
    text = response.text.strip()
    if text.startswith("```"):
        idx = text.find("\n")
        if idx != -1:
            text = text[idx:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
            
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ["chunks", "data", "elements"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
                
    if not isinstance(data, list):
        raise ValueError("Gemini chunker output is not a JSON array.")
        
    return data

def character_splitter_fallback(full_text: str) -> list[dict]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_text(full_text)
    chunks = []
    for idx, split in enumerate(splits):
        chunks.append({
            "chunk_index": idx,
            "section_role": "other",
            "content": split,
            "word_count": len(split.split()),
            "citations_mentioned": [],
            "statutes_mentioned": [],
            "summary": ""
        })
    return chunks

def get_files_from_db():
    try:
        with db_cursor() as cur:
            cur.execute(
                """
                SELECT j.source_file AS case_name, MAX(c.chunk_index) + 1 AS page_count
                FROM judgments j
                JOIN legal_chunks c ON c.judgment_id = j.id
                GROUP BY j.id, j.source_file;
                """
            )
            rows = cur.fetchall()
            files = []
            for row in rows:
                files.append({
                    "case_name": row[0],
                    "page_count": row[1]
                })
            return files
    except Exception as e:
        logger.error(f"Error fetching files from PostgreSQL: {e}")
        return []

def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_metadata(metadata):
    try:
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving metadata file: {e}")

def auto_index_pdfs():
    logger.info(f"Running auto-indexing for user PDFs in {USER_PDFS_DIR}...")
    if not os.path.exists(USER_PDFS_DIR):
        logger.warning(f"Directory {USER_PDFS_DIR} does not exist.")
        return
        
    pdf_pattern = os.path.join(USER_PDFS_DIR, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    if not pdf_files:
        logger.info("No PDF files found in user folder.")
        return
        
    existing_files = get_files_from_db()
    existing_names = {f["case_name"] for f in existing_files}
    
    logger.info(f"Found {len(pdf_files)} PDF files in user directory. Already indexed: {len(existing_names)}")
    
    added_any = False
    for filepath in pdf_files:
        filename = os.path.basename(filepath)
        if filename not in existing_names:
            logger.info(f"Indexing new PDF: {filename}")
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                full_text = "\n".join([page.page_content for page in pages])
                
                # Parse metadata from first 300 words
                words = full_text.split()
                first_300 = " ".join(words[:300])
                meta = extract_metadata(first_300, filename)
                
                # Insert into judgments
                judgment_id = None
                with db_cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO judgments (title, citation, court, decision_date, full_text, source_file)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (meta.get("title"), meta.get("citation"), meta.get("court"), meta.get("decision_date"), full_text, filename)
                    )
                    judgment_id = cur.fetchone()[0]
                    
                # Call Gemini chunker with fallback
                chunks = None
                if len(full_text) > 150000:
                    logger.warning(f"Document {filename} is extremely large ({len(full_text)} characters). Bypassing Gemini chunker directly to fallback splitter.")
                else:
                    for attempt in range(2):
                        try:
                            chunks = call_gemini_chunker(full_text)
                            break
                        except Exception as ex:
                            logger.warning(f"Gemini chunker attempt {attempt+1} failed for {filename}: {ex}")
                        
                if not chunks:
                    logger.warning(f"Falling back to character splitting for {filename}")
                    chunks = character_splitter_fallback(full_text)
                    
                # Insert chunks
                with db_cursor() as cur:
                    for chunk in chunks:
                        content = chunk.get("content", "").strip()
                        if not content:
                            continue
                        embedding = embed_text(content)
                        
                        cur.execute(
                            """
                            INSERT INTO legal_chunks (
                                judgment_id, chunk_index, section_role, content, word_count, 
                                citations_mentioned, statutes_mentioned, summary, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                            """,
                            (
                                judgment_id, 
                                chunk.get("chunk_index", 0), 
                                chunk.get("section_role", "other"), 
                                content, 
                                chunk.get("word_count", len(content.split())),
                                chunk.get("citations_mentioned", []), 
                                chunk.get("statutes_mentioned", []), 
                                chunk.get("summary", ""), 
                                embedding
                            )
                        )
                logger.info(f"Successfully indexed {filename} ({len(pages)} pages, {len(chunks)} chunks)")
                added_any = True
            except Exception as e:
                logger.error(f"Failed to index {filename}: {e}")
        else:
            logger.debug(f"Skipping {filename} (already indexed)")
            
    if added_any:
        files = get_files_from_db()
        save_metadata(files)
        logger.info("Updated metadata.json after auto-indexing.")

@app.on_event("startup")
async def startup_event():
    global HAS_PGVECTOR
    # 1. Connect to database and execute schema.sql if tables don't exist
    if pool:
        try:
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    # Check if pgvector extension is available on the system
                    try:
                        cur.execute("SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector');")
                        HAS_PGVECTOR = cur.fetchone()[0]
                    except Exception:
                        HAS_PGVECTOR = False
                    logger.info(f"pgvector extension availability status: {HAS_PGVECTOR}")
                    
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'judgments'
                        );
                        """
                    )
                    exists = cur.fetchone()[0]
                    if not exists:
                        if HAS_PGVECTOR:
                            logger.info("judgments table does not exist. Initializing schema.sql with pgvector...")
                            with open("schema.sql", "r", encoding="utf-8") as f:
                                schema_content = f.read()
                            schema_content = schema_content.replace("{{INSERT_DIM}}", str(EMBEDDER_DIM))
                            cur.execute(schema_content)
                        else:
                            logger.info("judgments table does not exist and pgvector is unavailable. Initializing fallback schema...")
                            fallback_schema = f"""
                            CREATE TABLE IF NOT EXISTS judgments (
                                id SERIAL PRIMARY KEY,
                                title TEXT,
                                citation TEXT,
                                court TEXT,
                                decision_date DATE,
                                full_text TEXT,
                                source_file TEXT UNIQUE,
                                created_at TIMESTAMPTZ DEFAULT NOW()
                            );

                            CREATE TABLE IF NOT EXISTS legal_chunks (
                                id SERIAL PRIMARY KEY,
                                judgment_id INTEGER REFERENCES judgments(id) ON DELETE CASCADE,
                                chunk_index INTEGER,
                                section_role TEXT,
                                content TEXT,
                                word_count INTEGER,
                                citations_mentioned TEXT[],
                                statutes_mentioned TEXT[],
                                summary TEXT,
                                embedding real[],
                                content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                                created_at TIMESTAMPTZ DEFAULT NOW()
                            );

                            CREATE INDEX IF NOT EXISTS legal_chunks_tsv_idx ON legal_chunks USING gin (content_tsv);

                            CREATE OR REPLACE FUNCTION cosine_similarity(a real[], b real[]) RETURNS double precision AS $$
                            DECLARE
                                dot double precision := 0;
                                norm_a double precision := 0;
                                norm_b double precision := 0;
                                i integer;
                            BEGIN
                                IF a IS NULL OR b IS NULL THEN
                                    RETURN 0;
                                END IF;
                                FOR i IN 1..cardinality(a) LOOP
                                    dot := dot + a[i] * b[i];
                                    norm_a := norm_a + a[i] * a[i];
                                    norm_b := norm_b + b[i] * b[i];
                                END LOOP;
                                IF norm_a = 0 OR norm_b = 0 THEN
                                    RETURN 0;
                                END IF;
                                RETURN dot / (sqrt(norm_a) * sqrt(norm_b));
                            END;
                            $$ LANGUAGE plpgsql;
                            """
                            cur.execute(fallback_schema)
                        conn.commit()
                        logger.info("Schema initialized successfully.")
            finally:
                pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error running database schema initialization on startup: {e}")
            
        # 2. Trigger auto indexing
        try:
            auto_index_pdfs()
        except Exception as e:
            logger.error(f"Error during auto-indexing on startup: {e}")
    else:
        logger.warning("Database connection pool is not initialized. Skipping schema checks and auto-indexing.")

def calculate_word_overlap(answer: str, context: str) -> float:
    def get_words(text: str):
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words)
    
    answer_words = get_words(answer)
    context_words = get_words(context)
    
    if not answer_words:
        return 1.0
        
    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words)

def rrf(dense_ids, sparse_ids, k=60):
    scores = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def bm25_search(question: str, top_k: int = 20) -> list[int]:
    """
    Two-stage BM25 search:
    Stage 1: PostgreSQL FTS pre-filter (websearch_to_tsquery) — gets up to 50 candidate chunk IDs fast.
    Stage 2: BM25Okapi re-ranking — applies true BM25 scoring on the candidate chunks' actual content.
    Returns a list of chunk IDs ranked by BM25 score (best first).
    """
    try:
        websearch_query = to_websearch_or(question)
        if not websearch_query:
            return []

        # Stage 1: FTS pre-filter — fetch candidate chunk IDs + content
        with db_cursor() as cur:
            cur.execute(
                """
                SELECT id, content
                FROM legal_chunks
                WHERE content_tsv @@ websearch_to_tsquery('english', %s)
                LIMIT 50;
                """,
                (websearch_query,)
            )
            rows = cur.fetchall()

        if not rows:
            return []

        candidate_ids = [row[0] for row in rows]
        candidate_texts = [row[1] for row in rows]

        # Stage 2: BM25 re-ranking on candidate content
        tokenized_corpus = [doc.lower().split() for doc in candidate_texts]
        tokenized_query = question.lower().split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Rank candidates by BM25 score (descending)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_ids = [candidate_ids[i] for i in ranked_indices[:top_k] if scores[i] > 0]

        logger.info(f"BM25 search returned {len(ranked_ids)} results from {len(candidate_ids)} FTS candidates.")
        return ranked_ids

    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return []

@app.get("/", response_class=HTMLResponse)
def get_frontend():
    index_path = os.path.join(".", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Frontend index.html not found in project root.</h3>"

@app.get("/files")
def get_files():
    files = get_files_from_db()
    if files:
        save_metadata(files)
    return {"status": "success", "files": files}

@app.get("/cases")
def get_cases():
    files = get_files_from_db()
    return files

@app.get("/pdf/{filename}")
def get_pdf(filename: str):
    filename = os.path.basename(filename)
    pdf_path = os.path.join(USER_PDFS_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file '{filename}' not found.")
    return FileResponse(pdf_path, media_type="application/pdf")

@app.get("/health")
def get_health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    current_files = get_files_from_db()
    current_count = len(current_files)
    
    if current_count + len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of 10 PDFs can be uploaded in total. Currently uploaded: {current_count}. Trying to upload: {len(files)}."
        )
    
    temp_dir = os.path.join(".", "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    existing_names = {f["case_name"] for f in current_files}
    new_files_metadata = []
    
    try:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF.")
            
            if file.filename in existing_names:
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' has already been uploaded.")
                
            temp_file_path = os.path.join(USER_PDFS_DIR, file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            try:
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()
                page_count = len(pages)
                full_text = "\n".join([page.page_content for page in pages])
                
                # Parse metadata from first 300 words
                words = full_text.split()
                first_300 = " ".join(words[:300])
                meta = extract_metadata(first_300, file.filename)
                
                # Insert into judgments
                judgment_id = None
                with db_cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO judgments (title, citation, court, decision_date, full_text, source_file)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (meta.get("title"), meta.get("citation"), meta.get("court"), meta.get("decision_date"), full_text, file.filename)
                    )
                    judgment_id = cur.fetchone()[0]
                    
                # Call Gemini chunker
                chunks = None
                if len(full_text) > 150000:
                    logger.warning(f"Document {file.filename} is extremely large ({len(full_text)} characters). Bypassing Gemini chunker directly to fallback splitter.")
                else:
                    for attempt in range(2):
                        try:
                            chunks = call_gemini_chunker(full_text)
                            break
                        except Exception as ex:
                            logger.warning(f"Gemini chunker failed during upload of {file.filename}: {ex}")
                        
                if not chunks:
                    logger.warning(f"Falling back to character splitting for {file.filename}")
                    chunks = character_splitter_fallback(full_text)
                    
                # Insert chunks
                with db_cursor() as cur:
                    for chunk in chunks:
                        content = chunk.get("content", "").strip()
                        if not content:
                            continue
                        embedding = embed_text(content)
                        
                        cur.execute(
                            """
                            INSERT INTO legal_chunks (
                                judgment_id, chunk_index, section_role, content, word_count, 
                                citations_mentioned, statutes_mentioned, summary, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                            """,
                            (
                                judgment_id, 
                                chunk.get("chunk_index", 0), 
                                chunk.get("section_role", "other"), 
                                content, 
                                chunk.get("word_count", len(content.split())),
                                chunk.get("citations_mentioned", []), 
                                chunk.get("statutes_mentioned", []), 
                                chunk.get("summary", ""), 
                                embedding
                            )
                        )
                
                file_meta = {"case_name": file.filename, "page_count": page_count}
                current_files.append(file_meta)
                new_files_metadata.append(file_meta)
                existing_names.add(file.filename)
                
            finally:
                pass
                    
        # Update metadata.json
        save_metadata(current_files)
        
        # Clean up temp dir
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass
            
        return {"status": "success", "uploaded": new_files_metadata, "all_files": current_files}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process files: {str(e)}")

def to_websearch_or(query_text: str) -> str:
    words = re.findall(r'\b\w+\b', query_text)
    stopwords = {
        'what', 'is', 'the', 'between', 'and', 'according', 'to', 'of', 'in', 'on', 'at', 
        'by', 'for', 'with', 'about', 'against', 'during', 'before', 'after', 'above', 
        'below', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again', 
        'further', 'then', 'once', 'a', 'an', 'or', 'this', 'that', 'these', 'those'
    }
    filtered_words = [w for w in words if w.lower() not in stopwords]
    if not filtered_words:
        filtered_words = words
    return " OR ".join(filtered_words)

def handle_gemini_error(e: Exception) -> str:
    err_msg = str(e)
    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg:
        return (
            "⚠️ **Gemini API Quota Exceeded**\n\n"
            "Your Gemini API Key has exceeded its Free Tier quota limit (20 requests per day).\n\n"
            "To fix this, you can:\n"
            "- Wait for the daily free quota to reset (typically resets at midnight PST).\n"
            "- Upgrade your Google AI Studio account to a pay-as-you-go plan (which features much higher free limits before billing commences).\n"
            "- Change the `GEMINI_API_KEY` in your [.env](file:///C:/llm%20model/.env) file to a billing-enabled key."
        )
    elif "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg:
        return (
            "⚠️ **Invalid Gemini API Key**\n\n"
            "The Gemini API key configured in your [.env](file:///C:/llm%20model/.env) file is invalid.\n\n"
            "Please check your key and update it in your [.env](file:///C:/llm%20model/.env) file."
        )
    else:
        return f"An error occurred while generating the answer from Gemini: {err_msg}"

@app.post("/query")
async def query_cases(payload: dict = Body(...)):
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    # Check if Gemini API Key is configured
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API Key is not set on the backend. Please create a .env file containing GEMINI_API_KEY=your_key."
        )
        
    try:
        # Step 1: Call external HTTP embedder
        query_vector = embed_text(question)
        
        # Step 2: Dense Cosine search (pgvector or fallback PL/pgSQL)
        dense_results = []
        if query_vector:
            try:
                with db_cursor() as cur:
                    if HAS_PGVECTOR:
                        cur.execute(
                            """
                            SELECT id, similarity FROM (
                                SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                                FROM legal_chunks
                                WHERE embedding IS NOT NULL
                                ORDER BY embedding <=> %s::vector
                                LIMIT 20
                            ) sub;
                            """,
                            (query_vector, query_vector)
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, similarity FROM (
                                SELECT id, cosine_similarity(embedding, %s::real[]) AS similarity
                                FROM legal_chunks
                                WHERE embedding IS NOT NULL
                                ORDER BY similarity DESC
                                LIMIT 20
                            ) sub;
                            """,
                            (query_vector,)
                        )
                    dense_results = cur.fetchall()
            except Exception as e:
                logger.error(f"Dense search error: {e}")
                
        # Step 3: BM25 sparse search (two-stage: FTS pre-filter + BM25Okapi re-ranking)
        sparse_results = []
        try:
            bm25_ids = bm25_search(question, top_k=20)
            sparse_results = [(cid,) for cid in bm25_ids]
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            
        dense_ids = [row[0] for row in dense_results]
        sparse_ids = [row[0] for row in sparse_results]
        similarity_map = {row[0]: row[1] for row in dense_results}
        
        # Step 4: Fusion using RRF
        fused_ids = rrf(dense_ids, sparse_ids, k=60)[:10]
        
        if not fused_ids:
            # Fallback to answering using general legal knowledge rather than throwing a 404
            client = genai.Client(api_key=api_key)
            system_prompt = (
                "You are a legal research assistant. The user's query could not be matched to any documents in the local database. "
                "Answer the query using your general legal knowledge of Indian law (such as IPC, CrPC, statutory provisions, or relevant case law). "
                "Your answer must be returned as a formal LEGAL RESEARCH REPORT using this exact structure:\n\n"
                "LEGAL RESEARCH REPORT\n\n"
                "1. SUMMARY OF INQUIRY\n"
                "---------------------\n"
                "[Brief overview of the legal question being analyzed]\n\n"
                "2. GENERAL LEGAL STANDARDS\n"
                "--------------------------\n"
                "[List bullet points of general legal facts or statutory provisions, indicating they are from general statutory knowledge rather than the database]\n\n"
                "3. DETAILED LEGAL ANALYSIS\n"
                "--------------------------\n"
                "[Thorough narrative analysis of the question, explaining the legal principles and statutory provisions]\n\n"
                "4. CONCLUSION\n"
                "-------------\n"
                "[Concluding findings based on general legal knowledge]\n\n"
                "Important Rule:\n"
                "- Prefix the report with this exact warning notice line at the very beginning:\n"
                "[NOTICE: No matching content was found in the database. The following analysis is generated based on general legal knowledge.]\n\n"
                "- Maintain an objective, professional legal research tone."
            )
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Question: {question}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0
                    )
                )
                answer = response.text.strip()
            except Exception as e:
                logger.error(f"Gemini fallback call failed: {e}")
                answer = handle_gemini_error(e)

            return {
                "status": "success",
                "confidence": 0.0,
                "hallucination": 0.0,
                "hallucination_risk": 0.0,
                "answer": answer,
                "chunks": [],
                "sources": []
            }
            
        # Step 5: Fetch rows for top 5 chunk IDs
        chunks_retrieved = []
        distances = []
        try:
            with db_cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.content, c.chunk_index, c.section_role, c.summary, j.source_file, j.title
                    FROM legal_chunks c
                    JOIN judgments j ON c.judgment_id = j.id
                    WHERE c.id IN %s;
                    """,
                    (tuple(fused_ids),)
                )
                rows = {row[0]: row for row in cur.fetchall()}
                
                for chunk_id in fused_ids:
                    if chunk_id in rows:
                        row = rows[chunk_id]
                        similarity = similarity_map.get(chunk_id, 0.5)
                        distances.append(similarity)
                        chunks_retrieved.append({
                            "content": row[1],
                            "case_name": row[5], # Use source_file (filename) to keep frontend links working
                            "page": row[2] + 1,  # chunk_index + 1
                            "similarity": similarity,
                            "section_role": row[3],
                            "summary": row[4],
                            "title": row[6]
                        })
        except Exception as e:
            logger.error(f"Error fetching retrieved chunks: {e}")
            raise HTTPException(status_code=500, detail=f"Query error during retrieval: {e}")
            
        if not distances:
            raise HTTPException(status_code=404, detail="No matching chunks could be fetched.")
            
        # Step 6: CRAG Evaluator & Refiner
        crag_graded_chunks = []
        crag_correct_count = 0
        crag_ambiguous_count = 0
        crag_incorrect_count = 0
        
        for chunk in chunks_retrieved:
            sim = chunk["similarity"]
            if sim >= 0.55:
                grade = "correct"
                crag_correct_count += 1
            elif sim >= 0.25:
                grade = "ambiguous"
                crag_ambiguous_count += 1
            else:
                grade = "incorrect"
                crag_incorrect_count += 1
            
            chunk["crag_grade"] = grade
            if grade in ("correct", "ambiguous"):
                crag_graded_chunks.append(chunk)
                
        logger.info(f"CRAG evaluation result: {crag_correct_count} correct, {crag_ambiguous_count} ambiguous, {crag_incorrect_count} incorrect.")
        
        if not crag_graded_chunks:
            logger.info("CRAG Action: All chunks evaluated as INCORRECT. Triggering general knowledge corrective fallback.")
            client = genai.Client(api_key=api_key)
            system_prompt = (
                "You are a legal research assistant. The retrieved case excerpts were evaluated as completely irrelevant to the query. "
                "Answer the query using your general legal knowledge of Indian law (such as IPC, CrPC, statutory provisions, or relevant case law). "
                "Your answer must be returned as a formal LEGAL RESEARCH REPORT using this exact structure:\n\n"
                "LEGAL RESEARCH REPORT\n\n"
                "1. SUMMARY OF INQUIRY\n"
                "---------------------\n"
                "[Brief overview of the legal question being analyzed]\n\n"
                "2. GENERAL LEGAL STANDARDS\n"
                "--------------------------\n"
                "[List bullet points of general legal facts or statutory provisions, indicating they are from general statutory knowledge rather than the database]\n\n"
                "3. DETAILED LEGAL ANALYSIS\n"
                "--------------------------\n"
                "[Thorough narrative analysis of the question, explaining the legal principles and statutory provisions]\n\n"
                "4. CONCLUSION\n"
                "-------------\n"
                "[Concluding findings based on general legal knowledge]\n\n"
                "Important Rule:\n"
                "- Prefix the report with this exact warning notice line at the very beginning:\n"
                "[NOTICE: Retrieved database documents were evaluated as irrelevant to the query. The following analysis is generated based on general legal knowledge.]\n\n"
                "- Maintain an objective, professional legal research tone."
            )
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Question: {question}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0
                    )
                )
                answer = response.text.strip()
            except Exception as e:
                logger.error(f"Gemini fallback call failed: {e}")
                answer = handle_gemini_error(e)

            return {
                "status": "success",
                "confidence": 0.0,
                "hallucination": 0.0,
                "hallucination_risk": 0.0,
                "answer": answer,
                "chunks": [],
                "sources": []
            }
            
        logger.info(f"CRAG Action: Refined context created containing {len(crag_graded_chunks)} relevant/ambiguous chunks.")
        
        # Calculate Confidence Score based on refined chunks
        refined_distances = [c["similarity"] for c in crag_graded_chunks]
        top_similarity = refined_distances[0]
        avg_similarity = sum(refined_distances) / len(refined_distances)
        confidence = (0.6 * top_similarity + 0.4 * avg_similarity) * 100.0
        confidence = round(confidence, 1)
        
        # We will generate the answer even if confidence is below 40%, but we will flag the status accordingly
        status = "success" if confidence >= 40.0 else "insufficient_confidence"
            
        # Step 7: Call Gemini gemini-2.5-flash with refined context
        client = genai.Client(api_key=api_key)
        context_parts = []
        for idx, chunk in enumerate(crag_graded_chunks):
            # Use human-readable title, fallback to filename if title is missing
            display_name = chunk.get('title') or chunk['case_name']
            summary_line = f"\n  [Summary: {chunk['summary']}]" if chunk.get('summary') else ""
            context_parts.append(
                f"[Excerpt {idx+1}] Case: {display_name} | Section: {chunk['section_role']} | Page: {chunk['page']}{summary_line}\n{chunk['content']}"
            )
        context_text = "\n\n---\n\n".join(context_parts)
        
        system_prompt = (
            "You are an expert Indian legal research assistant with deep knowledge of constitutional law, IPC, CrPC, and Supreme Court jurisprudence.\n\n"
            "You will be given excerpts from legal documents retrieved from a database. Your task is to answer the user's legal question "
            "as thoroughly and accurately as possible using these excerpts as your primary source of truth.\n\n"
            "INSTRUCTIONS:\n"
            "1. Read ALL provided excerpts carefully before answering.\n"
            "2. Synthesise information across multiple excerpts to form a complete answer.\n"
            "3. For every factual claim or legal finding, cite the case title and page number in parentheses, e.g. (Kesavananda Bharati v. State of Kerala, Page 5).\n"
            "4. If a concept appears in multiple excerpts, mention all of them.\n"
            "5. Structure your answer in clear sections:\n"
            "   - OVERVIEW: 2-3 sentences summarising what the documents say about the query.\n"
            "   - KEY LEGAL FINDINGS: Bullet points of the most important facts, holdings, or provisions found.\n"
            "   - DETAILED ANALYSIS: Thorough paragraph-form analysis drawing from the excerpts.\n"
            "   - CONCLUSION: A clear, direct answer to the user's question.\n"
            "6. Do NOT say 'Not found in retrieved documents' unless you have genuinely exhausted all excerpts.\n"
            "7. Do NOT make up facts, dates, names, or amounts not present in the excerpts.\n"
            "8. Write in a professional legal tone — precise, analytical, and well-organised."
        )
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Case Excerpts:\n{context_text}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0
                )
            )
            answer = response.text.strip()
            
            # Step 8: Calculate Hallucination Risk
            refusal_phrases = ["not found in retrieved documents", "not found in the retrieved documents", "information not found"]
            is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)
            
            if is_refusal:
                hallucination_risk = 0.0
            else:
                word_overlap = calculate_word_overlap(answer, context_text)
                hallucination_risk = ((1.0 - word_overlap) * 0.6 + (1.0 - avg_similarity) * 0.4) * 100.0
                hallucination_risk = max(0.0, min(100.0, hallucination_risk))
                
            hallucination_risk = round(hallucination_risk, 1)
        except Exception as e:
            logger.error(f"Gemini normal content generation failed: {e}")
            answer = handle_gemini_error(e)
            hallucination_risk = 0.0
        
        # Step 9: Format unique sources
        unique_sources = []
        seen_sources = set()
        for chunk in crag_graded_chunks:
            source_key = (chunk["case_name"], chunk["page"])
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append({
                    "case_name": chunk["case_name"],
                    "page": chunk["page"],
                    "page_number": chunk["page"],
                    "similarity": chunk["similarity"],
                    "text": chunk["content"]
                })
                
        return {
            "status": status,
            "confidence": confidence,
            "hallucination": hallucination_risk,
            "hallucination_risk": hallucination_risk,
            "answer": answer,
            "chunks": crag_graded_chunks,
            "sources": unique_sources
        }
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/reset")
def reset_db():
    try:
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
        with db_cursor() as cur:
            cur.execute("TRUNCATE TABLE judgments CASCADE;")
        logger.info("Database cleared successfully via reset route.")
        return {"status": "success", "message": "Database and file list successfully cleared."}
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
