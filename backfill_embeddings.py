"""
Backfill embeddings for all legal_chunks rows where embedding IS NULL.
Run once with: .\\venv\\Scripts\\python.exe scratch/backfill_embeddings.py
"""
import os, sys, time
import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDER_URL = os.getenv("EMBEDDER_URL", "http://localhost:8081/embed")
EMBEDDER_DIM = int(os.getenv("EMBEDDER_DIM", "1536"))

def embed(text: str):
    try:
        r = requests.post(EMBEDDER_URL, json={"text": text}, timeout=10.0)
        r.raise_for_status()
        emb = r.json().get("embedding")
        if isinstance(emb, list) and len(emb) == EMBEDDER_DIM:
            return emb
    except Exception as e:
        print(f"  [WARN] Embedder error: {e}")
    return None

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM legal_chunks WHERE embedding IS NULL")
total = cur.fetchone()[0]
print(f"Found {total} chunks with NULL embeddings. Starting backfill...")

cur.execute("SELECT id, content FROM legal_chunks WHERE embedding IS NULL ORDER BY id")
rows = cur.fetchall()

success, failed = 0, 0
for i, (chunk_id, content) in enumerate(rows):
    emb = embed(content)
    if emb:
        cur.execute(
            "UPDATE legal_chunks SET embedding = %s WHERE id = %s",
            (emb, chunk_id)
        )
        success += 1
    else:
        failed += 1

    if (i + 1) % 50 == 0:
        conn.commit()
        print(f"  Progress: {i+1}/{total} | Success: {success} | Failed: {failed}")

conn.commit()
cur.close()
conn.close()

print(f"\nDone! Embedded {success}/{total} chunks. Failed: {failed}")
