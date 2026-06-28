"""
GPU-Accelerated Direct Backfill:
1. Uses device="cuda" to run on the RTX 4060 Laptop GPU.
2. Loads model in float16 for maximum Tensor Core speed and minimal VRAM footprint.
3. Uses a large batch size of 64 chunks for maximum parallel GPU execution.

Run: .\\venv\\Scripts\\python.exe backfill_embeddings.py
"""
import os, time, torch
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Disable parallel tokenization warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_NAME   = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
BATCH_SIZE   = 64   # Large batch size for GPU parallelization
EMBEDDER_DIM = int(os.getenv("EMBEDDER_DIM", "1536"))

print(f"Loading model {MODEL_NAME} directly onto NVIDIA GPU (RTX 4060) in float16...")
model = SentenceTransformer(
    MODEL_NAME, 
    trust_remote_code=True, 
    device='cuda',
    model_kwargs={'torch_dtype': torch.float16}
)

# Disable cache for embedding generation (no autoregressive decoding needed)
model.config.use_cache = False
model._first_module().auto_model.config.use_cache = False

print(f"Model loaded successfully on GPU!")

conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()

cur.execute("SELECT COUNT(*) FROM legal_chunks WHERE embedding IS NULL")
total = cur.fetchone()[0]
print(f"Found {total} chunks with NULL embeddings. Starting GPU-accelerated backfill (batch={BATCH_SIZE})...")

cur.execute("SELECT id, content FROM legal_chunks WHERE embedding IS NULL ORDER BY id")
rows = cur.fetchall()

success, failed = 0, 0
t0 = time.time()

for batch_start in range(0, len(rows), BATCH_SIZE):
    batch  = rows[batch_start : batch_start + BATCH_SIZE]
    ids    = [r[0] for r in batch]
    texts  = [r[1] for r in batch]

    try:
        # Encode on GPU
        vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
        for chunk_id, vec in zip(ids, vectors):
            emb = vec.tolist()
            if len(emb) == EMBEDDER_DIM:
                cur.execute("UPDATE legal_chunks SET embedding = %s WHERE id = %s", (emb, chunk_id))
                success += 1
            else:
                failed += 1
        conn.commit()
    except Exception as e:
        failed += len(batch)
        print(f"  [WARN] Batch {batch_start} failed: {e}")

    done    = batch_start + len(batch)
    elapsed = time.time() - t0
    rate    = done / elapsed if elapsed > 0 else 0
    eta     = (total - done) / rate if rate > 0 else 0
    print(f"  [{done}/{total}] {round(done/total*100,1)}% | {round(rate,1)} chunks/s | ETA {round(eta)}s")

conn.commit()
cur.close()
conn.close()
print(f"\n[SUCCESS] Done! GPU backfill complete: {success}/{total} chunks embedded. Failed: {failed}")
