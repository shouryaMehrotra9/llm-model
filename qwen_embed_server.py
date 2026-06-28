import os
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Qwen Embedding Server")

# Default model: Alibaba-NLP/gte-Qwen2-1.5B-instruct
# Note: You can change this to any model you prefer.
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
model = None

class EmbedRequest(BaseModel):
    text: str

class EmbedBatchRequest(BaseModel):
    texts: list[str]

import torch

@app.on_event("startup")
def startup_event():
    global model
    print(f"Loading embedding model: {MODEL_NAME}...")
    
    # Use GPU if available, fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    
    print(f"Target device: {device} | precision: {'float16' if device == 'cuda' else 'float32'}")
    model = SentenceTransformer(
        MODEL_NAME, 
        trust_remote_code=True, 
        device=device, 
        model_kwargs=model_kwargs
    )
    
    # Disable cache for embedding generation
    model.config.use_cache = False
    model._first_module().auto_model.config.use_cache = False
    
    print("Model loaded successfully!")
    dummy_emb = model.encode("test")
    print(f"Embedding dimension of {MODEL_NAME} is: {len(dummy_emb)}")

@app.post("/embed")
def embed_text(payload: EmbedRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        vector = model.encode(payload.text).tolist()
        return {"embedding": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_batch")
def embed_batch(payload: EmbedBatchRequest):
    """Embed multiple texts in a single model.encode() call — much faster than one-by-one."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not payload.texts:
        return {"embeddings": []}
    try:
        vectors = model.encode(payload.texts, batch_size=32, show_progress_bar=False)
        return {"embeddings": [v.tolist() for v in vectors]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--port", type=int, default=8081, help="Port to run the server on")
    args = parser.parse_args()
    
    MODEL_NAME = args.model
    uvicorn.run(app, host="127.0.0.1", port=args.port)
