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

@app.on_event("startup")
def startup_event():
    global model
    print(f"Loading embedding model: {MODEL_NAME}...")
    # trust_remote_code=True is required for Qwen models on HuggingFace
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print("Model loaded successfully!")
    # Print the model's actual embedding dimension
    dummy_emb = model.encode("test")
    print(f"Embedding dimension of {MODEL_NAME} is: {len(dummy_emb)}")

@app.post("/embed")
def embed_text(payload: EmbedRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        # Generate embedding
        vector = model.encode(payload.text).tolist()
        return {"embedding": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--port", type=int, default=8081, help="Port to run the server on")
    args = parser.parse_args()
    
    MODEL_NAME = args.model
    uvicorn.run(app, host="127.0.0.1", port=args.port)
