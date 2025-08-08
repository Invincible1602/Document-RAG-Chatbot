# store.py
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import time
import os
from typing import List, Dict, Any

# Configure your Gemini API key for embeddings
GEMINI_API_KEY = "AIzaSyDVajhOu5FSFiH0cXICpeRdi0l9-6w7vtk" # Replace with your actual API key or set as environment variable
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Please set it as an environment variable or in the script.")
    print("You can get an API key from https://ai.google.dev/gemini-api/docs/get-started/python")

genai.configure(api_key=GEMINI_API_KEY)

# Configure your Pinecone API key and environment
PINECONE_API_KEY = "pcsk_7MxGB2_FZ1SiXcuug2jDG6AFAMpk8ErxBhHjc2zrovEAgDpuHxscPSwHPUnjjweSPNYQzA" # Replace with your actual API key
PINECONE_ENVIRONMENT = "us-east-1" # e.g., "us-east-1" or "gcp-starter"
INDEX_NAME = "pdf-chatbot-index"
EMBEDDING_MODEL = "models/embedding-001"  # Updated model name
EMBEDDING_DIMENSION = 768 # Default for gemini-embedding-001, can be 3072, 1536, or 768

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    print("WARNING: Pinecone API key or environment not set. Please configure them.")


pc = None
try:
    if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("Pinecone client initialized.")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    pc = None


def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    if not genai or not text.strip():
        return []
    try:
        resp = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type=task_type)
        emb = None
        if isinstance(resp, dict):
            emb = resp.get('embedding') or (resp.get('data', [{}])[0].get('embedding'))
        elif hasattr(resp, 'embedding'):
            emb = resp.embedding
        if emb and len(emb) == EMBEDDING_DIMENSION:
            return list(emb)
    except Exception as e:
        print(f"Error generating embedding: {e}")
    return []


def store_chunks_in_pinecone(chunks: List[str], pdf_filename: str = "unknown_pdf") -> None:
    if not pc:
        print("Pinecone client not initialized. Cannot store chunks.")
        return

    # Ensure index exists (create if missing)
    try:
        existing = set(pc.list_indexes()) if hasattr(pc, 'list_indexes') else set()
        if INDEX_NAME not in existing:
            try:
                kwargs = {"name": INDEX_NAME, "dimension": EMBEDDING_DIMENSION, "metric": "cosine"}
                if ServerlessSpec:
                    kwargs["spec"] = ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
                pc.create_index(**kwargs)
                print(f"Created index {INDEX_NAME}.")
            except Exception as e:
                if getattr(e, 'status', None) == 409 or 'ALREADY_EXISTS' in str(e):
                    print(f"Index {INDEX_NAME} already exists, continuing.")
                else:
                    raise
        else:
            print(f"Index {INDEX_NAME} already exists.")
    except Exception as e:
        print(f"Error ensuring index exists: {e}")
        return

    index = pc.Index(INDEX_NAME)
    vectors = []
    for i, chunk_text in enumerate(chunks):  # Change variable name to be clearer
        text = str(chunk_text).strip()
        if not text:
            continue
        
        # Generate the embedding from the actual text chunk
        emb = get_embedding(text) 
        if not emb:
            print(f"Skipping chunk {i}: no embedding.")
            continue
        
        vectors.append({
            'id': f"{pdf_filename}_chunk_{i}",
            'values': emb,
            # The metadata 'text' field now correctly stores the original text
            'metadata': {'text': text, 'pdf_filename': pdf_filename, 'chunk_index': i}
        })
    if not vectors:
        print("No vectors to upsert.")
        return
    # Upsert in batches
    for start in range(0, len(vectors), 100):
        batch = vectors[start:start+100]
        try:
            index.upsert(vectors=batch)
            print(f"Upserted batch {start//100 + 1}")
        except Exception as e:
            print(f"Error upserting batch {start//100}: {e}")
    print(f"Stored {len(vectors)} vectors in {INDEX_NAME}.")

if __name__ == "__main__":
    print("store.py loaded. Call store_chunks_in_pinecone(...) after configuring env variables.")











