"""
Corrected llmanswer.py
- Uses environment variables for API keys (falls back to hard-coded for convenience but warns).
- Robust embedding parsing from the Gemini response.
- Heuristic to skip "embedding-like" strings when building LLM context.
- Truncates context to a safe max character length before sending to LLM.
- Better error handling and logging.
"""

import os
import re
import time
from typing import List, Dict, Any

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

# Load API keys from environment first (safer).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "pdf-chatbot-index"
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIMENSION = 768

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Set it via environment variable GEMINI_API_KEY.")
if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

pc = None
if Pinecone and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("Pinecone client initialized.")
    except Exception as e:
        print(f"Could not initialize Pinecone client: {e}")
        pc = None
else:
    print("Pinecone client not configured. Set PINECONE_API_KEY and ensure pinecone SDK is installed.")

# === Helper to detect embedding-like strings ===
def _is_embedding_like_string(s: str, threshold: float = 0.6) -> bool:
    if not isinstance(s, str): return False
    s = s.strip()
    if not s: return False
    if re.search(r"[A-Za-z]", s): return False
    tokens = re.split(r"[\s,]+", s)
    if not tokens: return False
    float_like = sum(1 for t in tokens if _is_float(t))
    return (float_like / len(tokens)) >= threshold

def _is_float(t: str) -> bool:
    try:
        float(t)
        return True
    except:
        return False

# === Embedding ===
def get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    if not genai:
        print("Gemini client (google.generativeai) not available.")
        return []
    if not text or not isinstance(text, str):
        return []
    try:
        resp = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type=task_type)
        # handle dict or object responses
        emb = None
        if isinstance(resp, dict):
            emb = resp.get('embedding') or (resp.get('data', [{}])[0].get('embedding'))
        elif hasattr(resp, 'embedding'):
            emb = resp.embedding
        if emb and isinstance(emb, (list, tuple)):
            return list(emb)
        print("Warning: embedding not found in Gemini response.")
    except Exception as e:
        print(f"Error generating embedding: {e}")
    return []

# === Semantic search ===
def semantic_search_pinecone(query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    if not pc:
        print("Pinecone client not initialized.")
        return []
    if not query_embedding:
        print("Empty query embedding.")
        return []
    try:
        index = pc.Index(INDEX_NAME)
        qr = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = qr.get('matches', []) if isinstance(qr, dict) else getattr(qr, 'matches', [])
        chunks = []
        for m in matches:
            md = m.get('metadata', {})
            text = md.get('text', '')
            if not text or _is_embedding_like_string(text):
                continue
            chunks.append({
                'id': m.get('id'),
                'score': m.get('score'),
                'text': text,
                'pdf_filename': md.get('pdf_filename', 'N/A'),
                'chunk_index': md.get('chunk_index', -1)
            })

            # print(chunks[0].text)

        return chunks
    except Exception as e:
        print(f"Error during Pinecone search: {e}")
        return []

# === Build truncated context ===
def _build_truncated_context(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    sorted_chunks = sorted(chunks, key=lambda c: c.get('score', 0), reverse=True)
    text_accum = []
    total = 0
    for c in sorted_chunks:
        t = c.get('text', '')
        if not t:
            continue
        # Check if adding the next chunk will exceed the max length
        if total + len(t) + 2 > max_chars:
            remaining = max_chars - total
            if remaining > 20:
                text_accum.append(t[:remaining] + "...")
            break
        text_accum.append(t)
        total += len(t) + 2

        # print("\n\n".join(text_accum))

    return "\n\n".join(text_accum)

# === LLM Answer ===
def get_llm_answer(query: str, retrieved_context: List[Dict[str, Any]], model_name: str = "gemini-2.5-flash-preview-05-20") -> str:
    context = _build_truncated_context(retrieved_context)
    
    # print("Context length:", len(context), "characters")
    # print("Context content (first 500 chars):", context[:500])
    
    # print("Context content:", context)

    if not context.strip():
        return "I couldn't find enough relevant information to answer your question."
    
    prompt = (
        f"You are a helpful assistant. Answer only based on the context below."
        f"\nQuestion: {query}\nContext:\n---\n{context}\n---\nAnswer:"
    )
    
    if not genai:
        return "LLM client unavailable."
    
    # OLDER CODE PATTERN: Instantiate a model object first.
    # The 'generate_content' method is then called on this object.
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error instantiating model {model_name}: {e}")
        return "I apologize, I'm unable to initialize the model."

    for attempt in range(1):
        try:
            # We now call generate_content on the model object, not the genai module.
            resp = model.generate_content(
                contents=[{'parts': [{'text': prompt}]}]
            )
            
            if resp.candidates:
                # The response structure is the same.
                text = resp.candidates[0].content.parts[0].text
                return text
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
            
    return "I apologize, I'm unable to generate a response now."

if __name__ == "__main__":
    print("llmanswer module loaded.")