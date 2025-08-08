from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import requests
import os

# ---------------------------
# IMPORT YOUR WORKFLOW LOGIC
# ---------------------------
import uploader
import makechunk
import cleanchunk
import store
import llmanswer
from pinecone import Pinecone

# ---------------------------
# CONFIG
# ---------------------------
API_TOKEN = "41d7b56c1fcc879ed1cb365bed099c998b31dee69b8cb82cea10be63df6fb13a"

PINECONE_API_KEY = "pcsk_7MxGB2_FZ1SiXcuug2jDG6AFAMpk8ErxBhHjc2zrovEAgDpuHxscPSwHPUnjjweSPNYQzA"
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "pdf-chatbot-index"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

app = FastAPI()

# ---------------------------
# Pydantic Models
# ---------------------------
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# ---------------------------
# AUTH
# ---------------------------
def verify_bearer_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.split("Bearer ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid bearer token")

# ---------------------------
# Helper Functions
# ---------------------------
def is_pdf_processed(pdf_filename: str) -> bool:
    try:
        index = pc.Index(INDEX_NAME)
        results = index.query(
            vector=[0]*768,
            top_k=1,
            filter={"pdf_filename": {"$eq": pdf_filename}},
            include_metadata=True
        )
        return len(results.matches) > 0
    except Exception as e:
        print(f"Error checking if PDF is processed: {e}")
        return False

def run_full_workflow(pdf_path: str, query: str) -> str:
    pdf_filename = os.path.basename(pdf_path)
    
    if not is_pdf_processed(pdf_filename):
        raw_text = uploader.extract_text_from_pdf(pdf_path)
        if not raw_text:
            return "No text extracted from PDF."

        initial_chunks = makechunk.chunk_text(raw_text, chunk_size=500, overlap=50)
        if not initial_chunks:
            return "No chunks created from PDF."

        store.store_chunks_in_pinecone(initial_chunks, pdf_filename)

    query_embedding = llmanswer.get_embedding(query, task_type="RETRIEVAL_QUERY")
    if not query_embedding:
        return "Could not generate embedding for the query."

    retrieved_data = llmanswer.semantic_search_pinecone(query_embedding, top_k=3)
    if not retrieved_data:
        return "No relevant data found in Pinecone."

    final_answer = llmanswer.get_llm_answer(query, retrieved_data)
    return final_answer if final_answer else "No answer generated."

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
def hackrx_run(request: HackRxRequest, auth: None = Depends(verify_bearer_token)):
    # Download PDF
    try:
        pdf_response = requests.get(request.documents)
        pdf_response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_response.content)
        tmp_pdf_path = tmp_pdf.name

    answers = []
    try:
        for q in request.questions:
            answer = run_full_workflow(tmp_pdf_path, q)
            answers.append(answer.strip() if answer else "No answer found")
    finally:
        os.remove(tmp_pdf_path)

    return {"answers": answers}
