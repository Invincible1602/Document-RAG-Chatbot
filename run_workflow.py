from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import requests
import os
from dotenv import load_dotenv
load_dotenv()

import uploader
import makechunk
import cleanchunk
import store
import llmanswer
from pinecone import Pinecone

API_TOKEN = os.getenv("API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "pdf-chatbot-index"

if not API_TOKEN:
    raise ValueError("Missing API_TOKEN environment variable")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY environment variable")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
app = FastAPI()

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

def verify_bearer_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.split("Bearer ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid bearer token")

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

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
def hackrx_run(request: HackRxRequest, auth: None = Depends(verify_bearer_token)):
    try:
        pdf_response = requests.get(request.documents)
        pdf_response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_response.content)
        tmp_pdf_path = tmp_pdf.name

    pdf_filename = os.path.basename(tmp_pdf_path)

    try:
        # Process PDF only once per request
        if not is_pdf_processed(pdf_filename):
            raw_text = uploader.extract_text_from_pdf(tmp_pdf_path)
            if not raw_text:
                return {"answers": ["No text extracted from PDF."] * len(request.questions)}

            initial_chunks = makechunk.chunk_text(raw_text, chunk_size=500, overlap=50)
            if not initial_chunks:
                return {"answers": ["No chunks created from PDF."] * len(request.questions)}

            # Store chunks with filename metadata so future requests can skip
            chunks_with_metadata = []
            for chunk in initial_chunks:
                chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": {"pdf_filename": pdf_filename}
                })
            store.store_chunks_in_pinecone(chunks_with_metadata, pdf_filename)

        answers = []
        for q in request.questions:
            query_embedding = llmanswer.get_embedding(q, task_type="RETRIEVAL_QUERY")
            if not query_embedding:
                answers.append("Could not generate embedding for the query.")
                continue

            retrieved_data = llmanswer.semantic_search_pinecone(query_embedding, top_k=3)
            if not retrieved_data:
                answers.append("No relevant data found in Pinecone.")
                continue

            final_answer = llmanswer.get_llm_answer(q, retrieved_data)
            answers.append(final_answer.strip() if final_answer else "No answer generated.")

        return {"answers": answers}

    finally:
        os.remove(tmp_pdf_path)
