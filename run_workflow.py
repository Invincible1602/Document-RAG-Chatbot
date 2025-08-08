

import uploader
import makechunk
import cleanchunk
import store
import llmanswer
import nltk
import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

# Configure your Pinecone API key and environment
PINECONE_API_KEY = "pcsk_7MxGB2_FZ1SiXcuug2jDG6AFAMpk8ErxBhHjc2zrovEAgDpuHxscPSwHPUnjjweSPNYQzA"
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "pdf-chatbot-index"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

def is_pdf_processed(pdf_filename: str) -> bool:
    """
    Checks if a PDF has already been processed and stored in Pinecone.
    
    Args:
        pdf_filename (str): The name of the PDF file to check
        
    Returns:
        bool: True if the PDF has been processed, False otherwise
    """
    try:
        index = pc.Index(INDEX_NAME)
        
        # Query for any vectors with this PDF filename in metadata
        # We just need to check if any exist, so limit=1 is sufficient
        results = index.query(
            vector=[0]*768,  # Dummy vector (dimension should match your index)
            top_k=1,
            filter={"pdf_filename": {"$eq": pdf_filename}},
            include_metadata=True
        )
        
        return len(results.matches) > 0
    except Exception as e:
        print(f"Error checking if PDF is processed: {e}")
        return False

def run_full_workflow(pdf_path: str, query: str):
    """
    Orchestrates the entire PDF processing and chatbot query workflow.
    Checks if PDF has been processed before to avoid reprocessing.
    """
    print("--- Starting PDF Processing Workflow ---")
    pdf_filename = os.path.basename(pdf_path)
    
    # Check if PDF has already been processed
    if is_pdf_processed(pdf_filename):
        print(f"\nPDF '{pdf_filename}' already processed. Skipping to query phase.")
    else:
        # Phase 1: Uploader.py - Extract text from PDF
        print(f"\n1. Extracting text from PDF: {pdf_path}")
        raw_text = uploader.extract_text_from_pdf(pdf_path)
        if not raw_text:
            print("Workflow aborted: No text extracted.")
            return

        # Phase 2: Makechunk.py - Create initial chunks
        print("\n2. Chunking raw text...")
        initial_chunks = makechunk.chunk_text(raw_text, chunk_size=500, overlap=50)
        if not initial_chunks:
            print("Workflow aborted: No chunks created.")
            return

        # Phase 3: No longer need to "refine" chunks.
        # The embedding generation happens inside store.py now.
        
        # Phase 4: Store.py - Generate embeddings and store in Pinecone
        print(f"\n4. Storing {len(initial_chunks)} text chunks in Pinecone index...")
        store.store_chunks_in_pinecone(initial_chunks, pdf_filename)

    print("\n--- Starting Chatbot Query Workflow ---")

    # Phase 5: LLMAnswer.py - Answer the query
    print(f"\n5. Processing user query: '{query}'")

    # Get embedding for the query
    query_embedding = llmanswer.get_embedding(query, task_type="RETRIEVAL_QUERY")
    if not query_embedding:
        print("Query failed: Could not generate embedding.")
        return

    # Perform semantic search
    retrieved_data = llmanswer.semantic_search_pinecone(query_embedding, top_k=3)
    if not retrieved_data:
        print("Query failed: No relevant data found in Pinecone.")
        return

    # Get the final answer from the LLM
    final_answer = llmanswer.get_llm_answer(query, retrieved_data)

    print("\n--- Final Answer ---")
    print(final_answer)
    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    pdf_path = "Communication.pdf"  # Replace with your PDF file path
    user_query = "What is meant by Stereotyping?"  # Replace with your query

    if not os.path.exists(pdf_path):
        print(f"Error: '{pdf_path}' does not exist. Please place the file in the script's directory.")
    else:
        run_full_workflow(pdf_path, user_query)