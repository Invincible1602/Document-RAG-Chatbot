# cleanchunk.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import os

# Configure your Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with your actual API key or set as environment variable
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Please set it as an environment variable or in the script.")
    print("You can get an API key from https://ai.google.dev/gemini-api/docs/get-started/python")

def call_embedding_with_retry(text: str, model_name: str = "models/embedding-001", max_retries: int = 5, initial_delay: int = 1):
    """
    Gets embeddings for text with exponential backoff for retries.

    Args:
        text (str): The text to embed.
        model_name (str): The name of the embedding model to use.
        max_retries (int): Maximum number of retries.
        initial_delay (int): Initial delay in seconds before the first retry.

    Returns:
        List[float]: The embedding vector for the text.
    """
    for attempt in range(max_retries):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=GEMINI_API_KEY
            )
            return embeddings.embed_query(text)
        except Exception as e:
            delay = initial_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print(f"Failed to get embeddings after {max_retries} attempts.")
    return None

def process_chunk_with_embeddings(chunk: str) -> list:
    """
    Processes a text chunk using embeddings to represent its semantic content.

    Args:
        chunk (str): The raw text chunk to be processed.

    Returns:
        list: The embedding vector representing the chunk's semantic content.
    """
    if not chunk.strip():
        return None

    # print(f"Processing chunk (first 100 chars): '{chunk[:100]}...'")
    embedding = call_embedding_with_retry(chunk)
    return embedding

if __name__ == "__main__":
    # Example Usage:
    sample_raw_chunks = [
        "This is a very verbose chunk of text that contains a lot of filler words and repetitions. The main point here is that AI is becoming more prevalent in daily life, but there's also some discussion about the weather and my cat's eating habits, which are not relevant to the core topic of AI. We should focus on the AI aspects.",
        "Another chunk about artificial intelligence. Specifically, AI is used in self-driving cars like Waymo, and also in creative applications such as ChatGPT for generating text and AI art tools. This demonstrates the wide range of AI capabilities.",
        "A short, clean chunk already."
    ]

    embeddings = []
    for i, raw_chunk in enumerate(sample_raw_chunks):
        # print(f"\nProcessing raw chunk {i+1}...")
        embedding = process_chunk_with_embeddings(raw_chunk)
        if embedding is not None:
            embeddings.append(embedding)
            # print(f"Generated embedding for chunk {i+1} (length: {len(embedding)})")
        else:
            print(f"Embedding generation failed for chunk {i+1}.")

    print("\n--- Embedding Information ---")
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0]) if embeddings else 0} dimensions")