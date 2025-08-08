# makechunk.py
import nltk
from nltk.tokenize import sent_tokenize
import math

# Download necessary NLTK data (only needs to be run once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer downloaded.")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits raw text into meaningful chunks with a specified size and overlap.
    Prioritizes sentence boundaries for more coherent chunks.

    Args:
        text (str): The raw text content to be chunked.
        chunk_size (int): The approximate target size of each chunk (in characters).
        overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        # If adding the current sentence exceeds the chunk size significantly,
        # or if it's the first sentence and already larger than chunk_size,
        # finalize the current chunk and start a new one.
        if current_chunk_len + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            overlap_text = " ".join(current_chunk)[-overlap:] if len(" ".join(current_chunk)) > overlap else ""
            current_chunk = [overlap_text.strip()] if overlap_text.strip() else []
            current_chunk_len = len(overlap_text.strip())

        current_chunk.append(sentence)
        current_chunk_len += sentence_len + 1 # +1 for space

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Fallback for very long sentences that exceed chunk_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.5: # If a chunk is still too large (e.g., a single very long sentence)
            # Simple character-based split for extremely long chunks
            num_sub_chunks = math.ceil(len(chunk) / (chunk_size - overlap))
            sub_chunk_len = math.ceil(len(chunk) / num_sub_chunks)
            for i in range(0, len(chunk), sub_chunk_len):
                sub_chunk = chunk[i:i + sub_chunk_len]
                final_chunks.append(sub_chunk)
        else:
            final_chunks.append(chunk)

    print(f"Original text chunked into {len(final_chunks)} segments.")
    return final_chunks

if __name__ == "__main__":
    print("\n--- Generated Chunks ---")


















