import math
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDVajhOu5FSFiH0cXICpeRdi0l9-6w7vtk"
genai.configure(api_key=GEMINI_API_KEY)

def split_sentences_with_gemini(text: str) -> list[str]:
    """Uses Google Gemini to split the text into sentences."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Split the following text into separate sentences, each on a new line:\n\n{text}"
    
    try:
        response = model.generate_content(prompt)
        sentences = [line.strip() for line in response.text.split("\n") if line.strip()]
        return sentences
    except Exception as e:
        print(f"Error using Gemini for sentence splitting: {e}")
        return [text]  # fallback: single sentence

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits raw text into coherent chunks using Gemini sentence splitting."""
    if not text:
        return []

    sentences = split_sentences_with_gemini(text)
    chunks = []
    current_chunk = []
    current_chunk_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_chunk_len + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_text = " ".join(current_chunk)[-overlap:] if len(" ".join(current_chunk)) > overlap else ""
            current_chunk = [overlap_text.strip()] if overlap_text.strip() else []
            current_chunk_len = len(overlap_text.strip())

        current_chunk.append(sentence)
        current_chunk_len += sentence_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 1.5:
            num_sub_chunks = math.ceil(len(chunk) / (chunk_size - overlap))
            sub_chunk_len = math.ceil(len(chunk) / num_sub_chunks)
            for i in range(0, len(chunk), sub_chunk_len):
                final_chunks.append(chunk[i:i + sub_chunk_len])
        else:
            final_chunks.append(chunk)

    print(f"Original text chunked into {len(final_chunks)} segments.")
    return final_chunks
