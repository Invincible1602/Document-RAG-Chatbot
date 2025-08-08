# uploader.py
import PyPDF2
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a given PDF file.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: The extracted raw text content from the PDF.
             Returns an empty string if the file is not found or cannot be processed.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return ""

    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        print(f"Successfully extracted text from {pdf_path}")
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""
    return text

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy PDF file for testing
    dummy_pdf_content = """
    This is the first paragraph of a dummy PDF.
    It contains some sample text to demonstrate PDF extraction.

    This is the second paragraph.
    It talks about the importance of clean data.
    """
    dummy_pdf_path = "dummy_document.pdf"

    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
        textobject = c.beginText()
        textobject.setTextOrigin(50, 750)
        for line in dummy_pdf_content.split('\n'):
            textobject.textLine(line)
        c.drawText(textobject)
        c.save()
        print(f"Dummy PDF '{dummy_pdf_path}' created for testing.")
    except ImportError:
        print("reportlab not installed. Cannot create dummy PDF.")
        print("Please install with: pip install reportlab PyPDF2")
        print("Using a pre-existing PDF or manually creating one is required for testing.")
        exit()

    extracted_text = extract_text_from_pdf(dummy_pdf_path)
    if extracted_text:
        print("\n--- Extracted Text ---")
        print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
    else:
        print("No text extracted.")

    # Clean up dummy PDF
    if os.path.exists(dummy_pdf_path):
        os.remove(dummy_pdf_path)
        print(f"Dummy PDF '{dummy_pdf_path}' removed.")
