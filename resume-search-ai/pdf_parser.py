"""
pdf_parser.py — Extract text from PDF files
=============================================
This module handles reading PDF files and extracting their text content.
We use PyPDF2 (a pure-Python PDF library) so there's no system dependency.

How it works:
    1. Takes an uploaded file (from Streamlit's file_uploader)
    2. Reads it as a PDF using PyPDF2
    3. Extracts text from every page
    4. Cleans up the text (remove extra whitespace, etc.)
    5. Returns the full text as a single string
"""

import io
import re
from PyPDF2 import PdfReader


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from a PDF file.

    Args:
        uploaded_file: A file-like object (from Streamlit's file_uploader
                       or a regular file opened in binary mode).

    Returns:
        A single string containing all the text from the PDF.
        Returns empty string if extraction fails.

    Example:
        >>> text = extract_text_from_pdf(uploaded_file)
        >>> print(text[:200])  # Print first 200 characters
    """
    try:
        # Read the PDF file
        # PdfReader can accept a file-like object or a file path
        reader = PdfReader(uploaded_file)

        # Collect text from all pages
        all_text_parts = []

        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()

            if page_text:
                all_text_parts.append(page_text)

        # Join all pages into one string
        full_text = "\n".join(all_text_parts)

        # Clean up the text
        full_text = clean_text(full_text)

        return full_text

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing extra whitespace and artifacts.

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Cleaned text string.
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces with a single space
    text = re.sub(r" {2,}", " ", text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove any null bytes or other control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


# ------------------------------------------------------------------
# Quick test: run this file directly to test PDF extraction
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    with open(pdf_path, "rb") as f:
        text = extract_text_from_pdf(f)

    print(f"Extracted {len(text)} characters from {pdf_path}")
    print("=" * 60)
    print(text[:1000])  # Print first 1000 characters
