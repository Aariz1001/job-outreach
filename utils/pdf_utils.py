"""Extract plain text from the CV PDF."""
import fitz  # PyMuPDF
from pathlib import Path


def extract_cv_text(pdf_path: str | Path) -> str:
    doc = fitz.open(str(pdf_path))
    return "\n".join(page.get_text() for page in doc).strip()
