from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text page by page.

    Returns:
    [
        {"page": 1, "text": "..."},
        {"page": 2, "text": "..."}
    ]
    """
    path = Path(pdf_path)
    reader = PdfReader(str(path))

    pages = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split()) 
        pages.append({
            "page": index,
            "text": text,
        })

    return pages