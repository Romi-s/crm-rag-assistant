from typing import List, Tuple

import fitz


def extract_text_from_pdf(
    pdf_bytes: bytes, max_pages: int = 100
) -> List[Tuple[int, str]]:
    """Return list of (page_number, text) tuples with non-empty pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(min(len(doc), max_pages)):
        text = doc[page_num].get_text()
        if text.strip():
            pages.append((page_num + 1, text))
    doc.close()
    return pages
