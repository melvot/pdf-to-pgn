"""Convert PDF pages to PNG and PDF utilities."""

import base64

import pymupdf
import pymupdf4llm

def pdf_pages_to_images(pdf_path, pages):
    """Convert PDF pages to base64-encoded PNG images."""
    doc = pymupdf.open(pdf_path)
    images = []
    for i in pages:
        pix = doc[i].get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        images.append(base64.standard_b64encode(img_bytes).decode())
    return images


def parse_page_range(page_str):
    """Parse page range string like '1-5' into list of 0-indexed page numbers."""
    if "-" in page_str:
        start, end = page_str.split("-", 1)
        return list(range(int(start) - 1, int(end)))
    return [int(page_str) - 1]


def extract_ocr_text(pdf_path, pages):
    """Extract markdown-formatted text from PDF pages for LLM input."""
    return pymupdf4llm.to_markdown(pdf_path, pages=pages)
