#!/usr/bin/env python3
"""Convert chess book PDF pages to PGN with full commentary using Claude Vision."""

import argparse
import base64
from datetime import datetime
import json
import os
import re
import sys
from pathlib import Path

import anthropic
import chess.pgn
import io
import pymupdf
import pymupdf4llm

CACHE_FILE = ".cache.json"


MODEL = "claude-sonnet-4-6"


def strip_code_fences(text):
    """Remove markdown code fences from LLM output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def pdf_pages_to_images(pdf_path, pages):
    """Convert PDF pages to base64-encoded PNG images."""
    doc = pymupdf.open(pdf_path)
    images = []
    for i in pages:
        pix = doc[i].get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        images.append(base64.standard_b64encode(img_bytes).decode())
    return images


def build_image_content(images_b64):
    """Build Claude API image content blocks."""
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img},
        }
        for img in images_b64
    ]


def pass1_extract_moves(client, images_b64):
    """Pass 1: Extract clean PGN moves from page images (no commentary)."""
    content = build_image_content(images_b64) + [
        {
            "type": "text",
            "text": (
                "These are pages from a chess book. Extract ONLY the chess moves "
                "as PGN. Use standard algebraic notation (K, Q, R, B, N for pieces). "
                "Do NOT include commentary, annotations, or variations. Just the "
                "main line moves. Include game headers [Event], [White], [Black], "
                "[Result], [Opening] if visible. Output only valid PGN, nothing else."
            ),
        }
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": content}],
    )
    return strip_code_fences(response.content[0].text)


def validate_pgn(pgn_text):
    """Validate PGN using python-chess. Returns (game, errors)."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return None, ["Failed to parse PGN"]

    errors = []
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        if move not in board.legal_moves:
            errors.append(f"Illegal move at ply {i+1}: {move}")
        board.push(move)
    return game, errors


def pass2_attach_commentary(client, images_b64, ocr_text, pgn_moves):
    """Pass 2: Attach full book commentary to the PGN moves."""
    content = build_image_content(images_b64) + [
        {
            "type": "text",
            "text": f"""Here are the clean PGN moves extracted from these chess book pages:

{pgn_moves}

And here is the OCR-extracted text from the same pages (chess piece symbols may be garbled but the prose is readable):

{ocr_text}

Using the page images for visual reference and layout ordering, and the OCR text for exact commentary wording, produce a complete PGN with ALL of the author's commentary embedded as PGN comments in curly braces {{}}.

CRITICAL:
- Include EVERY paragraph of the author's explanations, attached to the move it follows.
- This is an instructional book — the prose commentary IS the content. Do not skip any of it.
- Do NOT include PGN variations in parentheses. Keep all analysis lines as text within the commentary.
- Preserve the author's exact wording from the OCR text (fixing only garbled piece symbols).
- Output only valid PGN, nothing else.""",
        },
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=16384,
        temperature=0,
        messages=[{"role": "user", "content": content}],
    )
    return strip_code_fences(response.content[0].text)


def parse_page_range(page_str):
    """Parse page range string like '1-5' into list of 0-indexed page numbers."""
    if "-" in page_str:
        start, end = page_str.split("-", 1)
        return list(range(int(start) - 1, int(end)))
    return [int(page_str) - 1]


def load_cache():
    """Load cache from disk."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def cache_key(pdf_path, pages_str):
    """Generate a cache key from PDF name and page range."""
    return f"{Path(pdf_path).stem}_p{pages_str}"


def main():
    parser = argparse.ArgumentParser(description="Convert chess book PDF to PGN")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--pages", required=True, help="Page range, e.g. '1-5' or '3'")
    parser.add_argument("--cached", action="store_true", help="Use cached API responses")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    pages = parse_page_range(args.pages)
    print(f"Processing {pdf_path}, pages {[p+1 for p in pages]}...", file=sys.stderr)

    cache = load_cache()
    key = cache_key(args.pdf, args.pages)

    if args.cached and key in cache:
        print("Using cached API responses", file=sys.stderr)
        pgn_moves = cache[key]["pass1"]
        full_pgn = cache[key]["pass2"]
    else:
        # Load API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)

        client = anthropic.Anthropic(api_key=api_key)

        # Convert pages to images
        images_b64 = pdf_pages_to_images(args.pdf, pages)

        # Pass 1: Extract moves
        print("Pass 1: Extracting moves...", file=sys.stderr)
        pgn_moves = pass1_extract_moves(client, images_b64)
        print(f"Moves:\n{pgn_moves}\n", file=sys.stderr)

        # Validate
        game, errors = validate_pgn(pgn_moves)
        if errors:
            print(f"Warning: PGN validation issues: {errors}", file=sys.stderr)
        else:
            print("Pass 1 PGN validated OK", file=sys.stderr)

        # Pass 2: Attach commentary
        print("Pass 2: Attaching commentary...", file=sys.stderr)
        ocr_text = pymupdf4llm.to_markdown(args.pdf, pages=pages)
        full_pgn = pass2_attach_commentary(client, images_b64, ocr_text, pgn_moves)

        # Save to cache
        cache[key] = {"pass1": pgn_moves, "pass2": full_pgn}
        save_cache(cache)
        print("Cached API responses", file=sys.stderr)

    # Strip result so Lichess doesn't spoil the ending
    full_pgn = re.sub(r'^\[Result "[^"]*"\]\n', '', full_pgn, flags=re.MULTILINE)
    # Remove result marker outside of comments: split on {}, only strip from non-comment parts
    parts = re.split(r'(\{[^}]*\})', full_pgn)
    parts = [
        re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*', ' ', p) if not p.startswith('{') else p
        for p in parts
    ]
    full_pgn = ''.join(parts).rstrip()

    # Save output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    stem = pdf_path.stem
    page_label = args.pages.replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{stem}_p{page_label}_{timestamp}.pgn"
    out_path.write_text(full_pgn)

    print(f"\nSaved to {out_path}", file=sys.stderr)
    print(full_pgn)


if __name__ == "__main__":
    main()
