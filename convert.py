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


def strip_result(pgn_text):
    """Remove result header and marker so Lichess doesn't spoil the ending."""
    pgn_text = re.sub(r'^\[Result "[^"]*"\]\n', '', pgn_text, flags=re.MULTILINE)
    parts = re.split(r'(\{[^}]*\})', pgn_text)
    parts = [
        re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*', ' ', p) if not p.startswith('{') else p
        for p in parts
    ]
    return ''.join(parts).rstrip()


def detect_games(pdf_path):
    """Detect game boundaries in a chess book PDF.
    Returns list of (game_number, start_page, end_page) tuples (0-indexed pages)."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    game_starts = {}  # game_number -> first page (0-indexed)

    for page_num in range(total_pages):
        text = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])
        # Match "Game N" anywhere on a line, handling OCR spaces in numbers
        matches = re.findall(r'Game\s+(\d[\d ]*)', text)
        for m in matches:
            num = int(m.replace(' ', ''))
            if num not in game_starts:
                game_starts[num] = page_num

    # Sort by game number and build ranges
    sorted_games = sorted(game_starts.items())
    games = []
    for i, (num, start) in enumerate(sorted_games):
        if i + 1 < len(sorted_games):
            end = sorted_games[i + 1][1] - 1
        else:
            end = total_pages - 1
        games.append((num, start, end))

    return games


def get_client():
    """Create and return an Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def process_game(client, pdf_path, pages, cache, cache_k, use_cache):
    """Process a single game through pass 1 and pass 2. Returns full PGN text."""
    if use_cache and cache_k in cache:
        print(f"  Using cached responses", file=sys.stderr)
        return cache[cache_k]["pass2"]

    if client is None:
        client = get_client()

    images_b64 = pdf_pages_to_images(pdf_path, pages)

    # Pass 1
    print(f"  Pass 1: Extracting moves...", file=sys.stderr)
    pgn_moves = pass1_extract_moves(client, images_b64)

    game, errors = validate_pgn(pgn_moves)
    if errors:
        print(f"  Warning: PGN validation issues: {errors}", file=sys.stderr)
    else:
        print(f"  Pass 1 validated OK", file=sys.stderr)

    # Pass 2
    print(f"  Pass 2: Attaching commentary...", file=sys.stderr)
    ocr_text = pymupdf4llm.to_markdown(pdf_path, pages=pages)
    full_pgn = pass2_attach_commentary(client, images_b64, ocr_text, pgn_moves)

    cache[cache_k] = {"pass1": pgn_moves, "pass2": full_pgn}
    save_cache(cache)

    return full_pgn


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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", help="Page range, e.g. '1-5' or '3'")
    group.add_argument("--book", action="store_true", help="Process entire book")
    parser.add_argument("--cached", action="store_true", help="Use cached API responses")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    cache = load_cache()
    stem = pdf_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Set up API client (not needed if fully cached)
    client = None
    if not args.cached:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)

    if args.book:
        print("Detecting game boundaries...", file=sys.stderr)
        games = detect_games(args.pdf)
        print(f"Found {len(games)} games", file=sys.stderr)

        all_pgns = []
        for game_num, start, end in games:
            pages = list(range(start, end + 1))
            print(f"\nGame {game_num}/{games[-1][0]} (pages {start+1}-{end+1})...", file=sys.stderr)
            cache_k = f"{stem}_game{game_num}"
            full_pgn = process_game(client, args.pdf, pages, cache, cache_k, args.cached)
            full_pgn = strip_result(full_pgn)
            all_pgns.append(full_pgn)

        combined = "\n\n".join(all_pgns)
        out_path = output_dir / f"{stem}_full_{timestamp}.pgn"
        out_path.write_text(combined)
        print(f"\nSaved {len(games)} games to {out_path}", file=sys.stderr)

    else:
        pages = parse_page_range(args.pages)
        print(f"Processing {pdf_path}, pages {[p+1 for p in pages]}...", file=sys.stderr)
        cache_k = cache_key(args.pdf, args.pages)
        full_pgn = process_game(client, args.pdf, pages, cache, cache_k, args.cached)
        full_pgn = strip_result(full_pgn)

        page_label = args.pages.replace("-", "_")
        out_path = output_dir / f"{stem}_p{page_label}_{timestamp}.pgn"
        out_path.write_text(full_pgn)
        print(f"\nSaved to {out_path}", file=sys.stderr)
        print(full_pgn)


if __name__ == "__main__":
    main()
