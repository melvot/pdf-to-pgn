#!/usr/bin/env python3
"""Convert chess book PDF pages to PGN with full commentary using Claude Vision."""

import argparse
import os
import sys
from pathlib import Path

from pgn_utils import validate_pgn, strip_result
from claude_api import get_client, pass1_extract_moves, pass2_attach_commentary
from pdf_io import pdf_pages_to_images, parse_page_range, extract_ocr_text
from storage import make_run_dir


def process_game(client, pdf_path, pages, game_hint=None):
    """Process a single game through pass 1 and pass 2. Returns (full_pgn, errors)."""

    if client is None:
        client = get_client()

    images_b64 = pdf_pages_to_images(pdf_path, pages)

    # Pass 1
    print(f"  Pass 1: Extracting moves...", file=sys.stderr)
    pgn_moves = pass1_extract_moves(client, images_b64, game_hint)

    _, errors = validate_pgn(pgn_moves)
    if errors:
        print(f"  Warning: PGN validation issues: {errors}", file=sys.stderr)
    else:
        print(f"  Pass 1 validated OK", file=sys.stderr)

    # Pass 2
    print(f"  Pass 2: Attaching commentary...", file=sys.stderr)
    ocr_text = extract_ocr_text(pdf_path, pages)
    full_pgn = pass2_attach_commentary(client, images_b64, ocr_text, pgn_moves, game_hint)

    _, pass2_errors = validate_pgn(full_pgn)
    if pass2_errors:
        print(f"  Warning: Pass 2 PGN invalid: {pass2_errors}", file=sys.stderr)
        errors = errors + pass2_errors
    else:
        print(f"  Pass 2 validated OK", file=sys.stderr)

    return full_pgn, errors


def main():
    parser = argparse.ArgumentParser(description="Convert chess book PDF to PGN")
    parser.add_argument("pdf", help="Path to PDF file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", help="Page range, e.g. '1-5' or '3'")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    stem = pdf_path.stem

    run_dir = make_run_dir(stem)

    client = get_client()

    pages = parse_page_range(args.pages)
    print(f"Processing {pdf_path}, pages {[p+1 for p in pages]}...", file=sys.stderr)
    full_pgn, errors = process_game(client, args.pdf, pages)
    full_pgn = strip_result(full_pgn)
    if errors:
        print(f"Validation errors: {errors}", file=sys.stderr)

    page_label = args.pages.replace("-", "_")
    out_path = run_dir / f"p{page_label}.pgn"
    out_path.write_text(full_pgn)
    print(f"\nSaved to {out_path}", file=sys.stderr)
    print(full_pgn)


if __name__ == "__main__":
    main()
