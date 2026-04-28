#!/usr/bin/env python3
"""Convert chess book PDF pages to PGN with full commentary using Claude Vision."""

import argparse
from datetime import datetime
import json
import os
import sys
from pathlib import Path

from pgn_utils import validate_pgn, strip_result
from claude_api import get_client, pass1_extract_moves, pass2_attach_commentary
from pdf_io import pdf_pages_to_images, parse_page_range, extract_ocr_text
from detect import detect_games

CACHE_FILE = ".cache.json"


def process_game(client, pdf_path, pages, cache, cache_k, use_cache, game_hint=None):
    """Process a single game through pass 1 and pass 2. Returns (full_pgn, errors)."""
    if use_cache and cache_k in cache:
        print(f"  Using cached responses", file=sys.stderr)
        return cache[cache_k]["pass2"], cache[cache_k].get("errors", [])

    if client is None:
        client = get_client()

    images_b64 = pdf_pages_to_images(pdf_path, pages)

    # Pass 1
    print(f"  Pass 1: Extracting moves...", file=sys.stderr)
    pgn_moves = pass1_extract_moves(client, images_b64, game_hint)

    game, errors = validate_pgn(pgn_moves)
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

    cache[cache_k] = {"pass1": pgn_moves, "pass2": full_pgn, "errors": errors}
    save_cache(cache)

    return full_pgn, errors


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


def make_run_dir(stem):
    """Create a new timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("output") / f"{stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_run_dir(args, stem):
    """Find the run directory from --manifest flag or latest run."""
    if args.manifest:
        return Path(args.manifest).parent
    candidates = sorted(Path("output").glob(f"{stem}_*/"), reverse=True)
    if not candidates:
        print(f"Error: no run found for '{stem}'. Run --detect first.", file=sys.stderr)
        sys.exit(1)
    return candidates[0]


def cache_key(pdf_path, pages_str):
    """Generate a cache key from PDF name and page range."""
    return f"{Path(pdf_path).stem}_p{pages_str}"


def cmd_detect(args, run_dir):
    """Detect game boundaries and write manifest."""
    print("Detecting game boundaries...", file=sys.stderr)
    games = detect_games(args.pdf)
    print(f"Found {len(games)} games", file=sys.stderr)

    manifest = {
        "pdf": args.pdf,
        "games": [
            {
                "game_num": num,
                "start_page": start,
                "end_page": end,
                "pages_human": f"{start+1}-{end+1}",
            }
            for num, start, end in games
        ],
    }

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}", file=sys.stderr)

    for g in manifest["games"]:
        print(f"  Game {g['game_num']}: pages {g['pages_human']}", file=sys.stderr)


def load_manifest(run_dir):
    """Load manifest from run directory."""
    path = run_dir / "manifest.json"
    if not path.exists():
        print(f"Error: manifest not found at {path}. Run --detect first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def cmd_generate(args, run_dir):
    """Process games from manifest into individual PGN files."""
    manifest = load_manifest(run_dir)
    games = manifest["games"]

    # Filter by --games if specified
    if args.games:
        selected = set(int(n) for n in args.games.split(","))
        games = [g for g in games if g["game_num"] in selected]
        print(f"Processing {len(games)} selected game(s)", file=sys.stderr)

    games_dir = run_dir / "games"
    games_dir.mkdir(exist_ok=True)

    cache = load_cache()
    stem = Path(args.pdf).stem

    client = None
    if not args.cached:
        client = get_client()

    for g in games:
        game_num = g["game_num"]
        pages = list(range(g["start_page"], g["end_page"] + 1))
        print(f"\nGame {game_num} (pages {g['pages_human']})...", file=sys.stderr)

        cache_k = f"{stem}_game{game_num}"
        game_hint = f"Game {game_num}"
        full_pgn, errors = process_game(client, args.pdf, pages, cache, cache_k, args.cached, game_hint)
        full_pgn = strip_result(full_pgn)

        pgn_path = games_dir / f"game_{game_num:02d}.pgn"
        pgn_path.write_text(full_pgn)

        errors_path = games_dir / f"game_{game_num:02d}.errors.txt"
        if errors:
            errors_path.write_text("\n".join(errors) + "\n")
            print(f"  Errors: {errors_path}", file=sys.stderr)
        elif errors_path.exists():
            errors_path.unlink()

    print(f"\nGenerated PGNs in {games_dir}", file=sys.stderr)


def cmd_combine(args, run_dir):
    """Combine individual game PGNs into a single file."""
    manifest = load_manifest(run_dir)
    games_dir = run_dir / "games"

    all_pgns = []
    all_errors = []
    for g in manifest["games"]:
        game_num = g["game_num"]
        pgn_path = games_dir / f"game_{game_num:02d}.pgn"
        if not pgn_path.exists():
            print(f"  Warning: {pgn_path} not found, skipping", file=sys.stderr)
            continue
        all_pgns.append(pgn_path.read_text().rstrip())

        errors_path = games_dir / f"game_{game_num:02d}.errors.txt"
        if errors_path.exists():
            for line in errors_path.read_text().strip().splitlines():
                all_errors.append(f"Game {game_num} (pages {g['pages_human']}): {line}")

    combined = "\n\n".join(all_pgns)
    out_path = run_dir / "combined.pgn"
    out_path.write_text(combined)
    print(f"Combined {len(all_pgns)} games into {out_path}", file=sys.stderr)

    if all_errors:
        errors_path = run_dir / "combined.errors.txt"
        errors_path.write_text("\n".join(all_errors) + "\n")
        print(f"Errors: {errors_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Convert chess book PDF to PGN")
    parser.add_argument("pdf", help="Path to PDF file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", help="Page range, e.g. '1-5' or '3'")
    group.add_argument("--detect", action="store_true", help="Detect games, write manifest")
    group.add_argument("--generate", action="store_true", help="Process games from manifest")
    group.add_argument("--combine", action="store_true", help="Combine game PGNs into one file")
    group.add_argument("--book", action="store_true", help="Full pipeline (detect+generate+combine)")
    parser.add_argument("--games", help="Comma-separated game numbers (with --generate)")
    parser.add_argument("--manifest", help="Path to manifest file")
    parser.add_argument("--cached", action="store_true", help="Use cached API responses")
    args = parser.parse_args()

    needs_api = (args.pages or args.generate or args.book) and not args.cached
    if needs_api and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    stem = pdf_path.stem

    if args.detect or args.book or args.pages:
        run_dir = make_run_dir(stem)
    else:
        run_dir = find_run_dir(args, stem)

    if args.detect:
        cmd_detect(args, run_dir)
    elif args.generate:
        cmd_generate(args, run_dir)
    elif args.combine:
        cmd_combine(args, run_dir)
    elif args.book:
        cmd_detect(args, run_dir)
        cmd_generate(args, run_dir)
        cmd_combine(args, run_dir)
    else:
        # --pages mode
        cache = load_cache()
        client = None
        if not args.cached:
            client = get_client()

        pages = parse_page_range(args.pages)
        print(f"Processing {pdf_path}, pages {[p+1 for p in pages]}...", file=sys.stderr)
        cache_k = cache_key(args.pdf, args.pages)
        full_pgn, errors = process_game(client, args.pdf, pages, cache, cache_k, args.cached)
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
