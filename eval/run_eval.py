"""Evaluate move extraction functions."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_api import pass1_extract_moves, get_client
from pgn_utils import validate_pgn
from pdf_io import pdf_pages_to_images, parse_page_range

EVAL_PATH = Path(__file__).parent / "eval_games.json"
PROJECT_DIR = Path(__file__).parent.parent

def load_games():
    """Load list of games selected for evaluation."""
    path = EVAL_PATH
    if not path.exists():
        print(f"Error: list of eval games not found at {path}.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

def main():
    client = get_client()
    print("Starting evaluation.")

    for game in load_games():
        book = game["book"]
        game_num = f"Game {game['game_num']}"
        pages = parse_page_range(game["pages"])
        print(f"Book: {book}, Game num: {game_num}, Pages: {pages}")

        path = PROJECT_DIR / book
        images_b64 = pdf_pages_to_images(path, pages)

        print(f"  Extracting moves...")
        pgn_moves = pass1_extract_moves(client, images_b64, game_num)
        print(pgn_moves)

        _, errors = validate_pgn(pgn_moves)
        if errors:
            print(f"  Warning: PGN validation issues: {errors}")
        else:
            print(f"  Pass 1 validated OK.")

if __name__ == "__main__":
    main()
