"""Detect game boundaries."""

import re

import pymupdf

MAX_GAME_PAGES = 15
RESULT_RE = re.compile(r'\b(1-0|0-1|1/2-1/2|½-½)\b')
# Use \S instead of [A-Za-z] to handle OCR-garbled piece symbols (^, §, £, etc.)
MOVE_NUM_RE = re.compile(r'\b(\d+)\.\s*\S')
CHAPTER_RE = re.compile(r'^(Part [IVX]+|Chapter \d+|Index of)', re.MULTILINE)


def max_move_num(text):
    """Return the highest move number found on a page, or 0 if none."""
    nums = [int(m) for m in MOVE_NUM_RE.findall(text) if int(m) < 150]
    return max(nums) if nums else 0


def find_game_end(doc, start_page, next_game_start, max_pages):
    """Find the end page of a game using multiple signals:
    1. Chapter/section heading — strong structural signal
    2. Result marker (1-0, 0-1, etc.)
    3. Move number reset — if we've seen move 10+ then see only moves 1-2, new game started
    4. next_game_start — don't go past where the next game begins
    5. max_pages cap — hard fallback
    """
    limit = min(start_page + max_pages, next_game_start, len(doc) - 1)
    max_move_seen = 0

    for page_num in range(start_page, limit + 1):
        text = doc[page_num].get_text()

        if CHAPTER_RE.search(text) and page_num > start_page:
            return page_num - 1

        if RESULT_RE.search(text):
            return min(page_num + 1, len(doc) - 1)

        page_max = max_move_num(text)
        if max_move_seen >= 10 and 0 < page_max <= 2:
            return page_num
        if page_max > max_move_seen:
            max_move_seen = page_max

    return limit


def detect_games(pdf_path):
    """Detect game boundaries in a chess book PDF.
    Returns list of (game_number, start_page, end_page) tuples (0-indexed pages)."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    game_starts = {}  # game_number -> first page (0-indexed)

    for page_num in range(total_pages):
        text = doc[page_num].get_text()
        # Match "Game N" only at the start of a line (real headings, not prose references)
        matches = re.findall(r'(?m)^Game\s+(\d[\d ]*)\s*$', text)
        for m in matches:
            num = int(m.replace(' ', ''))
            if num not in game_starts:
                game_starts[num] = page_num

    # Sort by game number and build ranges
    sorted_games = sorted(game_starts.items())
    games = []
    for i, (num, start) in enumerate(sorted_games):
        next_start = sorted_games[i + 1][1] if i + 1 < len(sorted_games) else total_pages - 1
        end = find_game_end(doc, start, next_start, MAX_GAME_PAGES)
        games.append((num, start, end))

    return games
