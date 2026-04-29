"""PGN validation and post-processing utilities."""

import io
import re
import chess.pgn

def validate_pgn(pgn_text):
    """Validate PGN using python-chess. Returns (game, errors)."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return None, ["Failed to parse PGN"]

    return game, [str(e) for e in game.errors]


def strip_result(pgn_text):
    """Remove result header and marker so Lichess doesn't spoil the ending."""
    pgn_text = re.sub(r'^\[Result "[^"]*"\]\n', '', pgn_text, flags=re.MULTILINE)
    # Collapse newlines inside comments (Lichess doesn't support multi-line comments)
    pgn_text = re.sub(r'\{[^}]*\}', lambda m: m.group().replace('\n', ' '), pgn_text, flags=re.DOTALL)
    # Collapse blank lines in movetext (they act as game separators in PGN)
    header, sep, movetext = pgn_text.partition('\n\n')
    if sep:
        movetext = re.sub(r'\n\n+', ' ', movetext)
        pgn_text = header + sep + movetext
    parts = re.split(r'(\{[^}]*\})', pgn_text)
    parts = [
        re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*', ' ', p) if not p.startswith('{') else p
        for p in parts
    ]
    return ''.join(parts).rstrip()
