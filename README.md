# pdf-to-pgn

Chess books contain thousands of annotated games locked in PDF format. This tool extracts them into PGN so you can analyze them with an engine on Lichess or any chess software.

## How it works

1. **Pass 1** — Extract main-line moves as clean PGN from page images.
2. **Validate** — Check moves with `python-chess`; log any illegal moves.
3. **Pass 2** — Attach the author's full prose commentary to each move as PGN `{ }` comments.
4. **Post-process** — Strip result headers and collapse comments for Lichess compatibility.

## Requirements

```
pip install anthropic chess pymupdf pymupdf4llm
export ANTHROPIC_API_KEY=sk-...
```

## Usage

Process a single page range:

```bash
python convert.py book.pdf --pages 1-5
```

Full book pipeline:

```bash
python convert.py book.pdf --detect    # detect game boundaries
python convert.py book.pdf --generate  # extract PGN for each game
python convert.py book.pdf --combine   # merge into one file
# or all at once:
python convert.py book.pdf --book
```

Run `python convert.py --help` for all options.
