# pdf-to-pgn

Extract annotated games from a PDF into PGN.

## How it works

1. **Pass 1**: Extract main-line moves as clean PGN from page images.
2. **Validate**: Check moves with `python-chess`; log any illegal moves.
3. **Pass 2**: Attach commentary to each move as PGN `{ }` comments.
4. **Post-process**: Strip result headers and collapse comments for Lichess compatibility.

## Usage

Process a single page range:

```bash
python convert.py book.pdf --pages 1-5
```

## Known limitations

- **Ambiguous moves**: The Claude Vision API does not handle disambiguation very well.
