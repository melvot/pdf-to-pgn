"""Anthropic API client and move/commentary extraction."""

import os
import sys
import re

import anthropic


MODEL = "claude-sonnet-4-6"


def get_client():
    """Create and return an Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def strip_code_fences(text):
    """Remove markdown code fences from LLM output.
    Handles preamble text before the first code block."""
    # If there's a code fence anywhere, extract just its contents
    match = re.search(r'```[^\n]*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_pgn_from_response(text):
    text = strip_code_fences(text).strip()
    match = re.search(r'1\.', text)
    return text[match.start():] if match else text


def build_image_content(images_b64):
    """Build Claude API image content blocks."""
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img},
        }
        for img in images_b64
    ]


def pass1_extract_moves(client, images_b64, game_hint=None):
    """Pass 1: Extract clean PGN moves from page images (no commentary)."""
    target = f" for {game_hint}" if game_hint else ""
    ignore = " Ignore any other games on these pages." if game_hint else ""
    content = build_image_content(images_b64) + [
        {
            "type": "text",
            "text": (
                "These are the pages of the chess book, extract only"
                f"the PGN moves{target}.{ignore}"
            ),
        }
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0,
        system=(
            "You are a PGN extractor. These are pages from a chess book. "
            "Extract ONLY the chess moves for a single whole game as PGN. "
            "Use standard algebraic notation (K, Q, R, B, N for pieces). "
            "If a move in the source includes a file or rank disambiguator "
            "(e.g. Nbd2, Rac1), you MUST include it exactly. Never drop "
            "disambiguating characters. Do NOT include commentary, annotations, or variations."
            "Only include the main line moves. Do not include any game headers such as [Event],"
            "[White], [Black], [Result], [Opening]. Output ONLY the valid PGN moves, nothing else. "
            "Your response must start immediately with '1.' and contain nothing else."
        ),
        messages=[{"role": "user", "content": content}],
    )
    return extract_pgn_from_response(response.content[0].text)


def pass2_attach_commentary(client, images_b64, ocr_text, pgn_moves, game_hint=None):
    """Pass 2: Attach full book commentary to the PGN moves."""
    target = f" for {game_hint}" if game_hint else ""
    content = build_image_content(images_b64) + [
        {
            "type": "text",
            "text": f"""Here are the clean PGN moves{target} extracted from these chess book pages:

{pgn_moves}

And here is the OCR-extracted text from the same pages (chess piece symbols may be garbled but the prose is readable):

{ocr_text}

Using the page images for visual reference and layout ordering, and the OCR text for exact commentary wording, produce a complete PGN with ALL of the author's commentary embedded as PGN comments in curly braces {{}}.

CRITICAL:
- Annotate ONLY the game provided above. If the pages contain other games, ignore them completely.
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
