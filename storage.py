"""Save cache, output directories and manifest."""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

from game import Game

CACHE_FILE = ".cache.json"


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


def load_manifest(run_dir):
    """Load manifest from run directory."""
    path = run_dir / "manifest.json"
    if not path.exists():
        print(f"Error: manifest not found at {path}. Run --detect first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
        return [Game.from_dict(g) for g in data["games"]]
