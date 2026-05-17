"""Save cache, output directories and manifest."""

from pathlib import Path
from datetime import datetime


def make_run_dir(stem):
    """Create a new timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("output") / f"{stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
