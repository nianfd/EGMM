from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_mining.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["--paper-dir", "data/paper1", *sys.argv[1:]]))
