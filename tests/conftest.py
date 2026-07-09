from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `src/` importable when running `pytest` from the repo root without an
# editable install. The editable install via `pip install -e .` is the
# preferred path, but tests should also work bare.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tests should never pick up a developer's real .env.
os.environ.setdefault("AI_API_KEY", "test-key")
os.environ.setdefault("AI_PROVIDER", "openai")
