"""Add claude_tools/ to sys.path so setup.py's `from constants import ...` resolves."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "claude_tools"))
