"""Compatibility boundary: Browser calls the shared context-processing rules."""
import sys
from pathlib import Path
_src = str(Path(__file__).resolve().parents[6] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
from trainable_agent.context_processing.browser import *
