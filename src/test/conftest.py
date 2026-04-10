#%%
"""Test helpers for SGUSCHI.

Ensures the project `src` directory is on `sys.path` for imports.
"""

from __future__ import annotations

import sys
from pathlib import Path


def EnsureSrcOnPath() -> None:
    """Insert the project src directory into sys.path if missing."""
    RootDir = Path(__file__).resolve().parents[2]
    SrcDir = RootDir / "src"
    if str(SrcDir) not in sys.path:
        sys.path.insert(0, str(SrcDir))


EnsureSrcOnPath()
