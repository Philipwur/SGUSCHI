"""Small, dependency-free helpers for Dir_VolSearch folder layout.

Stdlib-only on purpose: this is a leaf module so it can be shared by the
lightweight SimulationSummary scanner and the heavier repair/workflow utilities
without pulling in pandas or creating import cycles.
"""

from pathlib import Path
from typing import List, Tuple


def NumericStepFolders(WorkDir: Path) -> List[int]:
    """Return sorted numeric step folders in a Dir_VolSearch directory."""
    if not WorkDir.is_dir():
        return []
    return sorted(
        int(Child.name)
        for Child in WorkDir.iterdir()
        if Child.is_dir() and Child.name.isdigit()
    )


def TrajectoryRoot(WorkDir: Path) -> Tuple[Path, str]:
    """Return (RootDir, TrajectoryName) for a Dir_VolSearch path."""
    return WorkDir.parents[1], WorkDir.parent.name
