"""Small, dependency-free helpers for Dir_VolSearch folder layout.

Stdlib-only on purpose: this is a leaf module so it can be shared by the
lightweight SimulationSummary scanner and the heavier repair/workflow utilities
without pulling in pandas or creating import cycles.
"""

from pathlib import Path
from typing import List, Optional, Tuple


# Name of the marker written by ResumeFromTrajectory into a resumed Dir_VolSearch.
# It records the seam step N: folders 1..N are empty placeholders that carry no
# per-step POSCAR/OUTCAR, so full-history repair/rollback cannot cross the seam.
RESUME_SEAM_MARKER = ".resume_seam"


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


def ResumeSeamStep(WorkDir: Path) -> Optional[int]:
    """Return the resume seam step N for a Dir_VolSearch, or None if not resumed.

    A workspace rebuilt by ResumeFromTrajectory carries a ``.resume_seam`` marker
    whose first non-comment token is the integer seam step N (folders 1..N are empty
    placeholders). Repair/rollback utilities read this to refuse operating across the
    seam. Returns None when the marker is absent or unparseable (i.e. an ordinary,
    non-resumed workspace behaves exactly as before).
    """
    MarkerPath = Path(WorkDir) / RESUME_SEAM_MARKER
    try:
        for RawLine in MarkerPath.read_text(encoding="utf-8").splitlines():
            Line = RawLine.strip()
            if not Line or Line.startswith("#"):
                continue
            try:
                return int(Line.split()[0])
            except (ValueError, IndexError):
                return None
        return None
    except OSError:
        return None
