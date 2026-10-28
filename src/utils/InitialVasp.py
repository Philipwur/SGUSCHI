"""Prepare the first INCAR before submission; never reset a started trajectory."""

import math
import sys
from pathlib import Path

# Also callable by volsearch_cont, from a trajectory's Dir_VolSearch.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import StatusLog
from utils.Incar import PatchIncar


def NeedsInitialVaspJob(VolSearchDir: Path) -> bool:
    """True only for an unsubmitted first step, including a rejected submission."""
    if any((VolSearchDir / Name).exists() for Name in (
        "OUTCAR", "OSZICAR", "CONTCAR", ".vasp_submitted_step", "volsearch_is_done",
    )):
        return False
    if any(Child.is_dir() and Child.name.isdigit() for Child in VolSearchDir.iterdir()):
        return False
    return StatusLog.LastEvent(VolSearchDir, "submitted") is None


def InitializeIncar(IncarPath: Path, Temp) -> None:
    """Apply SLUSCHI's initial settings to a new/rebuilt input, preserving POTIM.

    Handles duplicate/case-insensitive tags, semicolon-separated assignments and
    #/! comments. Write once, atomically, so a failure leaves the original intact.
    Callers must ensure this INCAR does not belong to an already submitted job.
    """
    Temperature = float(Temp)
    if not math.isfinite(Temperature) or Temperature <= 0:
        raise ValueError("Initial VASP temperature must be a positive finite number")
    Settings = {
        # Keep the existing volsearch_cont coefficient and awk print precision.
        "SIGMA": format(0.000086 * Temperature, ".6g"),
        "TEBEG": format(Temperature, ".15g"),
        "TEEND": format(Temperature, ".15g"),
        "NSW": "80",
        "SMASS": "0",
    }
    Original = IncarPath.read_text(encoding="utf-8")
    Text = PatchIncar(Original, Settings, Remove=("NBANDS",))
    if Text == Original:
        return
    Temporary = IncarPath.with_name(IncarPath.name + ".sguschi.tmp")
    try:
        Temporary.write_text(Text, encoding="utf-8")
        Temporary.chmod(IncarPath.stat().st_mode)
        Temporary.replace(IncarPath)
    finally:
        Temporary.unlink(missing_ok=True)


def PrepareInitialVasp(VolSearchDir: Path) -> None:
    """Initialize only unsubmitted inputs, including folders prepared by older versions."""
    if not NeedsInitialVaspJob(VolSearchDir):
        return
    from workflow.VaspIO import ReadKeyValueFile

    try:
        Params = ReadKeyValueFile(VolSearchDir / "job.in", RequiredKeys=["temp"])
        InitializeIncar(VolSearchDir / "INCAR", Params["temp"])
    except (OSError, ValueError) as Error:
        raise ValueError("{}: {}".format(VolSearchDir, Error)) from Error


if __name__ == "__main__":
    try:
        PrepareInitialVasp(Path.cwd())
    except (OSError, ValueError) as Error:
        print("FATAL initial VASP preparation: {}".format(Error), file=sys.stderr)
        sys.exit(1)
