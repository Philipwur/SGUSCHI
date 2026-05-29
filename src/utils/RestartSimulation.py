"""
RestartSimulation.py — automatic recovery from a corrupted / failed VolSearch run.

Motivation
----------
DetermineSize.x can (when it misbehaves) write overflow values (``***``) or
otherwise implausible lattice vectors into ``lattice_predict.out``. volsearch_cont
splices that block straight into the next ``POSCAR``. The result is a corrupted
``Dir_VolSearch/POSCAR`` and (usually) a ``sguschi_failed`` marker. Simply
relaunching volsearch_cont would resubmit VASP with the broken lattice.

This module detects that state and rolls the simulation back to the last clean
step folder, re-running that step from its own (saved, clean) POSCAR. It reuses
RollbackTrajectory for the actual file surgery + XYZ/RateAnalysis repair, then
resets the markers volsearch_cont's startup guard inspects so the next launch
resubmits the recovered geometry instead of rebuilding from CONTCAR.

Usage
-----
Called automatically by SGUSCHI.py before relaunching a pending simulation.
Can also be run standalone for a single Dir_VolSearch:

    python src/utils/RestartSimulation.py [/path/to/Dir_VolSearch]

(defaults to the current directory).
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

# A lattice vector component this large is never physical for these small cells
# (solid + gas region is at most a few tens of Angstrom). Catches silent drift
# that grew large but had not yet overflowed the Fortran F15.6 write field.
MaxLatticeComponent = 1000.0

# A near-zero cell volume means the lattice is degenerate / unusable.
MinCellVolume = 1.0e-6


class RecoveryError(Exception):
    """Raised when a corrupted POSCAR is found but cannot be safely recovered."""


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------

def _Determinant3(Matrix: List[List[float]]) -> float:
    A, B, C = Matrix
    return (
        A[0] * (B[1] * C[2] - B[2] * C[1])
        - A[1] * (B[0] * C[2] - B[2] * C[0])
        + A[2] * (B[0] * C[1] - B[1] * C[0])
    )


def LatticeIsCorrupt(PoscarPath: Union[str, Path]) -> Tuple[bool, str]:
    """Return (IsCorrupt, Reason) for the lattice block of a POSCAR file.

    Detects: Fortran overflow markers (``***``), non-numeric or non-finite
    values, too-few columns, implausibly large components, and degenerate
    (near-zero-volume) cells. A clean POSCAR returns (False, "").
    """
    PoscarPath = Path(PoscarPath)
    try:
        Lines = PoscarPath.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as Error:
        return True, "could not read POSCAR: {}".format(Error)

    if len(Lines) < 5:
        return True, "POSCAR has fewer than 5 lines"

    # Line 2 (index 1) is the global scale factor.
    ScaleToken = Lines[1].split()[0] if Lines[1].split() else ""
    if "*" in Lines[1]:
        return True, "scale line contains overflow markers (***)"
    try:
        Scale = float(ScaleToken)
    except ValueError:
        return True, "scale line is non-numeric ({!r})".format(ScaleToken)
    if not math.isfinite(Scale) or Scale == 0.0:
        return True, "scale factor is non-finite or zero ({!r})".format(ScaleToken)

    # Lines 3-5 (indices 2-4) are the three lattice vectors.
    Matrix: List[List[float]] = []
    for Offset, RawRow in enumerate(Lines[2:5]):
        RowIndex = Offset + 1
        if "*" in RawRow:
            return True, "lattice row {} contains overflow markers (***)".format(RowIndex)
        Tokens = RawRow.split()
        if len(Tokens) < 3:
            return True, "lattice row {} has fewer than 3 values".format(RowIndex)
        Values: List[float] = []
        for Token in Tokens[:3]:
            try:
                Value = float(Token)
            except ValueError:
                return True, "lattice row {} has non-numeric value {!r}".format(RowIndex, Token)
            if not math.isfinite(Value):
                return True, "lattice row {} has non-finite value {!r}".format(RowIndex, Token)
            if abs(Value) > MaxLatticeComponent:
                return True, (
                    "lattice row {} value {} exceeds {} A (implausible)".format(
                        RowIndex, Value, MaxLatticeComponent
                    )
                )
            Values.append(Value)
        Matrix.append(Values)

    if abs(_Determinant3(Matrix)) < MinCellVolume:
        return True, "lattice is degenerate (near-zero cell volume)"

    return False, ""


# ---------------------------------------------------------------------------
# Step folder selection
# ---------------------------------------------------------------------------

def StepFolders(VolSearchDir: Path) -> List[int]:
    """Return sorted numeric step folder indices present in VolSearchDir."""
    try:
        return sorted(
            int(Child.name)
            for Child in VolSearchDir.iterdir()
            if Child.is_dir() and Child.name.isdigit()
        )
    except OSError:
        return []


def FindRestartStep(VolSearchDir: Path) -> Optional[int]:
    """Return the last *clean* step G to re-run, or None if none is usable.

    G is one less than the first step whose saved POSCAR is corrupt or missing.
    If no step folder is corrupt (corruption is only in the live POSCAR), G is
    the highest step folder. The returned step is always a folder with a clean,
    existing POSCAR.
    """
    Steps = StepFolders(VolSearchDir)
    if not Steps:
        return None

    FirstBad: Optional[int] = None
    for Step in Steps:
        Poscar = VolSearchDir / str(Step) / "POSCAR"
        if not Poscar.exists() or LatticeIsCorrupt(Poscar)[0]:
            FirstBad = Step
            break

    if FirstBad is None:
        return Steps[-1]            # every saved step is clean → re-run the last one

    Restart = FirstBad - 1
    if Restart < Steps[0]:
        return None                 # even the earliest step is corrupt → cannot recover
    return Restart


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def _EnsureSrcOnPath() -> None:
    SrcDir = str(Path(__file__).resolve().parents[1])
    if SrcDir not in sys.path:
        sys.path.insert(0, SrcDir)


def _OutcarFrameCount(StepDir: Path) -> int:
    """Number of MD frames in StepDir/OUTCAR (raises if it cannot be parsed)."""
    _EnsureSrcOnPath()
    from workflow import VaspIO as Vio
    Data = Vio.OutcarParser(StepDir)
    return len(Data.get("TimesFs") or [])


def TruncateRateAnalysis(VolSearchDir: Path, KeepRows: int, Log: Callable[[str], None]) -> bool:
    """Keep the first KeepRows rows of RateAnalysis.csv (initial row + one per
    surviving step). Returns False — signalling the caller to fall back to a full
    FixRateAnalysis rebuild — if the file is missing or already too short.

    Truncation is exact because the smoothing/cumulative columns are causal: each
    row depends only on earlier frames, so the surviving rows are identical to a
    rebuild of the same step folders.
    """
    import pandas as pd

    WorkDirCsv = VolSearchDir / "RateAnalysis.csv"
    if not WorkDirCsv.exists():
        Log("  RateAnalysis.csv missing; will rebuild")
        return False
    try:
        Frame = pd.read_csv(WorkDirCsv)
    except Exception as Error:
        Log("  RateAnalysis.csv unreadable ({}); will rebuild".format(Error))
        return False
    if len(Frame) < KeepRows:
        Log("  RateAnalysis.csv has {} rows, need {}; will rebuild".format(len(Frame), KeepRows))
        return False

    Trimmed = Frame.head(KeepRows)
    Trimmed.to_csv(WorkDirCsv, index=False)

    RootDir = VolSearchDir.parents[1]
    TrajectoryName = VolSearchDir.parent.name
    RootCsv = RootDir / "xyz_files" / "RateAnalysis_{}.csv".format(TrajectoryName)
    if RootCsv.parent.exists():
        Trimmed.to_csv(RootCsv, index=False)
    Log("  RateAnalysis.csv truncated to {} rows".format(KeepRows))
    return True


def TruncateXyzByDroppingLastFrames(XyzPath: Path, DropFrames: int, Log: Callable[[str], None]) -> bool:
    """Drop the last DropFrames frames from an extended-XYZ trajectory in place.

    Streams the file once (cheap: reads atom-count headers, skips coordinate
    lines — no parsing into arrays), records frame-start byte offsets, then
    truncates at the start of the first frame to remove. Returns False if the
    file is missing/malformed or the frame accounting is inconsistent, so the
    caller can fall back to a full FixXYZ rebuild.
    """
    if DropFrames <= 0:
        return True  # nothing to drop; surviving prefix is already correct
    if not XyzPath.exists():
        Log("  XYZ trajectory missing; will rebuild")
        return False

    FrameStartOffsets: List[int] = []
    try:
        with open(XyzPath, "rb") as Handle:
            while True:
                Offset = Handle.tell()
                Line = Handle.readline()
                if not Line:
                    break
                Stripped = Line.strip()
                if not Stripped:
                    continue  # tolerate stray blank lines between frames
                try:
                    NumAtoms = int(Stripped)
                except ValueError:
                    Log("  XYZ malformed (expected atom count); will rebuild")
                    return False
                FrameStartOffsets.append(Offset)
                Handle.readline()  # comment line
                for _ in range(NumAtoms):
                    if not Handle.readline():
                        Log("  XYZ truncated mid-frame; will rebuild")
                        return False
    except OSError as Error:
        Log("  XYZ unreadable ({}); will rebuild".format(Error))
        return False

    Total = len(FrameStartOffsets)
    Keep = Total - DropFrames
    if Keep <= 0 or Keep >= Total:
        Log("  XYZ frame accounting off (total={}, drop={}); will rebuild".format(Total, DropFrames))
        return False

    CutOffset = FrameStartOffsets[Keep]
    with open(XyzPath, "r+b") as Handle:
        Handle.truncate(CutOffset)
    Log("  XYZ truncated: dropped {} of {} frames".format(DropFrames, Total))
    return True


def FastRollback(VolSearchDir: Path, TargetStep: int, Log: Callable[[str], None] = print) -> None:
    """Roll back to step TargetStep+1's clean POSCAR without rebuilding outputs.

    Mirrors RollbackTrajectory's file surgery (copy the continuation POSCAR,
    delete later step folders, drop WAVECAR) but repairs RateAnalysis/XYZ by
    *truncation* instead of full recomputation. Falls back to the slow-but-safe
    FixRateAnalysis / FixXYZ rebuild for whichever artifact cannot be truncated
    consistently.
    """
    VolSearchDir = Path(VolSearchDir).resolve()
    NextStep = TargetStep + 1
    SourcePoscar = VolSearchDir / str(NextStep) / "POSCAR"
    if not SourcePoscar.exists():
        raise SystemExit(1)  # caller's precondition violated; surfaced as RecoveryError

    # Count frames in the folders we are about to delete, while they still exist.
    DropFrames = 0
    FrameCountOk = True
    for Step in StepFolders(VolSearchDir):
        if Step > TargetStep:
            try:
                DropFrames += _OutcarFrameCount(VolSearchDir / str(Step))
            except Exception as Error:
                Log("  could not count frames in step {} ({}); XYZ will rebuild".format(Step, Error))
                FrameCountOk = False
                break

    # --- File surgery (same order RollbackTrajectory uses) ---
    shutil.copy(SourcePoscar, VolSearchDir / "POSCAR")
    for Step in StepFolders(VolSearchDir):
        if Step > TargetStep:
            shutil.rmtree(VolSearchDir / str(Step))
    (VolSearchDir / "WAVECAR").unlink(missing_ok=True)

    Remaining = StepFolders(VolSearchDir)
    Log("rolling back: kept steps 1..{}, re-running step {}".format(TargetStep, NextStep))

    # --- RateAnalysis: truncate to (surviving steps + initial row), else rebuild ---
    if not TruncateRateAnalysis(VolSearchDir, KeepRows=len(Remaining) + 1, Log=Log):
        _EnsureSrcOnPath()
        from utils.FixRateAnalysis import FixRateAnalysis
        FixRateAnalysis(VolSearchDir)

    # --- XYZ: drop the deleted folders' frames, else rebuild ---
    RootDir = VolSearchDir.parents[1]
    XyzPath = RootDir / "xyz_files" / "{}.xyz".format(VolSearchDir.parent.name)
    if not (FrameCountOk and TruncateXyzByDroppingLastFrames(XyzPath, DropFrames, Log=Log)):
        _EnsureSrcOnPath()
        from utils.FixXYZ import FixXYZ
        FixXYZ(VolSearchDir)


def _DefaultRollback(VolSearchDir: Path, TargetStep: int) -> None:
    """Adapter around RollbackTrajectory (imported lazily to keep deps optional)."""
    _EnsureSrcOnPath()
    from utils.RollbackTrajectory import RollbackTrajectory
    RollbackTrajectory(VolSearchDir, TargetStep)


def RecoverFailedSimulation(
    VolSearchDir: Union[str, Path],
    Log: Callable[[str], None] = print,
    RollbackFn: Callable[[Path, int], None] = FastRollback,
) -> Optional[str]:
    """Recover a corrupted/failed Dir_VolSearch so volsearch_cont can resume.

    Returns a one-line description of the action taken, or None if no recovery
    was needed. Raises RecoveryError if the POSCAR is corrupt but no clean step
    folder is available to roll back to (the caller should then refuse to launch
    that directory).

    Recovery only rolls back when the live POSCAR lattice is actually corrupt.
    A stale ``sguschi_failed`` marker with a clean POSCAR is treated as a
    transient failure: the marker is cleared and the run is relaunched as-is.
    """
    VolSearchDir = Path(VolSearchDir).resolve()
    Poscar = VolSearchDir / "POSCAR"
    FailedMarker = VolSearchDir / "sguschi_failed"

    PoscarCorrupt, Reason = (False, "")
    if Poscar.exists():
        PoscarCorrupt, Reason = LatticeIsCorrupt(Poscar)

    if not PoscarCorrupt:
        if FailedMarker.exists():
            FailedMarker.unlink()
            return "cleared sguschi_failed; POSCAR is clean, relaunching without rollback"
        return None

    Restart = FindRestartStep(VolSearchDir)
    if Restart is None:
        raise RecoveryError(
            "corrupted POSCAR ({}) but no clean step folder to recover from; "
            "manual intervention required".format(Reason)
        )
    if Restart < 2:
        # Keeping folders 1..G-1 would leave nothing, and volsearch_cont's
        # startup guard only auto-resubmits when nstep > 1.
        raise RecoveryError(
            "corrupted POSCAR ({}); only step {} is clean, which is too early "
            "for automatic rollback — restart this simulation manually".format(Reason, Restart)
        )

    Log("  corrupted POSCAR detected ({}); rolling back to clean step {}".format(Reason, Restart))

    # Re-run step G from its own clean POSCAR: RollbackTrajectory(T=G-1) copies
    # G/POSCAR into the live POSCAR, keeps 1..G-1, deletes >=G, drops WAVECAR,
    # and rebuilds the XYZ trajectory + RateAnalysis from the surviving steps.
    try:
        RollbackFn(VolSearchDir, Restart - 1)
    except SystemExit as Error:
        raise RecoveryError(
            "rollback to step {} failed (exit {}); manual intervention required".format(
                Restart, Error.code
            )
        )

    # Reset the markers volsearch_cont's startup guard inspects so it resubmits
    # nstep=G with OUR recovered POSCAR instead of rebuilding from CONTCAR.
    (VolSearchDir / "OUTCAR").unlink(missing_ok=True)
    (VolSearchDir / ".vasp_submitted_step").unlink(missing_ok=True)
    (VolSearchDir / "poscar_built_for_step").write_text(str(Restart), encoding="utf-8")
    FailedMarker.unlink(missing_ok=True)

    return "rolled back to clean step {} (corruption: {})".format(Restart, Reason)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Target = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        Result = RecoverFailedSimulation(Target)
    except RecoveryError as Err:
        print("RestartSimulation: {}".format(Err))
        raise SystemExit(1)
    if Result is None:
        print("RestartSimulation: no recovery needed for {}".format(Path(Target).resolve()))
    else:
        print("RestartSimulation: {}".format(Result))
