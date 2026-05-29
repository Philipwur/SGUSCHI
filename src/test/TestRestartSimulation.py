"""Tests for utils/RestartSimulation.py — corruption detection and rollback."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from utils.RestartSimulation import (
    FindRestartStep,
    LatticeIsCorrupt,
    RecoverFailedSimulation,
    RecoveryError,
    TruncateRateAnalysis,
    TruncateXyzByDroppingLastFrames,
)


@pytest.fixture(name="TmpPath")
def FixtureTmpPath(tmp_path: Path) -> Path:
    """Expose pytest tmp_path as PascalCase fixture name."""
    return tmp_path


CleanLattice = "10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 9.296344"
CorruptLattice = (
    "******************************      -0.000028\n"
    "******************************      -0.000136\n"
    "******************************       9.296344"
)


def WritePoscar(PoscarPath: Path, Lattice: str = CleanLattice, Scale: str = "1.0") -> None:
    """Write a minimal POSCAR with the given lattice block."""
    PoscarPath.parent.mkdir(parents=True, exist_ok=True)
    Body = (
        "comment line\n"
        "{scale}\n"
        "{lattice}\n"
        "C N Zr O\n"
        "1 1 1 1\n"
        "Direct\n"
        "0.0 0.0 0.0\n"
    ).format(scale=Scale, lattice=Lattice)
    PoscarPath.write_text(Body, encoding="utf-8")


def MakeSteps(VolDir: Path, CleanSteps, CorruptSteps=()) -> None:
    for Step in CleanSteps:
        WritePoscar(VolDir / str(Step) / "POSCAR")
    for Step in CorruptSteps:
        WritePoscar(VolDir / str(Step) / "POSCAR", Lattice=CorruptLattice)


# --------------------------------------------------------------------------- #
# LatticeIsCorrupt
# --------------------------------------------------------------------------- #

def TestCleanPoscarIsNotCorrupt(TmpPath: Path) -> None:
    Poscar = TmpPath / "POSCAR"
    WritePoscar(Poscar)
    Corrupt, Reason = LatticeIsCorrupt(Poscar)
    assert Corrupt is False
    assert Reason == ""


def TestOverflowMarkersDetected(TmpPath: Path) -> None:
    Poscar = TmpPath / "POSCAR"
    WritePoscar(Poscar, Lattice=CorruptLattice)
    Corrupt, Reason = LatticeIsCorrupt(Poscar)
    assert Corrupt is True
    assert "overflow" in Reason


def TestNonNumericLatticeDetected(TmpPath: Path) -> None:
    Poscar = TmpPath / "POSCAR"
    WritePoscar(Poscar, Lattice="10.0 0.0 0.0\nNaNx 10.0 0.0\n0.0 0.0 10.0")
    Corrupt, _ = LatticeIsCorrupt(Poscar)
    assert Corrupt is True


def TestImplausiblyLargeComponentDetected(TmpPath: Path) -> None:
    Poscar = TmpPath / "POSCAR"
    WritePoscar(Poscar, Lattice="50000.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0")
    Corrupt, Reason = LatticeIsCorrupt(Poscar)
    assert Corrupt is True
    assert "implausible" in Reason


def TestDegenerateCellDetected(TmpPath: Path) -> None:
    Poscar = TmpPath / "POSCAR"
    WritePoscar(Poscar, Lattice="10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 0.0")
    Corrupt, Reason = LatticeIsCorrupt(Poscar)
    assert Corrupt is True
    assert "degenerate" in Reason


def TestTruncatedPoscarDetected(TmpPath: Path) -> None:
    Poscar = TmpPath / "POSCAR"
    Poscar.write_text("comment\n1.0\n10.0 0.0 0.0\n", encoding="utf-8")
    Corrupt, _ = LatticeIsCorrupt(Poscar)
    assert Corrupt is True


# --------------------------------------------------------------------------- #
# FindRestartStep
# --------------------------------------------------------------------------- #

def TestAllCleanReturnsLastStep(TmpPath: Path) -> None:
    MakeSteps(TmpPath, CleanSteps=[1, 2, 3, 4, 5])
    assert FindRestartStep(TmpPath) == 5


def TestFirstCorruptStepSetsBoundary(TmpPath: Path) -> None:
    MakeSteps(TmpPath, CleanSteps=[1, 2, 3], CorruptSteps=[4, 5])
    assert FindRestartStep(TmpPath) == 3


def TestMissingPoscarTreatedAsBoundary(TmpPath: Path) -> None:
    MakeSteps(TmpPath, CleanSteps=[1, 2])
    (TmpPath / "3").mkdir()  # folder 3 exists but has no POSCAR
    WritePoscar(TmpPath / "4" / "POSCAR")
    assert FindRestartStep(TmpPath) == 2


def TestEarliestStepCorruptReturnsNone(TmpPath: Path) -> None:
    MakeSteps(TmpPath, CleanSteps=[], CorruptSteps=[1, 2])
    assert FindRestartStep(TmpPath) is None


def TestNoStepFoldersReturnsNone(TmpPath: Path) -> None:
    assert FindRestartStep(TmpPath) is None


# --------------------------------------------------------------------------- #
# RecoverFailedSimulation
# --------------------------------------------------------------------------- #

def TestCleanPoscarNoMarkerIsNoop(TmpPath: Path) -> None:
    WritePoscar(TmpPath / "POSCAR")
    Calls = []
    Result = RecoverFailedSimulation(
        TmpPath, Log=lambda _m: None, RollbackFn=lambda d, t: Calls.append((d, t))
    )
    assert Result is None
    assert Calls == []


def TestCleanPoscarWithFailedMarkerClearsMarker(TmpPath: Path) -> None:
    WritePoscar(TmpPath / "POSCAR")
    (TmpPath / "sguschi_failed").write_text("boom", encoding="utf-8")
    Calls = []
    Result = RecoverFailedSimulation(
        TmpPath, Log=lambda _m: None, RollbackFn=lambda d, t: Calls.append((d, t))
    )
    assert Result is not None and "relaunching without rollback" in Result
    assert not (TmpPath / "sguschi_failed").exists()
    assert Calls == []


def TestCorruptPoscarRollsBackAndResetsMarkers(TmpPath: Path) -> None:
    WritePoscar(TmpPath / "POSCAR", Lattice=CorruptLattice)
    MakeSteps(TmpPath, CleanSteps=[1, 2, 3, 4, 5])
    (TmpPath / "sguschi_failed").write_text("boom", encoding="utf-8")
    (TmpPath / "OUTCAR").write_text("stale", encoding="utf-8")
    (TmpPath / ".vasp_submitted_step").write_text("6", encoding="utf-8")

    Calls = []
    Result = RecoverFailedSimulation(
        TmpPath, Log=lambda _m: None, RollbackFn=lambda d, t: Calls.append((d, t))
    )

    # All saved steps clean → re-run step 5 → RollbackTrajectory(TargetStep=4).
    assert Calls == [(TmpPath, 4)]
    assert "step 5" in Result
    # Markers reset for volsearch_cont's startup guard.
    assert not (TmpPath / "OUTCAR").exists()
    assert not (TmpPath / ".vasp_submitted_step").exists()
    assert not (TmpPath / "sguschi_failed").exists()
    assert (TmpPath / "poscar_built_for_step").read_text(encoding="utf-8") == "5"


def TestCorruptPoscarFirstBadStepPicksPriorClean(TmpPath: Path) -> None:
    WritePoscar(TmpPath / "POSCAR", Lattice=CorruptLattice)
    MakeSteps(TmpPath, CleanSteps=[1, 2, 3], CorruptSteps=[4, 5])
    Calls = []
    RecoverFailedSimulation(
        TmpPath, Log=lambda _m: None, RollbackFn=lambda d, t: Calls.append((d, t))
    )
    # First corrupt is step 4 → re-run step 3 → RollbackTrajectory(TargetStep=2).
    assert Calls == [(TmpPath, 2)]
    assert (TmpPath / "poscar_built_for_step").read_text(encoding="utf-8") == "3"


def TestRollbackWritesNoteToLogOut(TmpPath: Path) -> None:
    VolDir = TmpPath / "873_1" / "Dir_VolSearch"
    VolDir.mkdir(parents=True)
    WritePoscar(VolDir / "POSCAR", Lattice=CorruptLattice)
    MakeSteps(VolDir, CleanSteps=[1, 2, 3])

    RecoverFailedSimulation(VolDir, Log=lambda _m: None, RollbackFn=lambda d, t: None)

    LogPath = VolDir.parent / "log.out"   # the {label} folder, one above Dir_VolSearch
    assert LogPath.exists()
    Text = LogPath.read_text(encoding="utf-8")
    assert "Automatic rollback addressed" in Text
    assert "clean step 3" in Text


def TestCorruptPoscarNoCleanStepRaises(TmpPath: Path) -> None:
    WritePoscar(TmpPath / "POSCAR", Lattice=CorruptLattice)
    MakeSteps(TmpPath, CleanSteps=[], CorruptSteps=[1, 2])
    with pytest.raises(RecoveryError):
        RecoverFailedSimulation(TmpPath, Log=lambda _m: None, RollbackFn=lambda d, t: None)


def TestCorruptPoscarOnlyStepOneCleanRaises(TmpPath: Path) -> None:
    WritePoscar(TmpPath / "POSCAR", Lattice=CorruptLattice)
    MakeSteps(TmpPath, CleanSteps=[1], CorruptSteps=[2])
    with pytest.raises(RecoveryError):
        RecoverFailedSimulation(TmpPath, Log=lambda _m: None, RollbackFn=lambda d, t: None)


# --------------------------------------------------------------------------- #
# TruncateRateAnalysis
# --------------------------------------------------------------------------- #

def MakeVolDir(TmpPath: Path) -> Path:
    """Return a Dir_VolSearch with the RootDir/<label>/Dir_VolSearch layout."""
    VolDir = TmpPath / "root" / "873_1" / "Dir_VolSearch"
    VolDir.mkdir(parents=True)
    (TmpPath / "root" / "xyz_files").mkdir(parents=True)
    return VolDir


def WriteRateAnalysis(VolDir: Path, NumRows: int) -> None:
    Frame = pd.DataFrame(
        {
            "Time (fs)": [float(I) for I in range(NumRows)],
            "O2 Count": list(range(NumRows)),
            "Smoothed O2 Count": [float(I) for I in range(NumRows)],
            "O2 Added": list(range(NumRows)),
            "Gas Removed": ["[]"] * NumRows,
            "Free Gas Fraction": [1.0] * NumRows,
        }
    )
    Frame.to_csv(VolDir / "RateAnalysis.csv", index=False)


def TestTruncateRateAnalysisKeepsPrefixAndMirrorsToRoot(TmpPath: Path) -> None:
    VolDir = MakeVolDir(TmpPath)
    WriteRateAnalysis(VolDir, NumRows=10)
    Ok = TruncateRateAnalysis(VolDir, KeepRows=4, Log=lambda _m: None)
    assert Ok is True

    Kept = pd.read_csv(VolDir / "RateAnalysis.csv")
    assert len(Kept) == 4
    assert list(Kept["O2 Count"]) == [0, 1, 2, 3]

    RootCsv = TmpPath / "root" / "xyz_files" / "RateAnalysis_873_1.csv"
    assert len(pd.read_csv(RootCsv)) == 4


def TestTruncateRateAnalysisFallsBackWhenTooShort(TmpPath: Path) -> None:
    VolDir = MakeVolDir(TmpPath)
    WriteRateAnalysis(VolDir, NumRows=3)
    assert TruncateRateAnalysis(VolDir, KeepRows=5, Log=lambda _m: None) is False


def TestTruncateRateAnalysisFallsBackWhenMissing(TmpPath: Path) -> None:
    VolDir = MakeVolDir(TmpPath)
    assert TruncateRateAnalysis(VolDir, KeepRows=2, Log=lambda _m: None) is False


# --------------------------------------------------------------------------- #
# TruncateXyzByDroppingLastFrames
# --------------------------------------------------------------------------- #

def WriteXyz(XyzPath: Path, AtomCounts) -> None:
    """Write an extended-XYZ file with one frame per entry in AtomCounts."""
    Lines = []
    for FrameIndex, Count in enumerate(AtomCounts):
        Lines.append(str(Count))
        Lines.append('Lattice="..." Step={}'.format(FrameIndex + 1))
        for AtomIndex in range(Count):
            Lines.append("O 0.0 0.0 {}.0".format(AtomIndex))
    XyzPath.write_text("\n".join(Lines) + "\n", encoding="utf-8")


def CountXyzFrames(XyzPath: Path) -> int:
    Steps = 0
    for Line in XyzPath.read_text(encoding="utf-8").splitlines():
        if Line.startswith("Lattice="):
            Steps += 1
    return Steps


def TestTruncateXyzDropsLastFrames(TmpPath: Path) -> None:
    Xyz = TmpPath / "traj.xyz"
    WriteXyz(Xyz, AtomCounts=[3, 4, 3, 5, 3])  # varying atom counts per frame
    Ok = TruncateXyzByDroppingLastFrames(Xyz, DropFrames=2, Log=lambda _m: None)
    assert Ok is True
    assert CountXyzFrames(Xyz) == 3
    # Surviving frames are an exact prefix.
    Text = Xyz.read_text(encoding="utf-8")
    assert "Step=3" in Text
    assert "Step=4" not in Text


def TestTruncateXyzDropZeroIsNoop(TmpPath: Path) -> None:
    Xyz = TmpPath / "traj.xyz"
    WriteXyz(Xyz, AtomCounts=[3, 4, 3])
    Before = Xyz.read_text(encoding="utf-8")
    assert TruncateXyzByDroppingLastFrames(Xyz, DropFrames=0, Log=lambda _m: None) is True
    assert Xyz.read_text(encoding="utf-8") == Before


def TestTruncateXyzInconsistentDropFallsBack(TmpPath: Path) -> None:
    Xyz = TmpPath / "traj.xyz"
    WriteXyz(Xyz, AtomCounts=[3, 4, 3])
    # Dropping all frames is inconsistent → signal rebuild rather than empty file.
    assert TruncateXyzByDroppingLastFrames(Xyz, DropFrames=3, Log=lambda _m: None) is False


def TestTruncateXyzMissingFileFallsBack(TmpPath: Path) -> None:
    assert TruncateXyzByDroppingLastFrames(
        TmpPath / "nope.xyz", DropFrames=1, Log=lambda _m: None
    ) is False
