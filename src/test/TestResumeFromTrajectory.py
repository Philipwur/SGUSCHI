"""Tests for the ResumeFromTrajectory utility and the repair-tool seam guards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def EnsureSrcOnPath() -> None:
    RootDir = Path(__file__).resolve().parents[2]
    SrcDir = RootDir / "src"
    if str(SrcDir) not in sys.path:
        sys.path.insert(0, str(SrcDir))


EnsureSrcOnPath()

from utils import ResumeFromTrajectory as Rft  # noqa: E402
from utils.FolderUtils import ResumeSeamStep, NumericStepFolders  # noqa: E402
from workflow import VaspIO as vio  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

CUBIC_10 = "10 0 0 0 10 0 0 0 10"
# A distinct lattice for the final frame, to prove ParseLastXYZFrame uses the LAST
# frame's lattice (not ReadXYZ's averaged first+last cell).
CUBIC_12 = "12 0 0 0 12 0 0 0 12"


def WriteXYZFile(Path_: Path, Frames: list[tuple[str, list[tuple[str, float, float, float]]]]) -> None:
    """Frames = [(comment, [(el, x, y, z), ...]), ...] (coords Cartesian)."""
    Lines: list[str] = []
    for Comment, Atoms in Frames:
        Lines.append(str(len(Atoms)))
        Lines.append(Comment)
        for El, X, Y, Z in Atoms:
            Lines.append(f"{El} {X} {Y} {Z}")
    Path_.write_text("\n".join(Lines) + "\n", encoding="utf-8")


def SimpleAtoms(offset: float) -> list[tuple[str, float, float, float]]:
    """Species contiguous, O last: Zr, C, O."""
    return [
        ("Zr", 1.0 + offset, 1.0, 1.0),
        ("C", 3.0 + offset, 3.0, 3.0),
        ("O", 5.0 + offset, 5.0, 5.0),
    ]


def MakeRateAnalysis(Rows: int) -> pd.DataFrame:
    """A RateAnalysis with `Rows` rows (=> N = Rows - 1)."""
    Data = []
    for I in range(Rows):
        Data.append({
            "Time (fs)": float(I * 80),
            "O2 Count": 5,
            "Smoothed O2 Count": 5.0,
            "O2 Added": 10,
            "Gas Removed": "[]",
            "Free Gas Fraction": 1.0,
        })
    return pd.DataFrame(Data)


def WritePotcar(Path_: Path, Elements: list[str]) -> None:
    Blocks = []
    for El in Elements:
        Blocks.append(f"  TITEL  = PAW_PBE {El} 08Apr2002\n  END of PSCTR\n")
    Path_.write_text("".join(Blocks), encoding="utf-8")


def BuildInputsDir(TmpPath: Path, ThmexpMax: int = 10000,
                   PotcarElements: list[str] | None = None) -> Path:
    """Create a self-contained inputs directory for BuildWorkspaceTree."""
    Inputs = TmpPath / "inputs"
    Inputs.mkdir()
    (Inputs / "INCAR").write_text("TEBEG = 300\nTEEND = 300\nNSW = 80\n", encoding="utf-8")
    (Inputs / "KPOINTS").write_text("Auto\n0\nGamma\n1 1 1\n", encoding="utf-8")
    (Inputs / "job.in").write_text(
        f"temp = 300\nnavg = 3\nvaspcmd = sbatch\nthmexp_max = {ThmexpMax}\nthmexp_min = 1\n",
        encoding="utf-8",
    )
    (Inputs / "jobsub").write_text("#!/bin/bash\n#SBATCH --job-name='old'\n", encoding="utf-8")
    WritePotcar(Inputs / "POTCAR", PotcarElements if PotcarElements is not None else ["Zr", "C", "O"])
    (Inputs / "OxParams").write_text(
        "InitO2Count = 10\nGasRatio = 2\nTemperatures = [1273]\nNSims = 4\n"
        "AtomicRadiusTol = 1.50\nO2Tol = 0.5\nOSmoothing = 0.001\n",
        encoding="utf-8",
    )
    (Inputs / "CovalentRadii").write_text("Zr = 1.45\nO = 0.66\nC = 0.76\n", encoding="utf-8")
    return Inputs


# ---------------------------------------------------------------------------
# ParseLastXYZFrame
# ---------------------------------------------------------------------------

def TestParseLastFrameUsesFinalLattice(tmp_path: Path) -> None:
    XyzPath = tmp_path / "1273_3.xyz"
    WriteXYZFile(XyzPath, [
        (f'Lattice="{CUBIC_10}" Properties=species:S:1:pos:R:3 Step=1 Time=0.0', SimpleAtoms(0.0)),
        (f'Lattice="{CUBIC_12}" Properties=species:S:1:pos:R:3 Step=2 Time=80.0', SimpleAtoms(1.0)),
    ])

    Frame = Rft.ParseLastXYZFrame(XyzPath)

    # CellDim must be the LAST frame's lattice (12), not an average of 10 and 12.
    assert np.allclose(Frame.CellDim.values, np.eye(3) * 12.0)
    assert Frame.NumAtoms == 3
    assert Frame.Step == 2
    assert Frame.TimeFs == 80.0

    # Fractional coords == Cartesian @ inv(lattice) of the LAST frame.
    Cart = np.array([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [6.0, 5.0, 5.0]])
    Expected = Cart @ np.linalg.inv(np.eye(3) * 12.0)
    assert np.allclose(Frame.PositionFrac[["x", "y", "z"]].values, Expected)
    assert list(Frame.PositionFrac["Element"]) == ["Zr", "C", "O"]


def TestParseLastFrameLegacyHeader(tmp_path: Path) -> None:
    """Legacy 'Frame N Lattice=... Step= Time=' headers (no Properties tag)."""
    XyzPath = tmp_path / "873_1.xyz"
    WriteXYZFile(XyzPath, [
        (f'Frame 1 Lattice="{CUBIC_10}" Step=1 Time=0.0', SimpleAtoms(0.0)),
    ])
    Frame = Rft.ParseLastXYZFrame(XyzPath)
    # Cartesian auto-detected (coords > 1.2) and converted against the 10-cell.
    assert np.allclose(Frame.CellDim.values, np.eye(3) * 10.0)
    OxygenFrac = Frame.PositionFrac.loc[2, ["x", "y", "z"]].to_numpy(dtype=float)
    assert np.allclose(OxygenFrac, [0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# DetermineN
# ---------------------------------------------------------------------------

def TestDetermineNMatch(tmp_path: Path) -> None:
    XyzPath = tmp_path / "t.xyz"
    Frames = [(f'Lattice="{CUBIC_10}" Step={I} Time={I*80}.0', SimpleAtoms(0.0))
              for I in range(1, 161)]  # 160 frames => N=2
    WriteXYZFile(XyzPath, Frames)
    RatePath = tmp_path / "r.csv"
    MakeRateAnalysis(3).to_csv(RatePath, index=False)  # N = 2
    assert Rft.DetermineN(RatePath, XyzPath) == 2


def TestDetermineNMismatchAborts(tmp_path: Path) -> None:
    XyzPath = tmp_path / "t.xyz"
    WriteXYZFile(XyzPath, [(f'Lattice="{CUBIC_10}" Step=1 Time=0.0', SimpleAtoms(0.0))])  # 1 frame
    RatePath = tmp_path / "r.csv"
    MakeRateAnalysis(6).to_csv(RatePath, index=False)  # N=5 but xyz ~0 steps
    with pytest.raises(ValueError, match="mismatch"):
        Rft.DetermineN(RatePath, XyzPath)
    # --force proceeds with N from RateAnalysis.
    assert Rft.DetermineN(RatePath, XyzPath, Force=True) == 5


def TestDetermineNRefusesFreshSim(tmp_path: Path) -> None:
    XyzPath = tmp_path / "t.xyz"
    WriteXYZFile(XyzPath, [(f'Lattice="{CUBIC_10}" Step=1 Time=0.0', SimpleAtoms(0.0))])
    RatePath = tmp_path / "r.csv"
    MakeRateAnalysis(1).to_csv(RatePath, index=False)  # seed only => N=0
    with pytest.raises(ValueError, match="nothing to resume"):
        Rft.DetermineN(RatePath, XyzPath)


# ---------------------------------------------------------------------------
# ValidatePotcarOrder / CheckStopCondition
# ---------------------------------------------------------------------------

def TestValidatePotcarOrderPass(tmp_path: Path) -> None:
    Potcar = tmp_path / "POTCAR"
    WritePotcar(Potcar, ["Zr", "C", "O"])
    Rft.ValidatePotcarOrder(["Zr", "C", "O"], Potcar)  # no raise


def TestValidatePotcarOrderRejectsONotLast(tmp_path: Path) -> None:
    Potcar = tmp_path / "POTCAR"
    WritePotcar(Potcar, ["Zr", "O", "C"])
    with pytest.raises(ValueError, match="end in 'O'"):
        Rft.ValidatePotcarOrder(["Zr", "O", "C"], Potcar)


def TestValidatePotcarOrderRejectsMismatch(tmp_path: Path) -> None:
    Potcar = tmp_path / "POTCAR"
    WritePotcar(Potcar, ["C", "Zr", "O"])
    with pytest.raises(ValueError, match="does not match"):
        Rft.ValidatePotcarOrder(["Zr", "C", "O"], Potcar)


def TestCheckStopConditionRejectsNGeMax(tmp_path: Path) -> None:
    JobIn = tmp_path / "job.in"
    JobIn.write_text("thmexp_max = 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stop after"):
        Rft.CheckStopCondition(JobIn, N=5)
    Rft.CheckStopCondition(JobIn, N=4)  # ok


def TestCheckStopConditionRejectsMissingKey(tmp_path: Path) -> None:
    JobIn = tmp_path / "job.in"
    JobIn.write_text("temp = 300\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thmexp_max"):
        Rft.CheckStopCondition(JobIn, N=3)


# ---------------------------------------------------------------------------
# ResumeSeamStep helper
# ---------------------------------------------------------------------------

def TestResumeSeamStep(tmp_path: Path) -> None:
    assert ResumeSeamStep(tmp_path) is None
    (tmp_path / ".resume_seam").write_text("42\n# resumed at ...\n", encoding="utf-8")
    assert ResumeSeamStep(tmp_path) == 42
    (tmp_path / ".resume_seam").write_text("# only a comment\n", encoding="utf-8")
    assert ResumeSeamStep(tmp_path) is None


# ---------------------------------------------------------------------------
# End-to-end reconstruction
# ---------------------------------------------------------------------------

def MakeSource(tmp_path: Path, N: int = 2) -> tuple[Path, Path]:
    """Create a source xyz_files dir with one trajectory of N steps, plus inputs."""
    XyzDir = tmp_path / "src_xyz"
    XyzDir.mkdir()
    NumFrames = N * Rft.MD_STEPS_PER_CYCLE
    Frames = [(f'Lattice="{CUBIC_12}" Properties=species:S:1:pos:R:3 Step={I} Time={I*10}.0',
               SimpleAtoms(0.0)) for I in range(1, NumFrames + 1)]
    WriteXYZFile(XyzDir / "1273_3.xyz", Frames)
    MakeRateAnalysis(N + 1).to_csv(XyzDir / "RateAnalysis_1273_3.csv", index=False)
    Inputs = BuildInputsDir(tmp_path)
    return XyzDir, Inputs


def TestResumeTrajectoryBuildsConsistentWorkspace(tmp_path: Path) -> None:
    XyzDir, Inputs = MakeSource(tmp_path, N=3)
    Target = tmp_path / "workspace"
    Target.mkdir()

    Summary = Rft.ResumeTrajectory("1273_3", XyzDir / "1273_3.xyz",
                                   XyzDir / "RateAnalysis_1273_3.csv",
                                   Target, Inputs)

    assert Summary["N"] == 3
    Vsd = Target / "1273_3" / "Dir_VolSearch"
    assert Vsd.is_dir()

    # Invariant: placeholders == N, RateAnalysis rows == N+1.
    assert NumericStepFolders(Vsd) == [1, 2, 3]
    assert len(pd.read_csv(Vsd / "RateAnalysis.csv")) == 4

    # Clean submission state so volsearch_cont recovery resubmits.
    assert not (Vsd / "OUTCAR").exists()
    assert not (Vsd / ".vasp_submitted_step").exists()

    # Carried-forward outputs present; seam marker records N.
    assert (Target / "xyz_files" / "1273_3.xyz").exists()
    assert (Target / "xyz_files" / "RateAnalysis_1273_3.csv").exists()
    assert ResumeSeamStep(Vsd) == 3

    # Reconstructed POSCAR parses and has velocity-less Zr/C/O in POTCAR order.
    Position, _Cell = vio.ReadPoscar(FileName=Vsd / "POSCAR")
    assert list(dict.fromkeys(Position["Element"])) == ["Zr", "C", "O"]

    # Root inputs placed for OxidationStep.
    assert (Target / "OxParams").exists()
    assert (Target / "CovalentRadii").exists()


def TestResumeDryRunWritesNothing(tmp_path: Path) -> None:
    XyzDir, Inputs = MakeSource(tmp_path, N=2)
    Target = tmp_path / "workspace"
    Target.mkdir()

    Summary = Rft.ResumeTrajectory("1273_3", XyzDir / "1273_3.xyz",
                                   XyzDir / "RateAnalysis_1273_3.csv",
                                   Target, Inputs, DryRun=True)

    assert Summary["N"] == 2
    assert "dry-run" in str(Summary["action"])
    assert not (Target / "1273_3").exists()


def TestInferLayoutFromWorkspaceRoot(tmp_path: Path) -> None:
    (tmp_path / "xyz_files").mkdir()
    Target, Inputs, XyzDir = Rft.InferLayout(tmp_path)
    assert Target == tmp_path.resolve()
    assert Inputs == tmp_path.resolve()
    assert XyzDir == (tmp_path / "xyz_files").resolve()


def TestInferLayoutFromInsideXyzFiles(tmp_path: Path) -> None:
    XyzFiles = tmp_path / "xyz_files"
    XyzFiles.mkdir()
    (XyzFiles / "1273_3.xyz").write_text("0\n\n", encoding="utf-8")
    Target, Inputs, XyzDir = Rft.InferLayout(XyzFiles)
    assert Target == tmp_path.resolve()
    assert XyzDir == XyzFiles.resolve()


def TestInferLayoutUnrelatedDirReturnsNone(tmp_path: Path) -> None:
    assert Rft.InferLayout(tmp_path) is None


def TestStarterSummaryWritten(tmp_path: Path) -> None:
    XyzDir, Inputs = MakeSource(tmp_path, N=2)
    Target = tmp_path / "workspace"
    Target.mkdir()
    Rft.ResumeTrajectory("1273_3", XyzDir / "1273_3.xyz",
                         XyzDir / "RateAnalysis_1273_3.csv", Target, Inputs)

    Rft.WriteStarterSummary(Target)

    from utils import SimulationSummary as Summary
    assert (Target / Summary.SUMMARY_TXT).exists()
    Text = (Target / Summary.SUMMARY_TXT).read_text(encoding="utf-8")
    assert "1273_3" in Text


def TestCarriedXYZAppendsNotTruncates(tmp_path: Path) -> None:
    """After reconstruction, WriteXYZ against the carried xyz must APPEND."""
    XyzDir, Inputs = MakeSource(tmp_path, N=2)
    Target = tmp_path / "workspace"
    Target.mkdir()
    Rft.ResumeTrajectory("1273_3", XyzDir / "1273_3.xyz",
                         XyzDir / "RateAnalysis_1273_3.csv", Target, Inputs)

    CarriedXyz = Target / "xyz_files" / "1273_3.xyz"
    FramesBefore = Rft.CountXYZFrames(CarriedXyz)

    # Minimal 1-frame OutcarData for WriteXYZ (fractional positions).
    OutcarData = {
        "CellVectors": pd.DataFrame(np.eye(3) * 12.0, columns=["x", "y", "z"]),
        "Positions": [pd.DataFrame({
            "Element": ["Zr", "C", "O"],
            "x": [0.1, 0.3, 0.5], "y": [0.1, 0.3, 0.5], "z": [0.1, 0.3, 0.5],
        })],
        "Energies": pd.DataFrame([{"Step": 1, "EFree": -1.0, "ETotal": -1.0}]),
        "TimesFs": [10.0],
    }
    vio.WriteXYZ(OutcarData, FilePath=CarriedXyz)

    assert Rft.CountXYZFrames(CarriedXyz) == FramesBefore + 1


# ---------------------------------------------------------------------------
# Seam guards on the repair/rollback tools
# ---------------------------------------------------------------------------

def MakeResumedVsd(tmp_path: Path, N: int = 3) -> Path:
    """A minimal Dir_VolSearch carrying a .resume_seam marker."""
    Vsd = tmp_path / "1273_3" / "Dir_VolSearch"
    Vsd.mkdir(parents=True)
    for I in range(1, N + 1):
        (Vsd / str(I)).mkdir()
    (Vsd / ".resume_seam").write_text(f"{N}\n", encoding="utf-8")
    return Vsd


def TestFixXYZRefusesAcrossSeam(tmp_path: Path) -> None:
    from utils.FixXYZ import FixXYZ
    Vsd = MakeResumedVsd(tmp_path)
    with pytest.raises(SystemExit):
        FixXYZ(Vsd)


def TestFixRateAnalysisRefusesAcrossSeam(tmp_path: Path) -> None:
    from utils.FixRateAnalysis import FixRateAnalysis
    Vsd = MakeResumedVsd(tmp_path)
    with pytest.raises(SystemExit):
        FixRateAnalysis(Vsd)


def TestRollbackRefusesAcrossSeam(tmp_path: Path) -> None:
    from utils.RollbackTrajectory import RollbackTrajectory
    Vsd = MakeResumedVsd(tmp_path)
    with pytest.raises(SystemExit):
        RollbackTrajectory(Vsd, 2)


def TestFixXYZWithoutSeamIsNotSeamBlocked(tmp_path: Path) -> None:
    """Without the marker, FixXYZ reaches its normal no-folders path (FileNotFoundError),
    proving the seam guard does not fire on ordinary workspaces."""
    from utils.FixXYZ import FixXYZ
    Vsd = tmp_path / "1273_3" / "Dir_VolSearch"
    Vsd.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        FixXYZ(Vsd)
