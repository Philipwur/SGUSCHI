"""Tests for the OxidationStep workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def EnsureSrcOnPath() -> None:
    """Insert the project src directory into sys.path if missing."""
    RootDir = Path(__file__).resolve().parents[2]
    SrcDir = RootDir / "src"
    if str(SrcDir) not in sys.path:
        sys.path.insert(0, str(SrcDir))


EnsureSrcOnPath()

from workflow import OxidationStep as Ox  # noqa: E402
from workflow import VaspIO as Vio  # noqa: E402


@pytest.fixture(name="TmpPath")
def FixtureTmpPath(tmp_path: Path) -> Path:
    """Expose pytest tmp_path as PascalCase fixture name."""
    return tmp_path


@pytest.fixture(name="MonkeyPatch")
def FixtureMonkeyPatch(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Expose pytest monkeypatch as PascalCase fixture name."""
    return monkeypatch


def MakeCell() -> pd.DataFrame:
    """Return a simple cubic cell for testing."""
    return pd.DataFrame(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        columns=["x", "y", "z"],
    )


def MakePositions() -> pd.DataFrame:
    """Return a small 3-atom structure."""
    return pd.DataFrame(
        {
            "Element": ["Zr", "O", "O"],
            "x": [0.0, 0.5, 0.55],
            "y": [0.0, 0.5, 0.5],
            "z": [0.0, 0.5, 0.5],
        }
    )


def MakeVelocities() -> pd.DataFrame:
    """Return velocities aligned with MakePositions()."""
    return pd.DataFrame(
        {
            "Element": ["Zr", "O", "O"],
            "vx": [0.01, -0.02, 0.03],
            "vy": [0.00, 0.01, -0.01],
            "vz": [0.02, -0.01, 0.00],
        }
    )


def MakeOutcarData() -> dict:
    """Build minimal OutcarData structure for WriteXYZ."""
    Cell = MakeCell()
    Frame1 = MakePositions()
    Frame2 = Frame1.copy()
    Frame2.loc[1, "x"] = 0.52

    Energies = pd.DataFrame(
        {
            "Step": [1, 2],
            "EFree": [-1.0, -1.1],
            "ETotal": [-0.9, -1.0],
            "Temperature": [1000.0, 1000.0],
            "Pressure": [0.0, 0.0],
        }
    )
    return {
        "Temperature": 1000.0,
        "TimesFs": [1.0, 2.0],
        "Energies": Energies,
        "Positions": [Frame1, Frame2],
        "CellVectors": Cell,
    }


def WriteRootInputs(Root: Path) -> None:
    """Write OxParams and CovalentRadii into the test root."""
    (Root / "xyz_files").mkdir(parents=True, exist_ok=True)

    (Root / "OxParams").write_text(
        "\n".join(
            [
                "AtomicRadiusTol = 1.10",
                "O2Tol = 0.10",
                "OSmoothing = 0.50",
                "GasRatio = 1.0",
                "InitO2Count = 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (Root / "CovalentRadii").write_text(
        "\n".join(
            [
                "O = 0.66",
                "C = 0.76",
                "Zr = 1.60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def MakeWorkDir(TmpPath: Path) -> Path:
    """Create a test workdir with POSCAR and required inputs."""
    Root = TmpPath / "Root"
    WorkDir = Root / "TrajA" / "Dir_VolSearch"
    WorkDir.mkdir(parents=True)
    StepDir = WorkDir / "1"
    StepDir.mkdir()
    (StepDir / "OUTCAR").write_text("", encoding="utf-8")

    WriteRootInputs(Root)

    Cell = MakeCell()
    Positions = MakePositions()
    Velocities = MakeVelocities()

    Vio.WritePoscar(WorkDir, Positions, Cell, Velocities)

    return WorkDir


def TestValidateOutcarDataRejectsUnknownElements(TmpPath: Path) -> None:
    """Parsed OUTCAR frames containing X should fail before analysis mutates files."""
    OutcarPath = TmpPath / "OUTCAR"
    Frame = pd.DataFrame(
        {
            "Element": ["Zr", "X"],
            "x": [0.0, 0.5],
            "y": [0.0, 0.5],
            "z": [0.0, 0.5],
        }
    )
    OutcarData = {
        "Temperature": 1000.0,
        "TimesFs": [1.0],
        "Positions": [Frame],
    }

    with pytest.raises(ValueError, match="unresolved element labels"):
        Ox.ValidateOutcarData(OutcarData, OutcarPath)


def TestOxidationStepFailsWhenExpectedFolderMissing(TmpPath: Path) -> None:
    """The workflow should not continue when RateAnalysis points to a missing job folder."""
    WorkDir = MakeWorkDir(TmpPath)
    MissingFolder = WorkDir / "1"
    (MissingFolder / "OUTCAR").unlink()
    MissingFolder.rmdir()

    with pytest.raises(FileNotFoundError, match="Expected completed job folder"):
        Ox.main(WorkDir, TestCase=True)


def TestOxidationStepTestCaseTrueWritesTestOut(
    TmpPath: Path,
    MonkeyPatch: pytest.MonkeyPatch,
) -> None:
    """TestCase=True should write a test.out file without mutating inputs."""
    WorkDir = MakeWorkDir(TmpPath)

    MonkeyPatch.setattr(Ox.vio, "OutcarParser", lambda _: MakeOutcarData())

    Ox.main(WorkDir, TestCase=True)

    TestOut = WorkDir / "test.out"
    assert TestOut.exists()
    Text = TestOut.read_text(encoding="utf-8")
    assert "O2Added (Bool): 0" in Text
    assert "GasRemovedStr: []" in Text


def TestOxidationStepWritesOutputs(
    TmpPath: Path,
    MonkeyPatch: pytest.MonkeyPatch,
) -> None:
    """Normal execution should write XYZ, RateAnalysis, and summary outputs."""
    WorkDir = MakeWorkDir(TmpPath)
    Root = WorkDir.parents[1]

    MonkeyPatch.setattr(Ox.vio, "OutcarParser", lambda _: MakeOutcarData())

    Ox.main(WorkDir, TestCase=False)

    XyzPath = Root / "xyz_files" / "TrajA.xyz"
    assert XyzPath.exists()

    RatePath = WorkDir / "RateAnalysis.csv"
    assert RatePath.exists()

    ParsedRate = Vio.ReadRateAnalysis(RatePath)
    assert len(ParsedRate) == 2

    SummaryPath = Root / "xyz_files" / "RateAnalysis_TrajA.csv"
    assert SummaryPath.exists()

    ParsedXyz = Vio.ReadXYZ(XyzPath, ReturnCoordinatesType="Direct")
    assert len(ParsedXyz["Positions"]) == 2


def TestMaxRuntimeStopsSimulation(
    TmpPath: Path,
    MonkeyPatch: pytest.MonkeyPatch,
) -> None:
    """MaxRuntime reached: volsearch_is_done + maxruntime_reached written, exit 1."""
    WorkDir = MakeWorkDir(TmpPath)
    Root = WorkDir.parents[1]

    # MakeOutcarData has TimesFs[-1]=2.0; initial RateAnalysis row has Time(fs)=0.
    # Cumulative = 2.0 fs = 0.002 ps.  MaxRuntime = 0.001 ps → threshold exceeded.
    (Root / "OxParams").write_text(
        "AtomicRadiusTol = 1.10\nO2Tol = 0.10\nOSmoothing = 0.50\n"
        "GasRatio = 1.0\nInitO2Count = 1\nMaxRuntime = 0.001\n",
        encoding="utf-8",
    )

    MonkeyPatch.setattr(Ox.vio, "OutcarParser", lambda _: MakeOutcarData())

    with pytest.raises(SystemExit) as ExcInfo:
        Ox.main(WorkDir, TestCase=False)

    assert ExcInfo.value.code == 1
    assert (WorkDir / "volsearch_is_done").exists()
    assert (WorkDir / "maxruntime_reached").exists()


def TestMaxRuntimeNotReachedDoesNotStop(
    TmpPath: Path,
    MonkeyPatch: pytest.MonkeyPatch,
) -> None:
    """MaxRuntime not reached: simulation completes normally, no done marker written."""
    WorkDir = MakeWorkDir(TmpPath)
    Root = WorkDir.parents[1]

    # Cumulative = 0.002 ps.  MaxRuntime = 1000.0 ps → far from threshold.
    (Root / "OxParams").write_text(
        "AtomicRadiusTol = 1.10\nO2Tol = 0.10\nOSmoothing = 0.50\n"
        "GasRatio = 1.0\nInitO2Count = 1\nMaxRuntime = 1000.0\n",
        encoding="utf-8",
    )

    MonkeyPatch.setattr(Ox.vio, "OutcarParser", lambda _: MakeOutcarData())

    Ox.main(WorkDir, TestCase=False)

    assert not (WorkDir / "volsearch_is_done").exists()
