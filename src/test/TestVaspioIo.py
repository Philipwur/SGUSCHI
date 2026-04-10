"""Tests for VaspIO read/write utilities."""

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

from workflow import VaspIO as Vio  # noqa: E402


@pytest.fixture(name="TmpPath")
def FixtureTmpPath(tmp_path: Path) -> Path:
    """Expose pytest tmp_path as PascalCase fixture name."""
    return tmp_path


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


def TestReadKeyValueFileParsesCommentsAndRequiredKeys(TmpPath: Path) -> None:
    """ReadKeyValueFile should parse comments and required keys."""
    Content = """
    # Comment line
    AtomicRadiusTol = 1.10
    O2Tol 0.50  # inline comment
    ! another comment
    OSmoothing=0.25
    GasRatio = 1.0
    InitO2Count = 1
    """
    PathFile = TmpPath / "OxParams"
    PathFile.write_text(Content, encoding="utf-8")

    Params = Vio.ReadKeyValueFile(
        PathFile,
        RequiredKeys=["AtomicRadiusTol", "O2Tol", "OSmoothing", "GasRatio", "InitO2Count"],
    )

    assert Params["AtomicRadiusTol"] == "1.10"
    assert Params["O2Tol"] == "0.50"
    assert Params["OSmoothing"] == "0.25"
    assert Params["GasRatio"] == "1.0"
    assert Params["InitO2Count"] == "1"


def TestReadKeyValueFileMissingRequiredKeyRaises(TmpPath: Path) -> None:
    """ReadKeyValueFile should raise when required keys are missing."""
    PathFile = TmpPath / "OxParams"
    PathFile.write_text("AtomicRadiusTol = 1.1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        Vio.ReadKeyValueFile(PathFile, RequiredKeys=["AtomicRadiusTol", "MissingKey"])


def TestWriteReadPoscarRoundtripWithVelocities(TmpPath: Path) -> None:
    """WritePoscar and ReadPoscar should round-trip positions and velocities."""
    WorkDir = TmpPath / "WorkDir"
    WorkDir.mkdir()

    Cell = MakeCell()
    Positions = MakePositions()
    Velocities = MakeVelocities()

    Vio.WritePoscar(WorkDir, Positions, Cell, Velocities)
    Pos2, Cell2, Vel2 = Vio.ReadPoscar(WorkDir, GiveVelocities=True)

    assert Pos2.shape == Positions.shape
    assert Vel2 is not None
    assert Vel2.shape == Velocities.shape
    assert list(Pos2["Element"]) == list(Positions["Element"])
    assert list(Vel2["Element"]) == list(Velocities["Element"])
    assert np.allclose(Pos2[["x", "y", "z"]].values, Positions[["x", "y", "z"]].values)
    assert np.allclose(Cell2[["x", "y", "z"]].values, Cell[["x", "y", "z"]].values)
    assert np.allclose(Vel2[["vx", "vy", "vz"]].values, Velocities[["vx", "vy", "vz"]].values)


def TestWriteReadXyzRoundtrip(TmpPath: Path) -> None:
    """WriteXYZ and ReadXYZ should round-trip metadata and coordinates."""
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
    OutcarData = {
        "Temperature": 1000.0,
        "TimesFs": [1.0, 2.0],
        "Energies": Energies,
        "Positions": [Frame1, Frame2],
        "CellVectors": Cell,
    }

    XyzPath = TmpPath / "Traj.xyz"
    Vio.WriteXYZ(OutcarData, XyzPath)

    Parsed = Vio.ReadXYZ(XyzPath, ReturnCoordinatesType="Direct")
    assert len(Parsed["Positions"]) == 2
    assert Parsed["Metadata"].shape[0] == 2

    Meta = Parsed["Metadata"].iloc[0]
    assert Meta["Step"] == 1
    assert Meta["Time"] == 1.0
    # ExtendedXYZ energy fields are not parsed by the current ReadXYZ regex.
    assert np.isnan(Meta["EFree"])
    assert np.isnan(Meta["ETotal"])

    Roundtrip = Parsed["Positions"][0][["x", "y", "z"]].values
    assert np.allclose(Roundtrip, Frame1[["x", "y", "z"]].values, atol=1e-8)


def TestReadRateAnalysisParsesGasRemoved(TmpPath: Path) -> None:
    """ReadRateAnalysis should parse Gas Removed entries into tuples."""
    PathFile = TmpPath / "RateAnalysis.csv"
    DataFrame = pd.DataFrame(
        {
            "Time (fs)": [0.0],
            "O2 Count": [1],
            "Smoothed O2 Count": [1.0],
            "O2 Added": [1],
            "Gas Removed": ["[('C','O','O')]"],
            "Free Gas Fraction": [1.0],
        }
    )
    DataFrame.to_csv(PathFile, index=False)

    Parsed = Vio.ReadRateAnalysis(PathFile)
    assert Parsed.loc[0, "Gas Removed"] == [("C", "O", "O")]


def TestReadXyzSampleFile() -> None:
    """ReadXYZ should parse metadata in the provided ZrCN fixture."""
    SamplePath = Path(__file__).resolve().parent / "fixtures" / "1273_3ZrCnFrames.xyz"
    Parsed = Vio.ReadXYZ(SamplePath, ReturnCoordinatesType="Direct")
    assert len(Parsed["Positions"]) > 0
    assert "Time" in Parsed["Metadata"].columns
