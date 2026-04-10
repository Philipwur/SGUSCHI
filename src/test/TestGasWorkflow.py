"""Tests for gas identification using recorded XYZ frames."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest


def EnsureSrcOnPath() -> None:
    """Insert the project src directory into sys.path if missing."""
    RootDir = Path(__file__).resolve().parents[2]
    SrcDir = RootDir / "src"
    if str(SrcDir) not in sys.path:
        sys.path.insert(0, str(SrcDir))


EnsureSrcOnPath()

from workflow import OxidationAnalysis as An  # noqa: E402
from workflow import OxidationStep as Ox  # noqa: E402
from workflow import VaspIO as Vio  # noqa: E402


AtomicRadiusTol = 1.50
CovalentRadii = {
    "Zr": 1.45,
    "O": 0.66,
    "C": 0.76,
    "N": 0.71,
}


def LoadXyz(PathFile: Path) -> Tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Load positions, metadata, and cell from an XYZ file."""
    Parsed = Vio.ReadXYZ(PathFile, ReturnCoordinatesType="Direct")
    Positions = Parsed["Positions"]
    Metadata = Parsed["Metadata"]
    Cell = Parsed["CellDim"]
    if Cell is None:
        pytest.skip(f"Missing lattice information in {PathFile.name}")
    return Positions, Metadata, Cell


def CountMolecules(Gasses: pd.DataFrame, Molecule: Tuple[str, ...]) -> int:
    """Count occurrences of a specific molecule tuple."""
    Count = 0
    for Mol in Gasses["Molecule"]:
        if tuple(Mol) == Molecule:
            Count += 1
    return Count


def FindTimeIndex(Metadata: pd.DataFrame, Target: float, Tol: float = 1e-3) -> int:
    """Locate the frame index with metadata Time closest to target."""
    if "Time" not in Metadata.columns:
        pytest.skip("Time column missing from metadata.")
    Diffs = (Metadata["Time"] - Target).abs()
    Index = int(Diffs.idxmin())
    if float(Diffs.loc[Index]) > Tol:
        pytest.skip(f"Time {Target} not found within tolerance {Tol}.")
    return Index


def FindGasses(Frame: pd.DataFrame, Cell: pd.DataFrame) -> pd.DataFrame:
    """Find gas-like molecules for a given frame."""
    return An.FindGases(
        Frame,
        Cell,
        CovalentRadii=CovalentRadii,
        AtomicRadiusTol=AtomicRadiusTol,
        MinimumComplexity=2,
        MaximumComplexity=3,
        ReturnBondMatrix=False,
    )


def TestTimeAddedFrame1Has10O2AndNoRemovedGas() -> None:
    """TimeAdded frame 1 should have 10 O2 and no removed gas."""
    PathFile = Path(__file__).resolve().parent / "fixtures" / "1273_3TimeAddedFrames.xyz"
    Positions, _, Cell = LoadXyz(PathFile)

    Frame1 = Positions[0]
    Gasses = FindGasses(Frame1, Cell)

    assert CountMolecules(Gasses, ("O", "O")) == 10
    NonO2 = [tuple(Mol) for Mol in Gasses["Molecule"] if tuple(Mol) != ("O", "O")]
    assert len(NonO2) == 0

    assert Ox.CreateGassesRemovedStr(Gasses) == "[]"


def TestTimeAddedFrame4559Has4O2And1Co() -> None:
    """TimeAdded frame 4559 should have 4 O2 and 1 CO."""
    PathFile = Path(__file__).resolve().parent / "fixtures" / "1273_3TimeAddedFrames.xyz"
    Positions, _, Cell = LoadXyz(PathFile)

    Frame4559 = Positions[1]
    Gasses = FindGasses(Frame4559, Cell)

    assert CountMolecules(Gasses, ("O", "O")) == 4
    assert CountMolecules(Gasses, ("C", "O")) == 1

    NonO2 = [tuple(Mol) for Mol in Gasses["Molecule"] if tuple(Mol) != ("O", "O")]
    assert len(NonO2) == 1

    Removed = ast.literal_eval(Ox.CreateGassesRemovedStr(Gasses))
    assert ("C", "O") in Removed


def TestZrCnFrame1Has10O2AndNoCGas() -> None:
    """ZrCN frame 1 should have 10 O2 and no C-bearing gas."""
    PathFile = Path(__file__).resolve().parent / "fixtures" / "1273_3ZrCnFrames.xyz"
    Positions, _, Cell = LoadXyz(PathFile)

    Frame1 = Positions[0]
    Gasses = FindGasses(Frame1, Cell)

    assert CountMolecules(Gasses, ("O", "O")) == 10
    assert CountMolecules(Gasses, ("C", "O")) == 0
    assert CountMolecules(Gasses, ("C", "O", "O")) == 0


def TestZrCnTime26344HasCoAndNoO2AndMetadata() -> None:
    """ZrCN time 26344 should have CO and no O2; metadata should be parsed."""
    PathFile = Path(__file__).resolve().parent / "fixtures" / "1273_3ZrCnFrames.xyz"
    Positions, Metadata, Cell = LoadXyz(PathFile)

    Index = FindTimeIndex(Metadata, 26344.0)
    assert abs(float(Metadata.loc[Index, "Time"]) - 26344.0) <= 1e-3

    Frame = Positions[Index]
    Gasses = FindGasses(Frame, Cell)

    assert CountMolecules(Gasses, ("O", "O")) == 0
    assert CountMolecules(Gasses, ("C", "O")) == 1
