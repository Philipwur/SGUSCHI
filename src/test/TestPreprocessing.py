"""Tests for OxidationPreprocessing.AddVacuum()."""

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

from preprocessing import OxidationPreprocessing as Opp  # noqa: E402


def MakeStructure(XPositions: list[float], GasRatio: float = 2.0) -> tuple:
    """Build minimal Position and CellDim DataFrames for testing."""
    Positions = pd.DataFrame({
        "Element": ["Zr"] * len(XPositions),
        "x": XPositions,
        "y": [0.5] * len(XPositions),
        "z": [0.5] * len(XPositions),
    })
    CellDim = pd.DataFrame({
        "x": [10.0, 0.0, 0.0],
        "y": [0.0, 5.0, 0.0],
        "z": [0.0, 0.0, 5.0],
    })
    return Positions, CellDim


class TestAddVacuumConsolidatesSplitBoundaryLayer:
    """Atoms at x≈0.001 and x≈0.999 (same layer, split by PBC) must end up
    in the same surface after expansion — not on opposite sides of the slab."""

    def test_split_layer_atoms_land_in_material_region(self):
        GasRatio = 2.0
        Scale = 1.0 + GasRatio
        # 6-layer structure; boundary layer straddles x=0
        XPos = [0.001, 0.999,  # boundary layer (split)
                0.167, 0.333, 0.500, 0.667, 0.833]
        Pos, Cell = MakeStructure(XPos, GasRatio)
        NewPos, _ = Opp.AddVacuum(Pos, Cell, GasRatio=GasRatio)

        MaterialLimit = 1.0 / Scale
        assert (NewPos["x"].values < MaterialLimit + 1e-9).all(), (
            "All material atoms must lie within [0, 1/scale) after expansion."
        )

    def test_split_halves_consolidate_near_same_x(self):
        GasRatio = 2.0
        XPos = [0.001, 0.999, 0.167, 0.333, 0.500, 0.667, 0.833]
        Pos, Cell = MakeStructure(XPos, GasRatio)
        NewPos, _ = Opp.AddVacuum(Pos, Cell, GasRatio=GasRatio)

        # The two atoms that were the split boundary layer
        X0 = NewPos["x"].iloc[0]  # was 0.001
        X1 = NewPos["x"].iloc[1]  # was 0.999
        assert abs(X0 - X1) < 0.01, (
            f"The two halves of the split boundary layer should consolidate "
            f"near the same x, but got {X0:.4f} and {X1:.4f}."
        )


class TestAddVacuumGasRegionIsCorrectFraction:
    """After expansion with GasRatio r, all atoms must be in [0, 1/(1+r))
    and the cell x-length must scale by (1+r)."""

    @pytest.mark.parametrize("GasRatio", [1.0, 2.0, 3.0, 0.5])
    def test_atoms_within_material_region(self, GasRatio):
        XPos = [0.1, 0.3, 0.5, 0.7, 0.9]
        Pos, Cell = MakeStructure(XPos)
        NewPos, NewCell = Opp.AddVacuum(Pos, Cell, GasRatio=GasRatio)

        Scale = 1.0 + GasRatio
        MaterialLimit = 1.0 / Scale
        assert (NewPos["x"].values < MaterialLimit + 1e-9).all()

    @pytest.mark.parametrize("GasRatio", [1.0, 2.0, 3.0, 0.5])
    def test_cell_x_dimension_scales(self, GasRatio):
        XPos = [0.1, 0.3, 0.5, 0.7, 0.9]
        Pos, Cell = MakeStructure(XPos)
        OrigX = Cell.iloc[0, 0]
        _, NewCell = Opp.AddVacuum(Pos, Cell, GasRatio=GasRatio)

        assert abs(NewCell.iloc[0, 0] - OrigX * (1.0 + GasRatio)) < 1e-9


class TestAddVacuumNoSplitStructurePreservesLayers:
    """When no layer straddles the boundary, the fix should be effectively a
    no-op on relative layer positions (all layers still land inside [0, 1/scale))."""

    def test_clean_structure_all_atoms_in_material_region(self):
        GasRatio = 2.0
        Scale = 1.0 + GasRatio
        # Layers cleanly away from x=0 and x=1
        XPos = [0.083, 0.250, 0.416, 0.583, 0.750, 0.916]
        Pos, Cell = MakeStructure(XPos)
        NewPos, _ = Opp.AddVacuum(Pos, Cell, GasRatio=GasRatio)

        MaterialLimit = 1.0 / Scale
        assert (NewPos["x"].values < MaterialLimit + 1e-9).all()


class TestAddVacuumPreservesChosenInterface:
    """Equal layer gaps must not change which decorated planes face the gas."""

    @pytest.mark.parametrize("Axis", ["x", "y", "z"])
    @pytest.mark.parametrize("GasRatio", [1.0, 2.0])
    def test_equal_gaps_keep_original_boundary(self, Axis, GasRatio):
        # Six ideal layers recovered from a centered 9a interface POSCAR.
        # Decimal conversion makes an interior gap larger by roundoff alone.
        LayerPositions = [
            0.08333333333333326, 0.24999999999999978,
            0.4166666666666665, 0.5833333333333335,
            0.7499999999999998, 0.9166666666666665,
        ]
        Pos, Cell = MakeStructure([0.5] * len(LayerPositions))
        Pos[Axis] = LayerPositions
        Pos["Element"] = ["Zr", "C", "N", "Zr", "N", "C"]
        OriginalPos, OriginalCell = Pos.copy(), Cell.copy()

        NewPos, NewCell = Opp.AddVacuum(Pos, Cell, GasRatio=GasRatio, Axis=Axis)

        # The same first and last decorated planes must remain the surfaces.
        assert NewPos[Axis].idxmin() == Pos[Axis].idxmin()
        assert NewPos[Axis].idxmax() == Pos[Axis].idxmax()
        np.testing.assert_allclose(
            NewPos[Axis].to_numpy() * (1.0 + GasRatio),
            LayerPositions, rtol=0, atol=1e-12,
        )
        CartesianShift = (
            NewPos[["x", "y", "z"]].to_numpy() @ NewCell.to_numpy()
            - Pos[["x", "y", "z"]].to_numpy() @ Cell.to_numpy()
        )
        np.testing.assert_allclose(CartesianShift, 0.0, rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(Pos, OriginalPos)
        pd.testing.assert_frame_equal(Cell, OriginalCell)
        assert NewPos["Element"].tolist() == Pos["Element"].tolist()

    def test_distinctly_larger_interior_gap_is_still_used(self):
        Pos, Cell = MakeStructure([0.05, 0.20, 0.60, 0.95])
        NewPos, _ = Opp.AddVacuum(Pos, Cell, GasRatio=2.0)

        # The midpoint of the unique largest gap is x=0.40.
        Expected = ((Pos["x"].to_numpy() - 0.40) % 1.0) / 3.0
        np.testing.assert_allclose(NewPos["x"], Expected, rtol=0, atol=1e-12)
