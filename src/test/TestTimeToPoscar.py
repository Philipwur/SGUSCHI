"""Tests for the TimeToPOSCAR utility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from DevArea.UsefulFunctions import TimeToPOSCAR as Ttp
from workflow import VaspIO as Vio


@pytest.fixture(name="TmpPath")
def FixtureTmpPath(tmp_path: Path) -> Path:
    """Expose pytest tmp_path as PascalCase fixture name."""
    return tmp_path


def MakeSimpleTrajectory(
    TmpPath: Path,
    FileName: str = "simple.xyz",
    FrameTimesFs: tuple[float, float, float] = (0.0, 1000.0, 2000.0),
) -> Path:
    """Create a compact extended XYZ trajectory with header times in fs."""
    XyzPath = TmpPath / FileName
    Lines = [
        "3",
        f'Frame 1 Lattice="10 0 0 0 10 0 0 0 10" Step=1 Time={FrameTimesFs[0]}',
        "O 0.0 0.0 0.0",
        "Zr 5.0 5.0 5.0",
        "C 8.0 8.0 8.0",
        "3",
        f'Frame 2 Lattice="10 0 0 0 10 0 0 0 10" Step=2 Time={FrameTimesFs[1]}',
        "O 1.0 0.0 0.0",
        "Zr 5.5 5.0 5.0",
        "C 8.5 8.0 8.0",
        "3",
        f'Frame 3 Lattice="10 0 0 0 10 0 0 0 10" Step=3 Time={FrameTimesFs[2]}',
        "O 2.0 0.0 0.0",
        "Zr 6.0 5.0 5.0",
        "C 9.0 8.0 8.0",
    ]
    XyzPath.write_text("\n".join(Lines) + "\n", encoding="utf-8")
    return XyzPath


def FirstAppearanceOrder(Elements: list[str]) -> list[str]:
    """Return unique species order from first appearance."""
    return list(dict.fromkeys(Elements))


def TestSelectClosestFrameExactMatch(TmpPath: Path) -> None:
    """Exact target times should return the matching frame."""
    XyzPath = MakeSimpleTrajectory(TmpPath)

    Frame = Ttp._SelectClosestFrame(XyzPath, 1.0)

    assert Frame is not None
    assert Frame.Step == 2
    assert Frame.TimePs == 1.0
    assert Frame.Elements == ["O", "Zr", "C"]
    assert np.allclose(Frame.CartesianCoordinates[0], [1.0, 0.0, 0.0])


def TestSelectClosestFrameNearestMatch(TmpPath: Path) -> None:
    """Inexact targets should return the closest-time frame."""
    XyzPath = MakeSimpleTrajectory(TmpPath)

    Frame = Ttp._SelectClosestFrame(XyzPath, 1.4)

    assert Frame is not None
    assert Frame.Step == 2
    assert Frame.TimePs == 1.0


def TestExportCreatesDefaultOutputLayout(TmpPath: Path) -> None:
    """ExportClosestSnapshots should create the default sibling output layout."""
    InputDir = TmpPath / "InputTraj"
    InputDir.mkdir()
    MakeSimpleTrajectory(InputDir, "traj_a.xyz")
    (InputDir / "note.txt").write_text("ignore me\n", encoding="utf-8")

    OutputBaseDir = TmpPath / "output"
    Ttp.ExportClosestSnapshots(InputDir=InputDir, TargetTimePs=1.0, OutputBaseDir=OutputBaseDir)

    OutputRoot = OutputBaseDir / "InputTraj_t1ps"
    PoscarPath = OutputRoot / "traj_a" / "POSCAR"
    assert PoscarPath.exists()

    Position, _ = Vio.ReadPoscar(FileName=PoscarPath)
    assert list(Position["Element"]) == ["O", "Zr", "C"]


def TestResolveOutputRootDefaultsToScriptOutputFolder() -> None:
    """Default output roots should live under the script-local output directory."""
    InputDir = Path(r"C:\Temp\SomeDataset")
    OutputRoot = Ttp._ResolveOutputRoot(InputDir, 1.0, None)

    assert OutputRoot == Path(Ttp.__file__).resolve().parent / "output" / "SomeDataset_t1ps"


def TestExportWritesBatchReportWithTimeRange(TmpPath: Path) -> None:
    """The batch report should summarize matched time ranges across exported POSCARs."""
    InputDir = TmpPath / "Reported"
    InputDir.mkdir()
    MakeSimpleTrajectory(InputDir, "traj_a.xyz", FrameTimesFs=(0.0, 1000.0, 2000.0))
    MakeSimpleTrajectory(InputDir, "traj_b.xyz", FrameTimesFs=(0.0, 3000.0, 6000.0))

    OutputBaseDir = TmpPath / "output"
    Ttp.ExportClosestSnapshots(InputDir=InputDir, TargetTimePs=2.4, OutputBaseDir=OutputBaseDir)

    ReportPath = OutputBaseDir / "Reported_t2.4ps" / "TimeToPOSCAR_Report.txt"
    assert ReportPath.exists()

    ReportText = ReportPath.read_text(encoding="utf-8")
    assert "Requested target time (ps): 2.400000" in ReportText
    assert "XYZ header time unit: fs (converted to ps for matching and reporting)" in ReportText
    assert "POSCAR files written: 2" in ReportText
    assert "Files skipped: 0" in ReportText
    assert "Matched time range (ps): 2.000000 to 3.000000" in ReportText
    assert "Match delta range (ps): 0.400000 to 0.600000" in ReportText
    assert "- traj_a.xyz: step=3, matched_time_ps=2.000000, delta_ps=0.400000" in ReportText
    assert "- traj_b.xyz: step=2, matched_time_ps=3.000000, delta_ps=0.600000" in ReportText


def TestExportSupportsLegacyFrameHeaders(TmpPath: Path) -> None:
    """Legacy 'Frame ... Lattice=... Time=...' headers should export correctly."""
    InputDir = TmpPath / "Legacy"
    InputDir.mkdir()
    FixturePath = Path(__file__).resolve().parent / "fixtures" / "1273_3TimeAddedFrames.xyz"
    TargetPath = InputDir / FixturePath.name
    TargetPath.write_text(FixturePath.read_text(encoding="utf-8"), encoding="utf-8")

    OutputBaseDir = TmpPath / "output"
    Ttp.ExportClosestSnapshots(InputDir=InputDir, TargetTimePs=0.0, OutputBaseDir=OutputBaseDir)

    PoscarPath = OutputBaseDir / "Legacy_t0ps" / "1273_3TimeAddedFrames" / "POSCAR"
    assert PoscarPath.exists()

    Position, _ = Vio.ReadPoscar(FileName=PoscarPath)
    assert FirstAppearanceOrder(list(Position["Element"])) == ["O", "Zr", "C"]
    assert Position.shape[0] == 116

    Title = PoscarPath.read_text(encoding="utf-8").splitlines()[0]
    assert "target 0.000000 ps" in Title
    assert "matched 0.000000 ps" in Title


def TestExportSupportsPropertiesHeaders(TmpPath: Path) -> None:
    """Headers beginning with Lattice=... Properties=... should export correctly."""
    InputDir = TmpPath / "Properties"
    InputDir.mkdir()
    FixturePath = Path(__file__).resolve().parent / "fixtures" / "1273_3ZrCnFrames.xyz"
    TargetPath = InputDir / FixturePath.name
    TargetPath.write_text(FixturePath.read_text(encoding="utf-8"), encoding="utf-8")

    OutputBaseDir = TmpPath / "output"
    Ttp.ExportClosestSnapshots(InputDir=InputDir, TargetTimePs=0.001, OutputBaseDir=OutputBaseDir)

    PoscarPath = OutputBaseDir / "Properties_t0.001ps" / "1273_3ZrCnFrames" / "POSCAR"
    assert PoscarPath.exists()

    Position, _ = Vio.ReadPoscar(FileName=PoscarPath)
    assert FirstAppearanceOrder(list(Position["Element"])) == ["C", "N", "Zr", "O"]
    assert Position.shape[0] == 116


def TestExplicitElementOrderReordersSpeciesAndPreservesWithinSpeciesOrder(TmpPath: Path) -> None:
    """Explicit element ordering should change species blocks without reordering atoms inside each block."""
    InputDir = TmpPath / "Ordered"
    InputDir.mkdir()
    FixturePath = Path(__file__).resolve().parent / "fixtures" / "1273_3ZrCnFrames.xyz"
    TargetPath = InputDir / FixturePath.name
    TargetPath.write_text(FixturePath.read_text(encoding="utf-8"), encoding="utf-8")

    ElementOrder = ["Zr", "O"]
    OutputBaseDir = TmpPath / "output"
    Ttp.ExportClosestSnapshots(
        InputDir=InputDir,
        TargetTimePs=0.001,
        ElementOrder=ElementOrder,
        OutputBaseDir=OutputBaseDir,
    )

    PoscarPath = OutputBaseDir / "Ordered_t0.001ps" / "1273_3ZrCnFrames" / "POSCAR"
    PositionOut, _ = Vio.ReadPoscar(FileName=PoscarPath)

    Frame = Ttp._SelectClosestFrame(TargetPath, 0.001)
    assert Frame is not None
    PositionOriginal = Ttp._BuildPositionFrame(Frame)

    assert FirstAppearanceOrder(list(PositionOut["Element"])) == ["Zr", "O", "C", "N"]

    for Element in ["Zr", "O", "C", "N"]:
        OriginalSubset = PositionOriginal.loc[PositionOriginal["Element"] == Element, ["x", "y", "z"]].to_numpy()
        OutputSubset = PositionOut.loc[PositionOut["Element"] == Element, ["x", "y", "z"]].to_numpy()
        assert np.allclose(OutputSubset, OriginalSubset)


def TestMalformedAndTimeMissingFilesAreSkipped(TmpPath: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Malformed or untimed files should be reported and skipped without aborting the batch."""
    InputDir = TmpPath / "Mixed"
    InputDir.mkdir()

    MakeSimpleTrajectory(InputDir, "valid.xyz")
    (InputDir / "untimed.xyz").write_text(
        "\n".join(
            [
                "2",
                'Frame 1 Lattice="10 0 0 0 10 0 0 0 10" Step=1',
                "O 0.0 0.0 0.0",
                "Zr 5.0 5.0 5.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (InputDir / "malformed.xyz").write_text(
        "\n".join(
            [
                "2",
                'Frame 1 Lattice="10 0 0 0 10 0 0 0 10" Step=1 Time=0.0',
                "O 0.0 0.0 0.0",
                "Zr only_two_columns",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    OutputBaseDir = TmpPath / "output"
    Ttp.ExportClosestSnapshots(InputDir=InputDir, TargetTimePs=1.0, OutputBaseDir=OutputBaseDir)

    OutputRoot = OutputBaseDir / "Mixed_t1ps"
    assert (OutputRoot / "valid" / "POSCAR").exists()
    assert not (OutputRoot / "untimed" / "POSCAR").exists()
    assert not (OutputRoot / "malformed" / "POSCAR").exists()
    assert (OutputRoot / "TimeToPOSCAR_Report.txt").exists()

    Stdout = capsys.readouterr().out
    assert "valid.xyz: step=2" in Stdout
    assert "untimed.xyz: skipped" in Stdout
    assert "malformed.xyz: skipped" in Stdout

    ReportText = (OutputRoot / "TimeToPOSCAR_Report.txt").read_text(encoding="utf-8")
    assert "Files skipped: 2" in ReportText
    assert "- untimed.xyz: no frame with parseable Time and Lattice found" in ReportText
    assert "- malformed.xyz: malformed atom line: 'Zr only_two_columns'" in ReportText
