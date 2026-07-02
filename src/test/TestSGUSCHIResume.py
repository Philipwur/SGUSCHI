"""Tests for SGUSCHI.ReopenTimeCappedSimulations — automatic MaxRuntime resume."""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest


def EnsureSrcOnPath() -> None:
    """Insert the project src directory into sys.path if missing."""
    RootDir = Path(__file__).resolve().parents[2]
    SrcDir = RootDir / "src"
    if str(SrcDir) not in sys.path:
        sys.path.insert(0, str(SrcDir))


EnsureSrcOnPath()

import SGUSCHI  # noqa: E402


@pytest.fixture(name="WorkDir")
def FixtureWorkDir() -> Path:
    """Create a repo-local temp workspace, removed after the test."""
    Root = Path.cwd() / ".test_tmp_sguschi_resume" / uuid.uuid4().hex
    Root.mkdir(parents=True)
    try:
        yield Root
    finally:
        shutil.rmtree(Root, ignore_errors=True)


def MakeParams(MaxRuntime: str | None = "500") -> dict:
    """Minimal OxParams dict for a single 873_1 simulation."""
    Params = {"Temperatures": "[873]", "NSims": "1"}
    if MaxRuntime is not None:
        Params["MaxRuntime"] = MaxRuntime
    return Params


def MakeSim(
    WorkDir: Path,
    RuntimeFs: float | None,
    VolsearchDone: bool = True,
    MaxRuntimeReached: bool = True,
) -> Path:
    """Create the 873_1 Dir_VolSearch with the requested markers and runtime.

    RuntimeFs=None omits RateAnalysis.csv entirely (runtime unreadable).
    """
    Vsd = WorkDir / "873_1" / "Dir_VolSearch"
    Vsd.mkdir(parents=True)
    if VolsearchDone:
        (Vsd / "volsearch_is_done").write_text("", encoding="utf-8")
    if MaxRuntimeReached:
        (Vsd / "maxruntime_reached").write_text("", encoding="utf-8")
    if RuntimeFs is not None:
        (Vsd / "RateAnalysis.csv").write_text(
            "Time (fs),O2 Count,Smoothed O2 Count,O2 Added,Gas Removed,Free Gas Fraction\n"
            "0,10,10.0,10,[],1.0\n"
            "{},9,9.5,10,[],0.98\n".format(RuntimeFs),
            encoding="utf-8",
        )
    return Vsd


def TestReopensWhenRuntimeBelowRaisedCap(WorkDir: Path) -> None:
    """runtime 200 ps < MaxRuntime 500 ps → both markers cleared, label returned."""
    Vsd = MakeSim(WorkDir, RuntimeFs=200_000)  # 200 ps
    Reopened = SGUSCHI.ReopenTimeCappedSimulations(WorkDir, MakeParams("500"))
    assert Reopened == ["873_1"]
    assert not (Vsd / "volsearch_is_done").exists()
    assert not (Vsd / "maxruntime_reached").exists()


def TestKeepsDoneWhenRuntimeMeetsCap(WorkDir: Path) -> None:
    """runtime 600 ps >= MaxRuntime 500 ps → markers stay, nothing reopened."""
    Vsd = MakeSim(WorkDir, RuntimeFs=600_000)  # 600 ps
    Reopened = SGUSCHI.ReopenTimeCappedSimulations(WorkDir, MakeParams("500"))
    assert Reopened == []
    assert (Vsd / "volsearch_is_done").exists()
    assert (Vsd / "maxruntime_reached").exists()


def TestNaturalConvergenceNeverTouched(WorkDir: Path) -> None:
    """volsearch_is_done without maxruntime_reached is a convergence stop: untouched."""
    Vsd = MakeSim(WorkDir, RuntimeFs=10_000, MaxRuntimeReached=False)
    Reopened = SGUSCHI.ReopenTimeCappedSimulations(WorkDir, MakeParams("500"))
    assert Reopened == []
    assert (Vsd / "volsearch_is_done").exists()


def TestDryRunReportsButDoesNotDelete(WorkDir: Path) -> None:
    """DryRun returns the label but leaves the markers in place."""
    Vsd = MakeSim(WorkDir, RuntimeFs=200_000)
    Reopened = SGUSCHI.ReopenTimeCappedSimulations(WorkDir, MakeParams("500"), DryRun=True)
    assert Reopened == ["873_1"]
    assert (Vsd / "volsearch_is_done").exists()
    assert (Vsd / "maxruntime_reached").exists()


def TestNoMaxRuntimeIsNoOp(WorkDir: Path) -> None:
    """With no MaxRuntime in OxParams the check does nothing (manual path preserved)."""
    Vsd = MakeSim(WorkDir, RuntimeFs=200_000)
    Reopened = SGUSCHI.ReopenTimeCappedSimulations(WorkDir, MakeParams(MaxRuntime=None))
    assert Reopened == []
    assert (Vsd / "volsearch_is_done").exists()
    assert (Vsd / "maxruntime_reached").exists()


def TestUnreadableRuntimeLeftAsDone(WorkDir: Path) -> None:
    """maxruntime_reached present but no RateAnalysis → stay done, do not guess."""
    Vsd = MakeSim(WorkDir, RuntimeFs=None)
    Reopened = SGUSCHI.ReopenTimeCappedSimulations(WorkDir, MakeParams("500"))
    assert Reopened == []
    assert (Vsd / "volsearch_is_done").exists()
    assert (Vsd / "maxruntime_reached").exists()


def TestNonNumericMaxRuntimeIsNoOp(WorkDir: Path) -> None:
    """A malformed MaxRuntime is skipped with a warning, not a crash."""
    Vsd = MakeSim(WorkDir, RuntimeFs=200_000)
    Reopened = SGUSCHI.ReopenTimeCappedSimulations(WorkDir, MakeParams("soon"))
    assert Reopened == []
    assert (Vsd / "maxruntime_reached").exists()
