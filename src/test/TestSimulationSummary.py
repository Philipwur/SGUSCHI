"""Tests for root-level simulation summary generation."""

from __future__ import annotations

import os
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

from utils import SimulationSummary as Summary  # noqa: E402


@pytest.fixture(name="RootDir")
def FixtureRootDir() -> Path:
    """Create a repo-local temp root without using pytest's tmp_path fixture."""
    Root = Path.cwd() / ".test_tmp_simsummary" / uuid.uuid4().hex
    Root.mkdir(parents=True)
    try:
        yield Root
    finally:
        shutil.rmtree(Root, ignore_errors=True)


def MakeWorkDir(Root: Path, Label: str) -> Path:
    """Create a simulation Dir_VolSearch folder."""
    WorkDir = Root / Label / "Dir_VolSearch"
    WorkDir.mkdir(parents=True)
    return WorkDir


def WriteExpected(Root: Path, Labels: list[str]) -> None:
    """Write the optional expected simulation list."""
    MetaDir = Root / ".simulation_summary"
    MetaDir.mkdir()
    Lines = ["Simulation\tWorkDir"]
    for Label in Labels:
        Lines.append(f"{Label}\t{Label}/Dir_VolSearch")
    (MetaDir / "expected.tsv").write_text("\n".join(Lines) + "\n", encoding="utf-8")


def WriteOxParams(Root: Path, Temperatures: list[int], NSims: int) -> Path:
    """Write a minimal OxParams file."""
    OxParams = Root / "OxParams"
    OxParams.write_text(
        f"Temperatures = {Temperatures!r}\nNSims = {NSims!r}\n",
        encoding="utf-8",
    )
    return OxParams


def FindRow(Rows: list[Summary.SimulationRow], Label: str) -> Summary.SimulationRow:
    """Find one row by label."""
    for Row in Rows:
        if Row.Simulation == Label:
            return Row
    raise AssertionError(f"Missing row {Label}")


def Touch(PathFile: Path, Timestamp: int) -> None:
    """Set a deterministic file timestamp."""
    os.utime(PathFile, (Timestamp, Timestamp))


def TestDoneSimulationIsDetected(RootDir: Path) -> None:
    """volsearch_is_done should produce DONE status."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    Step = WorkDir / "1"
    Step.mkdir()
    Done = WorkDir / "volsearch_is_done"
    Done.write_text("", encoding="utf-8")
    Touch(Step, 100)
    Touch(Done, 200)

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.Status == "DONE"
    assert Row.Done == "Y"
    assert Row.Folders == "1"
    assert Row.Latest == "1"
    assert Row.WallTime != "-"


def TestFailedSimulationMarkerIsDetected(RootDir: Path) -> None:
    """sguschi_failed should produce FAILED status."""
    WorkDir = MakeWorkDir(RootDir, "873_2")
    (WorkDir / "sguschi_failed").write_text("", encoding="utf-8")

    Row = FindRow(Summary.BuildSummary(RootDir), "873_2")

    assert Row.Status == "FAILED"
    assert Row.Failed == "Y"
    assert Row.Detail == "sguschi_failed"


def TestFailedSimulationFatalLogIsDetected(RootDir: Path) -> None:
    """A FATAL line in log.out should produce FAILED status."""
    WorkDir = MakeWorkDir(RootDir, "873_3")
    (WorkDir.parent / "log.out").write_text(
        "normal line\nFATAL volsearch_cont: AdjustBMIX failed.\n",
        encoding="utf-8",
    )

    Row = FindRow(Summary.BuildSummary(RootDir), "873_3")

    assert Row.Status == "FAILED"
    assert "AdjustBMIX failed" in Row.Detail


def TestNotStartedSimulationIsDetected(RootDir: Path) -> None:
    """An empty Dir_VolSearch should be NOT_STARTED."""
    MakeWorkDir(RootDir, "973_1")

    Row = FindRow(Summary.BuildSummary(RootDir), "973_1")

    assert Row.Status == "NOT_STARTED"
    assert Row.Detail == "no step folders"


def TestRunningSimulationWithoutStepFoldersIsDetected(RootDir: Path) -> None:
    """Fresh activity before the first archived step should be RUNNING."""
    WorkDir = MakeWorkDir(RootDir, "973_2")
    (WorkDir / "OUTCAR").write_text("running\n", encoding="utf-8")

    Row = FindRow(Summary.BuildSummary(RootDir), "973_2")

    assert Row.Status == "RUNNING"
    assert Row.Detail == "recent file activity"


def TestMissingExpectedSimulationIsDetected(RootDir: Path) -> None:
    """Expected simulations without Dir_VolSearch should be MISSING."""
    WriteExpected(RootDir, ["1073_1"])

    Row = FindRow(Summary.BuildSummary(RootDir), "1073_1")

    assert Row.Status == "MISSING"
    assert Row.Folders == "-"
    assert Row.Detail == "Dir_VolSearch missing"


def TestOxParamsExpectedSimulationsAreDetected(RootDir: Path) -> None:
    """OxParams should infer expected simulation folders."""
    OxParams = WriteOxParams(RootDir, [873, 973], 2)

    Rows = Summary.BuildSummary(RootDir, OxParams)

    assert [Row.Simulation for Row in Rows] == ["873_1", "873_2", "973_1", "973_2"]
    assert FindRow(Rows, "973_2").Status == "MISSING"


def TestRateAnalysisRowsAndSimTimeAreParsed(RootDir: Path) -> None:
    """RateAnalysis.csv should provide row count and latest simulated time."""
    WorkDir = MakeWorkDir(RootDir, "1273_4")
    (WorkDir / "1").mkdir()
    (WorkDir / "RateAnalysis.csv").write_text(
        "Time (fs),O2 Count\n0,10\n160.5,9\n",
        encoding="utf-8",
    )

    Row = FindRow(Summary.BuildSummary(RootDir), "1273_4")

    assert Row.RateRows == "2"
    assert Row.SimTime_fs == "160.5"


def TestSummaryOutputsAreWritten(RootDir: Path) -> None:
    """The CLI writer should produce fixed-width text and TSV outputs."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    (WorkDir / "volsearch_is_done").write_text("", encoding="utf-8")

    Rows = Summary.BuildSummary(RootDir)
    Summary.WriteOutputs(RootDir, Rows)

    Text = (RootDir / "SimulationSummary.txt").read_text(encoding="utf-8")
    Tsv = (RootDir / "SimulationSummary.tsv").read_text(encoding="utf-8")

    assert "Simulation" in Text
    assert "873_1" in Text
    assert "Simulation\tStatus" in Tsv


def TestCliParsingAcceptsWatchOptions() -> None:
    """The CLI should expose one-shot, watch, and daemon options."""
    Watch = Summary.ParseArgs(
        [".", "--watch", "--interval", "12.5", "--oxparams", "OxParams", "--quiet"]
    )
    Daemon = Summary.ParseArgs([".", "--watch-daemon", "--interval", "60"])

    assert Watch.watch is True
    assert Watch.interval == 12.5
    assert Watch.oxparams == "OxParams"
    assert Watch.quiet is True
    assert Daemon.watch_daemon is True


def TestWatchDaemonPassesParentPid(monkeypatch: pytest.MonkeyPatch, RootDir: Path) -> None:
    """Daemon startup should pass the original shell PID to the watcher."""
    Captured = {}

    class FakeProcess:
        pid = 999

    def FakePopen(Command: list[str], **Kwargs: object) -> FakeProcess:
        Captured["Command"] = Command
        Captured["Kwargs"] = Kwargs
        return FakeProcess()

    monkeypatch.setattr(Summary.os, "getppid", lambda: 4321)
    monkeypatch.setattr(Summary.subprocess, "Popen", FakePopen)

    ExitCode = Summary.StartWatchDaemon(RootDir, RootDir / "OxParams", 60.0, True)
    Command = Captured["Command"]

    assert ExitCode == 0
    assert "--watch" in Command
    assert Command[Command.index("--parent-pid") + 1] == "4321"
    assert "--quiet" in Command


def TestExampleOxidationMasterStartsSummaryDaemon() -> None:
    """The example master should use the one-line summary watcher."""
    RootDir = Path(__file__).resolve().parents[2]
    Text = (RootDir / "example" / "OxidationMaster").read_text(
        encoding="utf-8"
    )

    assert "SimulationSummary.py" in Text
    assert "--watch-daemon" in Text
    assert "--oxparams OxParams" in Text
    assert "expected.tsv" not in Text
    assert not (RootDir / "src" / "utils" / "OxiMasterFailFast.sbatch").exists()
