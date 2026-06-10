"""Tests for root-level simulation summary generation."""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from datetime import datetime, timedelta
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
from utils import StatusLog  # noqa: E402


@pytest.fixture(autouse=True)
def NoScheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to 'scheduler unknown' so no real squeue/qstat runs.

    Stuck tests inject an explicit LiveIds set via BuildSummary(..., LiveIds=...),
    which bypasses LiveJobIds entirely. The module-level scheduler cache is reset
    so auto-path tests never see a value cached by a prior test.
    """
    monkeypatch.setattr(Summary, "LiveJobIds", lambda: None)
    Summary._SchedulerCacheValue = None
    Summary._SchedulerCacheStamp = None


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


def WriteOutcar(WorkDir: Path, Step: int, StartStamp: str) -> None:
    """Write a minimal OUTCAR carrying VASP's 'executed on ... date ...' header.

    Step 0 writes the in-flight top-level OUTCAR; any other step writes the
    archived <step>/OUTCAR. StartStamp is 'YYYY.MM.DD  HH:MM:SS'.
    """
    if Step == 0:
        OutcarPath = WorkDir / "OUTCAR"
    else:
        StepDir = WorkDir / str(Step)
        StepDir.mkdir(exist_ok=True)
        OutcarPath = StepDir / "OUTCAR"
    OutcarPath.write_text(
        f" vasp.6.3.0\n executed on             LinuxIFC date {StartStamp}\n",
        encoding="utf-8",
    )


def SeedSimLog(WorkDir: Path, Rows: list[tuple]) -> None:
    """Write sim_log.tsv directly. Rows are (datetime|None, source, event, detail).

    None timestamp → now. Lets tests control submit times (for grace/queue logic).
    """
    Lines = []
    for Stamp, Source, Event, Detail in Rows:
        if Stamp is None:
            Stamp = datetime.now()
        Lines.append("\t".join((Stamp.isoformat(timespec="seconds"), Source, Event, Detail)))
    (WorkDir / StatusLog.SIM_LOG).write_text("\n".join(Lines) + "\n", encoding="utf-8")


def WriteSubmitTimes(WorkDir: Path, Lines: list[str]) -> None:
    """Seed sim_log.tsv 'submitted' events from '<step> <YYYY.MM.DD HH:MM:SS>' lines.

    Mirrors what volsearch_cont logs (the event timestamp is the submit time). A
    dummy job id is appended. Malformed inputs are written verbatim so the log
    reader skips them exactly as it would a corrupt line.
    """
    Out: list[str] = []
    for Line in Lines:
        Parts = Line.split()
        if len(Parts) < 2:
            Out.append(Line)  # junk -> < 3 TSV fields -> skipped by ReadEvents
            continue
        Step, Stamp = Parts[0], " ".join(Parts[1:])
        try:
            Iso = datetime.strptime(Stamp, "%Y.%m.%d %H:%M:%S").isoformat(timespec="seconds")
        except ValueError:
            Out.append(Line)  # unparseable timestamp -> skipped by ReadEvents
            continue
        Out.append("\t".join((Iso, "volsearch_cont", "submitted", "{} 100".format(Step))))
    (WorkDir / StatusLog.SIM_LOG).write_text("\n".join(Out) + "\n", encoding="utf-8")


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
    assert Row.WallTime != "-"


def TestFailedSimulationMarkerIsDetected(RootDir: Path) -> None:
    """sguschi_failed should produce FAILED status."""
    WorkDir = MakeWorkDir(RootDir, "873_2")
    (WorkDir / "sguschi_failed").write_text("", encoding="utf-8")

    Row = FindRow(Summary.BuildSummary(RootDir), "873_2")

    assert Row.Status == "FAILED"
    assert Row.Failed == "Y"
    assert Row.Detail == "sguschi_failed"


def TestDoneMarkerOverridesOlderFatalLog(RootDir: Path) -> None:
    """Completed trajectories should not stay failed because of recoverable helper FATAL text."""
    WorkDir = MakeWorkDir(RootDir, "873_4")
    (WorkDir / "1").mkdir()
    (WorkDir / "volsearch_is_done").write_text("", encoding="utf-8")
    (WorkDir.parent / "log.out").write_text(
        "FATAL PressureAvg.x: invalid pressure3.out\n"
        "WARNING volsearch_cont: PressureAvg.x failed or did not create pressure3_total.out; "
        "skipping pressure-controlled lattice update for this job.\n"
        "Volume search completed.\n",
        encoding="utf-8",
    )

    Row = FindRow(Summary.BuildSummary(RootDir), "873_4")

    assert Row.Status == "DONE"
    assert Row.Done == "Y"
    assert Row.Failed == "N"
    assert Row.Detail == "done"


def TestRestartedRunIgnoresPreviousRunFatalLog(RootDir: Path) -> None:
    """A FATAL line from a prior run must not mark a restarted, running sim FAILED.

    log.out is appended to (not truncated) across restarts, so a previous run's
    FATAL line lingers in the log. Failure is determined solely by the
    self-clearing sguschi_failed marker (cleared on restart), not by scanning the
    log for FATAL text, so a running sim with a 'started' event stays RUNNING.
    """
    WorkDir = MakeWorkDir(RootDir, "1073_2")
    (WorkDir / "1").mkdir()
    (WorkDir / "OUTCAR").write_text("running\n", encoding="utf-8")
    SeedSimLog(WorkDir, [(None, "SGUSCHI", "started", "")])

    # Stale FATAL from the previous, failed run still sitting in the log.
    (WorkDir.parent / "log.out").write_text(
        "FATAL volsearch_cont: AdjustBMIX failed.\n"
        "INFO volsearch_cont: continuing run\n",
        encoding="utf-8",
    )

    Row = FindRow(Summary.BuildSummary(RootDir), "1073_2")

    assert Row.Status == "RUNNING"
    assert Row.Failed == "N"


def TestNotStartedSimulationIsDetected(RootDir: Path) -> None:
    """An empty Dir_VolSearch should be NOT_STARTED."""
    MakeWorkDir(RootDir, "973_1")

    Row = FindRow(Summary.BuildSummary(RootDir), "973_1")

    assert Row.Status == "NOT_STARTED"
    assert Row.Detail == "no steps"


def TestRunningSimulationWithoutStepFoldersIsDetected(RootDir: Path) -> None:
    """Fresh activity before the first archived step should be RUNNING."""
    WorkDir = MakeWorkDir(RootDir, "973_2")
    (WorkDir / "OUTCAR").write_text("running\n", encoding="utf-8")

    Row = FindRow(Summary.BuildSummary(RootDir), "973_2")

    assert Row.Status == "RUNNING"
    assert Row.Detail == "recent activity"


def TestMissingExpectedSimulationIsDetected(RootDir: Path) -> None:
    """Expected simulations without Dir_VolSearch should be MISSING."""
    WriteExpected(RootDir, ["1073_1"])

    Row = FindRow(Summary.BuildSummary(RootDir), "1073_1")

    assert Row.Status == "MISSING"
    assert Row.Folders == "-"
    assert Row.Detail == "missing"


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
    assert Row.SimTime_ps == "0.1605"


def TestRateAnalysisO2AndRemovedTotalsAreParsed(RootDir: Path) -> None:
    """RateAnalysis.csv should provide final O2 added and removed molecule totals."""
    WorkDir = MakeWorkDir(RootDir, "1273_5")
    (WorkDir / "1").mkdir()
    (WorkDir / "RateAnalysis.csv").write_text(
        "Time (fs),O2 Count,Smoothed O2 Count,O2 Added,Gas Removed,Free Gas Fraction\n"
        "0,10,10,10,[],1\n"
        "80,8,8.5,11,\"[('C', 'O')]\",0.9\n"
        "160.5,7,7.5,12,\"[('C', 'O', 'O')]\",0.8\n",
        encoding="utf-8",
    )

    Row = FindRow(Summary.BuildSummary(RootDir), "1273_5")

    assert Row.TotalO2Added == "12"
    assert Row.MoleculesRemoved == "2"


def TestMissingGasRemovedColumnLeavesRemovedTotalBlank(RootDir: Path) -> None:
    """Missing Gas Removed should not prevent other RateAnalysis metrics."""
    WorkDir = MakeWorkDir(RootDir, "1273_6")
    (WorkDir / "1").mkdir()
    (WorkDir / "RateAnalysis.csv").write_text(
        "Time (fs),O2 Count,O2 Added\n0,10,10\n160.5,9,12\n",
        encoding="utf-8",
    )

    Row = FindRow(Summary.BuildSummary(RootDir), "1273_6")

    assert Row.RateRows == "2"
    assert Row.SimTime_ps == "0.1605"
    assert Row.TotalO2Added == "12"
    assert Row.MoleculesRemoved == "-"


def TestMalformedGasRemovedColumnDoesNotCrash(RootDir: Path) -> None:
    """Malformed Gas Removed rows should be skipped; valid rows are still counted.

    One valid row ([]) counts 0 molecules; one unparseable row is skipped.
    The result is '0?' — a partial count with '?' signalling incomplete parsing.
    """
    WorkDir = MakeWorkDir(RootDir, "1273_7")
    (WorkDir / "1").mkdir()
    (WorkDir / "RateAnalysis.csv").write_text(
        "Time (fs),O2 Count,O2 Added,Gas Removed\n0,10,10,[]\n160.5,9,12,not-a-list\n",
        encoding="utf-8",
    )

    Row = FindRow(Summary.BuildSummary(RootDir), "1273_7")

    assert Row.RateRows == "2"
    assert Row.TotalO2Added == "12"
    assert Row.MoleculesRemoved == "0?"


def TestSummaryOutputsAreWritten(RootDir: Path) -> None:
    """The CLI writer should produce fixed-width text and TSV outputs."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    (WorkDir / "volsearch_is_done").write_text("", encoding="utf-8")

    Rows = Summary.BuildSummary(RootDir)
    Summary.WriteOutputs(RootDir, Rows)

    Text = (RootDir / "SimulationSummary").read_text(encoding="utf-8")
    Tsv = (RootDir / "logs" / "SimulationSummary.tsv").read_text(encoding="utf-8")

    assert "Simulation" in Text
    assert "873_1" in Text
    assert (RootDir / "logs").is_dir()
    assert "Simulation\tStatus" in Tsv
    assert "Time_ps" in Text
    assert "SimTime_fs" not in Tsv
    assert "O2Added" in Text
    assert "GasRemoved" in Tsv


def TestSummaryOutputsUseCurrentSchema(RootDir: Path) -> None:
    """Summary outputs should use the current schema and output locations."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    (WorkDir / "1").mkdir()
    (WorkDir / "RateAnalysis.csv").write_text(
        "Time (fs),O2 Count,O2 Added,Gas Removed\n0,10,10,[]\n",
        encoding="utf-8",
    )

    Summary.WriteOutputs(RootDir, Summary.BuildSummary(RootDir))

    Text = (RootDir / "SimulationSummary").read_text(encoding="utf-8")
    Tsv = (RootDir / "logs" / "SimulationSummary.tsv").read_text(encoding="utf-8")
    assert "O2Added" in Text
    assert "GasRemoved" in Tsv
    # New columns present, retired "Latest" column gone.
    for Document in (Text, Tsv):
        assert "Queue" in Document
        assert "Age" in Document
        assert "Latest" not in Document


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


def TestMaxRuntimeReachedDetailIsShown(RootDir: Path) -> None:
    """volsearch_is_done + maxruntime_reached should produce DONE with 'max runtime' detail."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    (WorkDir / "volsearch_is_done").write_text("", encoding="utf-8")
    (WorkDir / "maxruntime_reached").write_text("", encoding="utf-8")

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.Status == "DONE"
    assert Row.Done == "Y"
    assert Row.Detail == "max runtime"


def TestExampleOxidationMasterStartsSummaryDaemon() -> None:
    """SGUSCHI.py (the orchestrator) should start the summary watcher daemon."""
    RootDir = Path(__file__).resolve().parents[2]
    # SimulationSummary daemon is now launched by SGUSCHI.py, not OxidationMaster directly.
    Text = (RootDir / "src" / "SGUSCHI.py").read_text(encoding="utf-8")

    assert "SimulationSummary" in Text
    assert "--watch-daemon" in Text
    assert "--oxparams" in Text
    assert not (RootDir / "src" / "utils" / "OxiMasterFailFast.sbatch").exists()
    # OxidationMaster is now a thin Slurm wrapper that calls SGUSCHI.py
    MasterText = (RootDir / "example" / "OxidationMaster").read_text(encoding="utf-8")
    assert "SGUSCHI.py" in MasterText


def TestExistingSummaryDocumentHeadersAreMigrated(RootDir: Path) -> None:
    """An existing summary document with old long headers is rewritten on refresh."""
    MakeWorkDir(RootDir, "873_1")

    TextPath = RootDir / Summary.SUMMARY_TXT
    TsvPath = RootDir / Summary.SUMMARY_TSV
    TsvPath.parent.mkdir(parents=True, exist_ok=True)
    # Seed both documents with the previous, long header names.
    OldHeaders = "Simulation\tStatus\tFolders\tLatest\tRateRows\tSimTime_ps\t" \
        "TotalO2Added\tMoleculesRemoved\tWallTime\tDone\tFailed\tDetail"
    TsvPath.write_text(OldHeaders + "\n", encoding="utf-8")
    TextPath.write_text(OldHeaders + "\n", encoding="utf-8")

    Summary.WriteSummaryOnce(RootDir, Quiet=True)

    TsvText = TsvPath.read_text(encoding="utf-8")
    TextText = TextPath.read_text(encoding="utf-8")
    for Document in (TsvText, TextText):
        assert "O2Added" in Document and "TotalO2Added" not in Document
        assert "GasRemoved" in Document and "MoleculesRemoved" not in Document
        assert "Time_ps" in Document and "SimTime_ps" not in Document


# --------------------------- Queue time ---------------------------------------


def TestQueueTimeIsComputedFromSubmitAndOutcarStart(RootDir: Path) -> None:
    """Queue time = OUTCAR start minus the recorded submission time."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    WriteOutcar(WorkDir, 1, "2024.03.11  09:21:45")
    WriteSubmitTimes(WorkDir, ["1 2024.03.11 09:21:00"])  # 45 s earlier

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.QueueTime == "45s"


def TestQueueTimeForInFlightTopLevelOutcar(RootDir: Path) -> None:
    """The currently-running step's OUTCAR (top-level) is also counted."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    WriteOutcar(WorkDir, 0, "2024.03.11  09:00:30")  # in-flight = step 1
    WriteSubmitTimes(WorkDir, ["1 2024.03.11 09:00:00"])

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.QueueTime == "30s"


def TestQueueTimeIsCumulativeAcrossSteps(RootDir: Path) -> None:
    """Per-step queue times sum into the reported total."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    WriteOutcar(WorkDir, 1, "2024.03.11  09:00:30")  # 30 s
    WriteOutcar(WorkDir, 2, "2024.03.11  10:00:45")  # 45 s
    WriteSubmitTimes(WorkDir, ["1 2024.03.11 09:00:00", "2 2024.03.11 10:00:00"])

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.QueueTime == "1m15s"  # 75 s


def TestQueueTimeResubmissionPicksLatestSubmitBeforeStart(RootDir: Path) -> None:
    """A restarted/resubmitted step pairs with the submission that launched it."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    WriteOutcar(WorkDir, 1, "2024.03.11  10:00:30")
    WriteSubmitTimes(WorkDir, [
        "1 2024.03.11 08:00:00",   # abandoned first submission
        "1 2024.03.11 10:00:00",   # resubmission that actually ran -> 30 s
    ])

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.QueueTime == "30s"


def TestQueueTimeIsZeroWhenStartEqualsSubmit(RootDir: Path) -> None:
    """A measured-but-zero wait reports '0s', distinct from '-' (no data)."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    WriteOutcar(WorkDir, 1, "2024.03.11  09:00:00")
    WriteSubmitTimes(WorkDir, ["1 2024.03.11 09:00:00"])

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.QueueTime == "0s"


def TestQueueTimeToleratesMalformedSubmitLines(RootDir: Path) -> None:
    """A corrupt .submit_times degrades to fewer pairings, never an error."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    WriteOutcar(WorkDir, 1, "2024.03.11  09:21:45")
    WriteSubmitTimes(WorkDir, [
        "garbage line with no timestamp",
        "1",                       # step only (failed `date`) -> skipped
        "x 2024.03.11 09:21:00",   # non-integer step -> skipped
        "1 not-a-date",            # unparseable timestamp -> skipped
        "1 2024.03.11 09:21:00",   # valid -> 45 s
    ])

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.QueueTime == "45s"


def TestQueueTimeIsDashWithoutSubmitTimes(RootDir: Path) -> None:
    """No .submit_times (e.g. orchestrator not yet redeployed) -> '-'."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    WriteOutcar(WorkDir, 1, "2024.03.11  09:21:45")

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.QueueTime == "-"


def TestQueueTimeIsDashWhenOutcarLacksStartLine(RootDir: Path) -> None:
    """An OUTCAR without the 'executed on ... date' line (e.g. some MLFF builds) -> '-'."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    Step = WorkDir / "1"
    Step.mkdir()
    (Step / "OUTCAR").write_text(" vasp.6.3.0\n no date header here\n", encoding="utf-8")
    WriteSubmitTimes(WorkDir, ["1 2024.03.11 09:21:00"])

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.QueueTime == "-"


# --------------------------- Last-update age ----------------------------------


def TestLastUpdateAgeReflectsRecentActivity(RootDir: Path) -> None:
    """A freshly written activity file yields a non-dash age."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    (WorkDir / "RateAnalysis.csv").write_text(
        "Time (fs),O2 Count\n0,10\n", encoding="utf-8"
    )

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.LastUpdate != "-"


def TestLastUpdateAgeIsDashWhenNoActivity(RootDir: Path) -> None:
    """An empty workdir has no activity timestamps -> '-'."""
    MakeWorkDir(RootDir, "873_1")

    Row = FindRow(Summary.BuildSummary(RootDir), "873_1")

    assert Row.LastUpdate == "-"


# --------------------------- Log lifecycle status -----------------------------


def TestKilledEventProducesKilled(RootDir: Path) -> None:
    """A 'killed' log event (SIGTERM) yields KILLED."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    SeedSimLog(WorkDir, [(None, "SGUSCHI", "started", ""),
                         (None, "SGUSCHI", "killed", "SIGTERM")])

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds=None), "873_1")

    assert Row.Status == "KILLED"
    assert Row.Failed == "Y"


def TestExitZeroEventProducesDone(RootDir: Path) -> None:
    """An 'exit 0' log event yields DONE even without volsearch_is_done."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    SeedSimLog(WorkDir, [(None, "SGUSCHI", "started", ""),
                         (None, "SGUSCHI", "exit", "0")])

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds=None), "873_1")

    assert Row.Status == "DONE"
    assert Row.Done == "Y"


def TestExitNonzeroEventProducesFailed(RootDir: Path) -> None:
    """A non-zero 'exit' log event (e.g. launch failure -1) yields FAILED."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    SeedSimLog(WorkDir, [(None, "SGUSCHI", "exit", "-1")])

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds=None), "873_1")

    assert Row.Status == "FAILED"
    assert Row.Failed == "Y"


def TestRestartStartedSupersedesEarlierExit(RootDir: Path) -> None:
    """A 'started' event after an earlier 'exit' (restart) is RUNNING, not FAILED."""
    WorkDir = MakeWorkDir(RootDir, "873_1")
    Old = datetime.now() - timedelta(hours=1)
    SeedSimLog(WorkDir, [(Old, "SGUSCHI", "exit", "1"),
                         (None, "SGUSCHI", "started", "")])

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds=None), "873_1")

    assert Row.Status == "RUNNING"
    assert Row.Failed == "N"


# --------------------------- Scheduler-aware stuck ----------------------------


def _SeedInFlightStep(WorkDir: Path, JobId: str, SubmitAgo: timedelta,
                      OutcarBody: str = "") -> None:
    """Set up a started sim mid-step-2: a step-1 folder, an in-flight OUTCAR, the
    .vasp_submitted_step sentinel, and started+submitted log events."""
    (WorkDir / "1").mkdir()
    (WorkDir / "OUTCAR").write_text(OutcarBody, encoding="utf-8")
    (WorkDir / ".vasp_submitted_step").write_text("2", encoding="utf-8")
    Stamp = datetime.now() - SubmitAgo
    SeedSimLog(WorkDir, [(Stamp, "SGUSCHI", "started", ""),
                         (Stamp, "volsearch_cont", "submitted", "2 {}".format(JobId))])


def TestStuckWhenJobAbsentFromScheduler(RootDir: Path) -> None:
    """Empty in-flight OUTCAR + job gone from the queue + past grace -> STUCK."""
    WorkDir = MakeWorkDir(RootDir, "1273_2")
    _SeedInFlightStep(WorkDir, "555", timedelta(minutes=10))

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds={"999"}), "1273_2")

    assert Row.Status == "STUCK"
    assert Row.Detail == "Check .out file and resubmit"
    assert Row.Failed == "N"


def TestRunningWhenJobPresentInScheduler(RootDir: Path) -> None:
    """Same state but the job is still in the queue -> RUNNING (queued)."""
    WorkDir = MakeWorkDir(RootDir, "1273_2")
    _SeedInFlightStep(WorkDir, "555", timedelta(minutes=10))

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds={"555"}), "1273_2")

    assert Row.Status == "RUNNING"
    assert Row.Detail == "queued"


def TestNotStuckWhenSchedulerUnknown(RootDir: Path) -> None:
    """Liveness unknown (off-cluster / query failed) never flags STUCK."""
    WorkDir = MakeWorkDir(RootDir, "1273_2")
    _SeedInFlightStep(WorkDir, "555", timedelta(minutes=10))

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds=None), "1273_2")

    assert Row.Status == "RUNNING"


def TestNotStuckWithinSubmitGrace(RootDir: Path) -> None:
    """A just-submitted job absent from the queue is within grace -> not STUCK."""
    WorkDir = MakeWorkDir(RootDir, "1273_2")
    _SeedInFlightStep(WorkDir, "555", timedelta(seconds=5))

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds={"999"}), "1273_2")

    assert Row.Status == "RUNNING"


def TestNotStuckWhenOutcarCompleted(RootDir: Path) -> None:
    """If OUTCAR already hit 'Total CPU', the step completed -> not STUCK."""
    WorkDir = MakeWorkDir(RootDir, "1273_2")
    _SeedInFlightStep(WorkDir, "555", timedelta(minutes=10),
                      OutcarBody=" Total CPU time used (sec):    1.0\n")

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds={"999"}), "1273_2")

    assert Row.Status != "STUCK"


def _SetOutcarAge(WorkDir: Path, Age: timedelta) -> None:
    """Backdate the OUTCAR mtime so the warrant gate sees it as stale."""
    Past = (datetime.now() - Age).timestamp()
    os.utime(WorkDir / "OUTCAR", (Past, Past))


def TestNotStuckWhenOutcarProgressing(RootDir: Path) -> None:
    """Job absent from the queue but OUTCAR was written recently -> not STUCK.

    A healthy run has an incomplete OUTCAR with a fresh mtime; the warrant gate
    skips the scheduler entirely, so an absent job id must not flag STUCK.
    """
    WorkDir = MakeWorkDir(RootDir, "1273_2")
    _SeedInFlightStep(WorkDir, "555", timedelta(minutes=10), OutcarBody="POSITION\n 0 0 0\n")

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds={"999"}), "1273_2")

    assert Row.Status == "RUNNING"


def TestStuckWhenOutcarFrozenPastThreshold(RootDir: Path) -> None:
    """Non-empty OUTCAR unmodified past the stale threshold + job gone -> STUCK."""
    WorkDir = MakeWorkDir(RootDir, "1273_2")
    _SeedInFlightStep(WorkDir, "555", timedelta(minutes=20), OutcarBody="POSITION\n 0 0 0\n")
    _SetOutcarAge(WorkDir, timedelta(minutes=20))

    Row = FindRow(Summary.BuildSummary(RootDir, LiveIds={"999"}), "1273_2")

    assert Row.Status == "STUCK"


def TestSchedulerNotQueriedForHealthySim(RootDir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scheduler is consulted only when a sim is warranted (stale OUTCAR)."""
    Calls = {"n": 0}

    def Counting() -> set:
        Calls["n"] += 1
        return {"999"}  # job 555 absent

    monkeypatch.setattr(Summary, "LiveJobIds", Counting)

    WorkDir = MakeWorkDir(RootDir, "1273_2")
    _SeedInFlightStep(WorkDir, "555", timedelta(minutes=20), OutcarBody="POSITION\n 0 0 0\n")

    # Healthy: fresh OUTCAR -> no scheduler query, not stuck.
    Row = FindRow(Summary.BuildSummary(RootDir), "1273_2")
    assert Row.Status == "RUNNING"
    assert Calls["n"] == 0

    # Stale: OUTCAR frozen past the threshold -> exactly one real query -> STUCK.
    _SetOutcarAge(WorkDir, timedelta(minutes=20))
    Row = FindRow(Summary.BuildSummary(RootDir), "1273_2")
    assert Row.Status == "STUCK"
    assert Calls["n"] == 1
