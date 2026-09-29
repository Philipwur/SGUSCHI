"""Grouped jobs: real workspace preparation and isolated, mocked scheduling."""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

import SGUSCHI as Controller
from utils import SimulationSummary as Summary
from utils import StatusLog
from utils.FolderUtils import TrajectoryRoot
from utils.JobSpecs import LoadJobSpecs, INPUT_RECORD
from workflow import VaspIO as Vio

RUN_ORCHESTRATION = Controller.RunOrchestration


def WriteWorkspace(Root, Element="C", GasRatio=1, O2Tol=0.1):
    Root.mkdir(parents=True, exist_ok=True)
    (Root / "OxParams").write_text(
        "Temperatures = [873]\nNSims = 1\nGasRatio = {}\nInitO2Count = 1\n"
        "AtomicRadiusTol = 1.5\nO2Tol = {}\nOSmoothing = 0.001\nMaxRuntime = 100\n".format(GasRatio, O2Tol),
        encoding="utf-8",
    )
    (Root / "POSCAR").write_text(
        "Synthetic structure\n1.0\n8 0 0\n0 8 0\n0 0 8\nZr {}\n2 2\nDirect\n"
        "0 0 0\n0.5 0.5 0.5\n0.25 0.25 0.25\n0.75 0.75 0.75\n".format(Element),
        encoding="utf-8",
    )
    (Root / "POTCAR").write_text("synthetic Zr {} O\n".format(Element), encoding="utf-8")
    (Root / "INCAR").write_text("TEBEG = 0\nTEEND = 0\nNSW = 80\n", encoding="utf-8")
    (Root / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (Root / "job.in").write_text("temp = 0\nnavg = 3\nvaspcmd = sbatch\n", encoding="utf-8")
    (Root / "jobsub").write_text("#!/bin/bash\n#SBATCH --job-name=original\n", encoding="utf-8")
    (Root / "CovalentRadii").write_text("Zr = 1.45\nC = 0.76\nN = 0.71\nO = 0.66\n", encoding="utf-8")


@pytest.fixture
def Campaign(tmp_path):
    (tmp_path / "OxParams").write_text('JobSpecs = ["jobs/low", "jobs/high"]\n', encoding="utf-8")
    WriteWorkspace(tmp_path / "jobs" / "low")
    WriteWorkspace(tmp_path / "jobs" / "high", Element="N", GasRatio=2, O2Tol=1.0)
    return tmp_path


@pytest.fixture(autouse=True)
def NoRealScheduler(monkeypatch):
    def Forbidden(*Args, **Kwargs):
        pytest.fail("Unexpected scheduler or orchestration call")
    monkeypatch.setattr(Controller.subprocess, "run", Forbidden)
    monkeypatch.setattr(Controller, "RunOrchestration", Forbidden)


def Invoke(monkeypatch, Root, Prepare=False, Dry=False):
    monkeypatch.setattr(Controller, "ParseArgs", lambda: SimpleNamespace(
        workdir=str(Root), prepare_only=Prepare, dry_run=Dry,
    ))
    return Controller.main()


def Snapshot(Root):
    return {P.relative_to(Root).as_posix(): P.read_bytes() for P in Root.rglob("*") if P.is_file()}


@pytest.mark.parametrize("Setting", ["", "JobSpecs = []\n"])
def TestLegacyIsDefault(tmp_path, monkeypatch, Setting):
    WriteWorkspace(tmp_path)
    with (tmp_path / "OxParams").open("a", encoding="utf-8") as File:
        File.write(Setting)
    assert LoadJobSpecs(tmp_path)[0].Id == ""
    assert Invoke(monkeypatch, tmp_path, Prepare=True) == 0
    assert (tmp_path / "873_1" / "Dir_VolSearch" / "POSCAR").exists()
    assert not (tmp_path / INPUT_RECORD).exists()
    assert "873_s_1" in (tmp_path / "873_1" / "jobsub").read_text()
    assert [Row.Simulation for Row in Summary.BuildSummary(tmp_path, LiveIds=None)] == ["873_1"]


def TestGroupedPreparationIsIsolatedAndIdempotent(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    for Name, Element, CellX in [("low", "C", 16), ("high", "N", 24)]:
        Root = Campaign / "jobs" / Name
        Vsd = Root / "873_1" / "Dir_VolSearch"
        assert TrajectoryRoot(Vsd) == (Root, "873_1")
        Positions, Cell = Vio.ReadPoscar(Vsd)
        assert set(Positions.Element) == {"Zr", Element, "O"}
        assert Cell.iloc[0, 0] == CellX
        assert (Vsd / "POTCAR").read_bytes() == (Root / "POTCAR").read_bytes()
        assert Vio.ReadKeyValueFile(Vsd / "job.in")["temp"] == "873"
        assert "{}_873_s_1".format(Name) in (Vsd / "jobsub").read_text()
        assert (Root / INPUT_RECORD).exists()
    Before = Snapshot(Campaign)
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    assert Snapshot(Campaign) == Before
    Rows = Summary.BuildSummary(Campaign, Campaign / "OxParams", LiveIds=None)
    assert {Row.Simulation for Row in Rows} == {"low/873_1", "high/873_1"}
    Summary.WriteOutputs(Campaign, Rows)
    Text = (Campaign / Summary.SUMMARY_TXT).read_text(encoding="utf-8")
    assert Text.splitlines()[2].split()[:3] == ["JobFolder", "Trajectory", "Status"]
    assert {tuple(Line.split()[:2]) for Line in Text.splitlines()[4:]} == {
        ("low", "873_1"), ("high", "873_1"),
    }
    with (Campaign / Summary.SUMMARY_TSV).open(encoding="utf-8", newline="") as File:
        Exported = list(csv.DictReader(File, delimiter="\t"))
    assert len(Exported) == 2
    assert {(Row["JobFolder"], Row["Trajectory"]) for Row in Exported} == {
        ("low", "873_1"), ("high", "873_1"),
    }
    assert all(Row["Status"] == "NOT_STARTED" for Row in Exported)


@pytest.mark.parametrize("Value", ["False", "True", "'jobs/low'", "[1]", "['']", "[", "{}"])
def TestInvalidJobSpecsFailWithoutMutation(Campaign, monkeypatch, Value):
    (Campaign / "OxParams").write_text("JobSpecs = {}\n".format(Value), encoding="utf-8")
    Before = Snapshot(Campaign)
    assert Invoke(monkeypatch, Campaign) == 1
    assert Snapshot(Campaign) == Before


@pytest.mark.parametrize("Entries", [
    ["jobs/low", "jobs/low"], ["."], ["../outside"], ["jobs/missing"],
    ["jobs/low", "jobs/low/child"], ["jobs/low", "other/low"],
])
def TestAmbiguousOrInvalidRootsAreRejected(Campaign, Entries):
    WriteWorkspace(Campaign / "jobs" / "low" / "child")
    WriteWorkspace(Campaign / "other" / "low")
    (Campaign / "OxParams").write_text("JobSpecs = {!r}\n".format(Entries), encoding="utf-8")
    with pytest.raises(ValueError):
        LoadJobSpecs(Campaign)


def TestNestedJobSpecsAreRejected(Campaign):
    Config = Campaign / "jobs" / "low" / "OxParams"
    Config.write_text(Config.read_text() + 'JobSpecs = ["child"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Nested JobSpecs"):
        LoadJobSpecs(Campaign)


@pytest.mark.parametrize("Change", ["missing_file", "duplicate_temperature", "invalid_replicas"])
def TestAllSpecsValidatedBeforeAnyMutation(Campaign, monkeypatch, Change):
    Root = Campaign / "jobs" / "high"
    if Change == "missing_file":
        (Root / "POTCAR").unlink()
    else:
        Config = Root / "OxParams"
        Text = Config.read_text()
        Text = Text.replace("[873]", "[873, 873]") if Change == "duplicate_temperature" else Text.replace("NSims = 1", "NSims = 0")
        Config.write_text(Text, encoding="utf-8")
    Before = Snapshot(Campaign)
    assert Invoke(monkeypatch, Campaign) == 1
    assert Snapshot(Campaign) == Before


def TestDryRunDoesNotWrite(Campaign, monkeypatch, capsys):
    Before = Snapshot(Campaign)
    assert Invoke(monkeypatch, Campaign, Dry=True) == 0
    assert Snapshot(Campaign) == Before
    Output = capsys.readouterr().out
    assert "low/873_1" in Output and "high/873_1" in Output


def TestPreparedRunsSubmitTogetherAndDoNotResubmitQueuedJobs(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Submitted, Launched = [], []
    def Submit(Argv, **Kwargs):
        Submitted.append(Path(Kwargs["cwd"]))
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 100", stderr="")
    monkeypatch.setattr(Controller.subprocess, "run", Submit)
    monkeypatch.setattr(Controller, "RunOrchestration", lambda Root, Params, Dirs: Launched.append(Dirs) or 0)
    assert Invoke(monkeypatch, Campaign) == 0
    assert len(Submitted) == 2
    assert len(Launched) == 1
    assert {Label for Label, _ in Launched[0]} == {"low/873_1", "high/873_1"}
    # No OUTCAR yet: a scheduler acceptance must still prevent a duplicate.
    assert Invoke(monkeypatch, Campaign) == 0
    assert len(Submitted) == 2


def TestFailedSubmissionIsIsolatedAndRetryable(Campaign, monkeypatch):
    Launched = []
    def Submit(Argv, **Kwargs):
        Failed = "low" in Path(Kwargs["cwd"]).parts
        return SimpleNamespace(returncode=int(Failed), stdout="" if Failed else "Submitted batch job 100", stderr="rejected" if Failed else "")
    monkeypatch.setattr(Controller.subprocess, "run", Submit)
    monkeypatch.setattr(Controller, "RunOrchestration", lambda Root, Params, Dirs: Launched.append(Dirs) or 0)
    assert Invoke(monkeypatch, Campaign) == 1
    assert [Label for Label, _ in Launched[0]] == ["high/873_1"]
    Vsd = Campaign / "jobs" / "low" / "873_1" / "Dir_VolSearch"
    assert not (Vsd / ".vasp_submitted_step").exists()
    assert (Vsd / "job.exit").read_text() == "-1"
    Submitted = []
    monkeypatch.setattr(Controller.subprocess, "run", lambda Argv, **Kwargs: Submitted.append(Path(Kwargs["cwd"])) or SimpleNamespace(returncode=0, stdout="Submitted batch job 101", stderr=""))
    assert Invoke(monkeypatch, Campaign) == 0
    assert Submitted == [Vsd]
    assert {Label for Label, _ in Launched[-1]} == {"low/873_1", "high/873_1"}


@pytest.mark.parametrize("Evidence", ["empty_outcar", "running_outcar", "finished_outcar", "step_folder", "submission_log"])
def TestExistingJobsAreNotDuplicated(tmp_path, Evidence):
    if Evidence == "step_folder":
        (tmp_path / "1").mkdir()
    elif Evidence == "submission_log":
        StatusLog.Append(tmp_path, "SGUSCHI", "submitted", "1 100")
    else:
        Text = {"empty_outcar": "", "running_outcar": "running", "finished_outcar": "Total CPU"}[Evidence]
        (tmp_path / "OUTCAR").write_text(Text)
    assert not Controller.NeedsInitialVaspJob(tmp_path)


@pytest.mark.parametrize("Filename,Old,New", [
    ("OxParams", "O2Tol = 0.1", "O2Tol = 0.2"),
    ("POSCAR", "Zr C", "Zr N"),
    ("POTCAR", "synthetic", "changed"),
    ("job.in", "navg = 3", "navg = 4"),
])
def TestScientificChangesCannotResume(Campaign, monkeypatch, Filename, Old, New):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Input = Campaign / "jobs" / "low" / Filename
    Input.write_text(Input.read_text().replace(Old, New), encoding="utf-8")
    Before = Snapshot(Campaign)
    assert Invoke(monkeypatch, Campaign) == 1
    assert Snapshot(Campaign) == Before


def TestAddingRunsAndExtendingRuntimeAreAllowed(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Config = Campaign / "jobs" / "low" / "OxParams"
    Config.write_text(Config.read_text().replace("[873]", "[873, 973]").replace("NSims = 1", "NSims = 2").replace("MaxRuntime = 100", "MaxRuntime = 200"), encoding="utf-8")
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    assert (Campaign / "jobs" / "low" / "973_2" / "Dir_VolSearch").is_dir()
    assert not (Campaign / "jobs" / "high" / "973_2").exists()
    assert len(Summary.BuildSummary(Campaign, LiveIds=None)) == 5


def TestRuntimeCapsAreIndependentAndDryRunMatchesResume(Campaign, monkeypatch, capsys):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    for Name in ("low", "high"):
        Vsd = Campaign / "jobs" / Name / "873_1" / "Dir_VolSearch"
        for Marker in ("volsearch_is_done", "maxruntime_reached"):
            (Vsd / Marker).touch()
        (Vsd / "job.exit").write_text("0")
        (Vsd / "OUTCAR").write_text("Total CPU")
        (Vsd / "RateAnalysis.csv").write_text("Time (fs)\n100000\n")
    Config = Campaign / "jobs" / "low" / "OxParams"
    Config.write_text(Config.read_text().replace("MaxRuntime = 100", "MaxRuntime = 200"), encoding="utf-8")
    Before = Snapshot(Campaign)
    capsys.readouterr()
    assert Invoke(monkeypatch, Campaign, Dry=True) == 0
    assert "Would start volsearch_cont in: low/873_1" in capsys.readouterr().out
    assert Snapshot(Campaign) == Before
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    assert Snapshot(Campaign) == Before
    Launched = []
    monkeypatch.setattr(Controller, "RunOrchestration", lambda Root, Params, Dirs: Launched.extend(Dirs) or 0)
    assert Invoke(monkeypatch, Campaign) == 0
    assert [Label for Label, _ in Launched] == ["low/873_1"]
    assert (Campaign / "jobs" / "high" / "873_1" / "Dir_VolSearch" / "volsearch_is_done").exists()


def TestRemovingSpecUpdatesSummaryWithoutTouchingItsRuns(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Removed = Snapshot(Campaign / "jobs" / "high")
    (Campaign / "OxParams").write_text('JobSpecs = ["jobs/low"]\n', encoding="utf-8")
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    assert [Row.Simulation for Row in Summary.BuildSummary(Campaign, LiveIds=None)] == ["low/873_1"]
    assert Snapshot(Campaign / "jobs" / "high") == Removed


def TestRootScientificSettingsAreIgnoredInGroupedMode(Campaign, monkeypatch):
    Config = Campaign / "OxParams"
    Config.write_text(Config.read_text() + "Temperatures = [999]\nNSims = 9\nO2Tol = 999\n", encoding="utf-8")
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Rows = Summary.BuildSummary(Campaign, Config, LiveIds=None)
    assert {Row.Simulation for Row in Rows} == {"low/873_1", "high/873_1"}
    assert not (Campaign / "999_1").exists()


def TestCampaignSummaryOnlyIncludesSelectedRuns(tmp_path, monkeypatch):
    WriteWorkspace(tmp_path / "873_1")
    # Leftover top-level trajectory from an earlier single-workspace campaign.
    (tmp_path / "973_1" / "Dir_VolSearch").mkdir(parents=True)
    (tmp_path / "OxParams").write_text('JobSpecs = ["873_1"]\n')
    assert Invoke(monkeypatch, tmp_path, Prepare=True) == 0
    for Config in (None, tmp_path / "OxParams"):
        Rows = Summary.BuildSummary(tmp_path, Config, LiveIds=None)
        assert [Row.Simulation for Row in Rows] == ["873_1/873_1"]


def TestDisablingGroupingRestoresLegacySummary(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    WriteWorkspace(Campaign)
    with (Campaign / "OxParams").open("a") as File:
        File.write("JobSpecs = []\n")
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Rows = Summary.BuildSummary(Campaign, Campaign / "OxParams", LiveIds=None)
    assert [Row.Simulation for Row in Rows] == ["873_1"]


@pytest.mark.parametrize("Entry", ["SimulationSummary", "SimulationSummary/caseA", ".simulation_summary/caseA", "logs/caseA"])
def TestControllerOutputLocationsCannotBeSpecs(tmp_path, monkeypatch, Entry):
    WriteWorkspace(tmp_path / Entry)
    (tmp_path / "OxParams").write_text("JobSpecs = {!r}\n".format([Entry]))
    Before = Snapshot(tmp_path)
    assert Invoke(monkeypatch, tmp_path) == 1
    assert Snapshot(tmp_path) == Before


def TestScientificMismatchDoesNotReopenOtherSpecs(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Low = Campaign / "jobs" / "low"
    Vsd = Low / "873_1" / "Dir_VolSearch"
    (Vsd / "RateAnalysis.csv").write_text("Time (fs)\n1000\n")
    (Vsd / "volsearch_is_done").touch()
    (Vsd / "maxruntime_reached").touch()
    Bad = Campaign / "jobs" / "high" / "OxParams"
    Bad.write_text(Bad.read_text().replace("O2Tol = 1.0", "O2Tol = 2.0"), encoding="utf-8")
    Before = Snapshot(Campaign)
    assert Invoke(monkeypatch, Campaign) == 1
    assert Snapshot(Campaign) == Before


def TestCorruptInputRecordFailsWithoutMutation(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    (Campaign / "jobs" / "low" / INPUT_RECORD).write_text("{broken", encoding="utf-8")
    Before = Snapshot(Campaign)
    assert Invoke(monkeypatch, Campaign) == 1
    assert Snapshot(Campaign) == Before


def TestDirectChildInvocationPreservesScientificGuard(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Root = Campaign / "jobs" / "low"
    Config = Root / "OxParams"
    Config.write_text(Config.read_text().replace("O2Tol = 0.1", "O2Tol = 0.2"), encoding="utf-8")
    Before = Snapshot(Campaign)
    assert Invoke(monkeypatch, Root) == 1
    assert Snapshot(Campaign) == Before


def TestLegacyScientificSettingsRemainEditable(tmp_path, monkeypatch):
    WriteWorkspace(tmp_path)
    assert Invoke(monkeypatch, tmp_path, Prepare=True) == 0
    Config = tmp_path / "OxParams"
    Config.write_text(Config.read_text().replace("O2Tol = 0.1", "O2Tol = 0.2"), encoding="utf-8")
    assert Invoke(monkeypatch, tmp_path, Prepare=True) == 0


def TestSchedulerTemplateChangesAreAllowed(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Root = Campaign / "jobs" / "low"
    (Root / "jobsub").write_text("#!/bin/bash\n#SBATCH --job-name=new\n#SBATCH --mem=4G\n")
    JobIn = Root / "job.in"
    JobIn.write_text(JobIn.read_text().replace("sbatch", "sbatch --parsable"))
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0


def TestAmbiguousTimeoutDoesNotDuplicateSubmission(Campaign, monkeypatch):
    Submitted = []
    def Timeout(Argv, **Kwargs):
        Submitted.append(Kwargs["cwd"])
        return SimpleNamespace(returncode=1, stdout="", stderr="Socket timed out on send/recv operation")
    monkeypatch.setattr(Controller.subprocess, "run", Timeout)
    assert Invoke(monkeypatch, Campaign) == 1
    assert len(Submitted) == 2
    monkeypatch.setattr(Controller, "RunOrchestration", lambda *Args: 0)
    assert Invoke(monkeypatch, Campaign) == 0
    assert len(Submitted) == 2


def TestOrchestrationStartsBothSpecsBeforeWaiting(Campaign, monkeypatch):
    assert Invoke(monkeypatch, Campaign, Prepare=True) == 0
    Events = []
    class Process:
        pid = 100
        def __init__(self, Name):
            self.Name = Name
        def wait(self):
            Events.append(("wait", self.Name))
            return 1 if self.Name == "low" else 0
    def Launch(Argv, **Kwargs):
        Name = Path(Kwargs["cwd"]).parents[1].name
        Events.append(("launch", Name))
        return Process(Name)
    monkeypatch.setattr(Controller.subprocess, "Popen", Launch)
    monkeypatch.setattr(Controller.os, "access", lambda *Args: True)
    monkeypatch.setattr(Controller.signal, "signal", lambda *Args: None)
    monkeypatch.setattr(Controller, "SIMULATION_SUMMARY_SCRIPT", Campaign / "absent_summary.py")
    Dirs = [(Name + "/873_1", Campaign / "jobs" / Name / "873_1" / "Dir_VolSearch") for Name in ("low", "high")]
    assert RUN_ORCHESTRATION(Campaign, {}, Dirs) == 1
    assert Events == [("launch", "low"), ("launch", "high"), ("wait", "low"), ("wait", "high")]
    assert (Dirs[0][1] / "job.exit").read_text() == "1"
    assert (Dirs[1][1] / "job.exit").read_text() == "0"


def TestSignalHandlerTracksBothSpecs(Campaign):
    Terminated = []
    class Process:
        def __init__(self, Name):
            self.Name = Name
        def poll(self):
            return None
        def terminate(self):
            Terminated.append(self.Name)
    Procs = {}
    for Name in ("low", "high"):
        Vsd = Campaign / "jobs" / Name / "873_1" / "Dir_VolSearch"
        Vsd.mkdir(parents=True)
        Procs[Name + "/873_1"] = (Process(Name), Vsd)
    with pytest.raises(SystemExit):
        Controller._MakeSigtermHandler(Procs)(None, None)
    assert Terminated == ["low", "high"]
    assert all(StatusLog.LastEvent(Vsd, "killed") for _, Vsd in Procs.values())
