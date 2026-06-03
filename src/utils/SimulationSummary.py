"""Create a terminal-readable summary of SGUSCHI simulation folders.

The scanner is intentionally independent from the Slurm controller.  It can be
called by the shell master script or imported by a future Python master.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import getpass
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Make the project's src/ importable when this scanner runs as a standalone
# script (sys.path[0] is then src/utils); a no-op when imported as a package.
_SrcDir = str(Path(__file__).resolve().parents[1])
if _SrcDir not in sys.path:
    sys.path.append(_SrcDir)

from utils.FolderUtils import NumericStepFolders
from utils import StatusLog

# Stuck detection: a step whose scheduler job is gone while its OUTCAR has not
# yet completed is dead. A short grace absorbs the submit->appears-in-queue and
# completed->processed races before the verdict is trusted.
SUBMIT_GRACE_SECONDS = 120

# Sentinel so callers can pass LiveIds=None ("scheduler unknown") explicitly,
# distinct from the default "auto-detect and query the scheduler once".
_AUTO_LIVE_IDS = object()


SUMMARY_TXT = Path("SimulationSummary")
SUMMARY_TSV = Path("logs") / "SimulationSummary.tsv"
EXPECTED_PATH = Path(".simulation_summary") / "expected.tsv"
SIMULATION_RE = re.compile(r"^\d+_\d+$")


@dataclass
class SimulationRow:
    """One row in the root-level simulation summary."""

    Simulation: str
    Status: str
    LastUpdate: str
    Folders: str
    RateRows: str
    SimTime_ps: str
    TotalO2Added: str
    MoleculesRemoved: str
    WallTime: str
    QueueTime: str
    Done: str
    Failed: str
    Detail: str


# Each entry is (field_name, display_header). Field names must be valid Python
# identifiers and must match the SimulationRow dataclass fields.
SUMMARY_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Simulation",       "Simulation"),
    ("Status",           "Status"),
    ("LastUpdate",       "Age"),
    ("Folders",          "Folders"),
    ("RateRows",         "RateRows"),
    ("SimTime_ps",       "Time_ps"),
    ("TotalO2Added",     "O2Added"),
    ("MoleculesRemoved", "GasRemoved"),
    ("WallTime",         "Wall"),
    ("QueueTime",        "Queue"),
    ("Done",             "Done"),
    ("Failed",           "Failed"),
    ("Detail",           "Detail"),
)
SUMMARY_HEADERS = tuple(Field for Field, _ in SUMMARY_COLUMNS)
SUMMARY_DISPLAY_HEADERS = tuple(Display for _, Display in SUMMARY_COLUMNS)


def ReadExpectedSimulations(RootDir: Path) -> List[Tuple[str, Path]]:
    """Read optional expected labels written by a master process."""
    ExpectedFile = RootDir / EXPECTED_PATH
    if not ExpectedFile.exists():
        return []

    Rows: List[Tuple[str, Path]] = []
    with ExpectedFile.open("r", encoding="utf-8", errors="ignore", newline="") as File:
        Reader = csv.DictReader(File, delimiter="\t")
        for Row in Reader:
            Label = (Row.get("Simulation") or "").strip()
            WorkDirRaw = (Row.get("WorkDir") or "").strip()
            if not Label:
                continue
            WorkDir = RootDir / WorkDirRaw if WorkDirRaw else RootDir / Label / "Dir_VolSearch"
            Rows.append((Label, WorkDir))
    return Rows


def ReadOxParamsSimulations(RootDir: Path, OxParamsPath: Optional[Path]) -> List[Tuple[str, Path]]:
    """Read expected simulation labels from an OxParams file."""
    if OxParamsPath is None or not OxParamsPath.exists():
        return []

    try:
        Text = OxParamsPath.read_text(encoding="utf-8", errors="ignore")
        Tree = ast.parse(Text, filename=str(OxParamsPath))
    except (OSError, SyntaxError):
        return []

    Assignments = {}
    for Node in Tree.body:
        if not isinstance(Node, ast.Assign):
            continue
        for Target in Node.targets:
            if isinstance(Target, ast.Name) and Target.id in {"Temperatures", "NSims"}:
                try:
                    Assignments[Target.id] = ast.literal_eval(Node.value)
                except (TypeError, ValueError, SyntaxError):
                    pass

    Temperatures = Assignments.get("Temperatures")
    NSims = Assignments.get("NSims")
    if not isinstance(Temperatures, (list, tuple)):
        return []
    if isinstance(NSims, bool) or not isinstance(NSims, int) or NSims < 1:
        return []

    Rows: List[Tuple[str, Path]] = []
    for Temperature in Temperatures:
        TemperatureLabel = FormatOxParamToken(Temperature)
        if not TemperatureLabel:
            continue
        for Case in range(1, NSims + 1):
            Label = f"{TemperatureLabel}_{Case}"
            Rows.append((Label, RootDir / Label / "Dir_VolSearch"))
    return Rows


def FormatOxParamToken(Value: object) -> str:
    """Format an OxParams token for folder names such as 873_1."""
    if isinstance(Value, bool):
        return ""
    if isinstance(Value, int):
        return str(Value)
    if isinstance(Value, float):
        return str(int(Value)) if Value.is_integer() else FormatFloat(str(Value))
    return str(Value).strip()


def DiscoverSimulations(
    RootDir: Path,
    OxParamsPath: Optional[Path] = None,
) -> List[Tuple[str, Path]]:
    """Find simulation folders under RootDir."""
    Expected = ReadExpectedSimulations(RootDir) + ReadOxParamsSimulations(RootDir, OxParamsPath)
    Found = {}
    for Label, WorkDir in Expected:
        Found.setdefault(Label, WorkDir)

    for Child in sorted(RootDir.iterdir(), key=lambda P: P.name):
        if not Child.is_dir():
            continue
        WorkDir = Child / "Dir_VolSearch"
        if WorkDir.is_dir() or SIMULATION_RE.match(Child.name):
            Found.setdefault(Child.name, WorkDir)

    return sorted(Found.items(), key=lambda Item: NaturalSortKey(Item[0]))


def NaturalSortKey(Text: str) -> Tuple:
    """Sort labels like 873_2 numerically when possible."""
    Parts = re.split(r"(\d+)", Text)
    Key = []
    for Part in Parts:
        Key.append(int(Part) if Part.isdigit() else Part)
    return tuple(Key)


def ReadRateAnalysis(WorkDir: Path) -> Tuple[str, str, str, str]:
    """Return compact RateAnalysis metrics without importing pandas."""
    RatePath = WorkDir / "RateAnalysis.csv"
    if not RatePath.exists():
        return "-", "-", "-", "-"

    try:
        with RatePath.open("r", encoding="utf-8-sig", errors="ignore", newline="") as File:
            Reader = csv.DictReader(File)
            Rows = list(Reader)
            FieldNames = Reader.fieldnames or []
    except Exception:
        return "ERR", "-", "-", "-"

    if not Rows:
        return "0", "-", "-", "-"

    TimeValue = "-"
    for Key in ("Time (fs)", "Time", "Time_fs"):
        Raw = Rows[-1].get(Key)
        if Raw not in (None, ""):
            TimeValue = FormatPicoseconds(Raw)
            break

    TotalO2Added = "-"
    RawO2Added = Rows[-1].get("O2 Added")
    if RawO2Added not in (None, ""):
        TotalO2Added = FormatFloat(RawO2Added)

    MoleculesRemoved = CountRemovedMolecules(Rows, FieldNames)
    return str(len(Rows)), TimeValue, TotalO2Added, MoleculesRemoved


def CountRemovedMolecules(Rows: Sequence[Dict[str, str]], FieldNames: Sequence[str]) -> str:
    """Count removed molecule entries from the Gas Removed column.

    Skips malformed individual rows rather than failing the entire count.
    Appends '?' to the result if any rows were unparseable.
    """
    if "Gas Removed" not in FieldNames:
        return "-"

    Total = 0
    Skipped = 0
    for Row in Rows:
        Raw = Row.get("Gas Removed")
        if Raw in (None, ""):
            continue
        try:
            Removed = ast.literal_eval(Raw)
        except (SyntaxError, ValueError, TypeError):
            Skipped += 1
            continue
        if not isinstance(Removed, (list, tuple)):
            Skipped += 1
            continue
        Valid = True
        for Molecule in Removed:
            if not isinstance(Molecule, (list, tuple)):
                Valid = False
                break
        if Valid:
            Total += len(Removed)
        else:
            Skipped += 1

    return f"{Total}?" if Skipped else str(Total)


def FormatFloat(Value: str) -> str:
    """Format a numeric string compactly, preserving non-numeric values."""
    try:
        Number = float(Value)
    except (TypeError, ValueError):
        return str(Value)
    if Number.is_integer():
        return str(int(Number))
    return f"{Number:.3f}".rstrip("0").rstrip(".")


def FormatPicoseconds(Value: str) -> str:
    """Convert a femtosecond string to compact picoseconds text."""
    try:
        Number = float(Value) / 1000.0
    except (TypeError, ValueError):
        return str(Value)
    if Number.is_integer():
        return str(int(Number))
    return f"{Number:.6f}".rstrip("0").rstrip(".")


def LiveJobIds() -> Optional[Set[str]]:
    """Return the set of the current user's live scheduler job IDs, or None.

    Cluster-agnostic and zero-config: auto-detects the scheduler client on PATH
    (Slurm `squeue`, else PBS `qstat`). Returns None ("liveness unknown") when no
    client is found or the query errors/times out — a plain sentinel that NEVER
    raises, calls sys.exit, or sets a non-zero process exit code, so the summary
    keeps running. Called once per refresh and reused for every simulation.
    """
    try:
        User = getpass.getuser()
    except Exception:
        User = os.environ.get("USER") or os.environ.get("USERNAME") or ""

    if shutil.which("squeue"):
        return _RunJobIdQuery(["squeue", "-h", "-u", User, "-o", "%i"], _ParseSlurmIds)
    if shutil.which("qstat"):
        return _RunJobIdQuery(["qstat", "-u", User], _ParsePbsIds)
    return None


def _RunJobIdQuery(Command: Sequence[str], Parser) -> Optional[Set[str]]:
    """Run a scheduler query and parse live job IDs; any failure → None."""
    try:
        Result = subprocess.run(
            list(Command), capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if Result.returncode != 0:
        return None
    try:
        return Parser(Result.stdout)
    except Exception:
        return None


def _ParseSlurmIds(Text: str) -> Set[str]:
    """Parse `squeue -h -o %i` output into a set of base numeric job IDs."""
    Ids: Set[str] = set()
    for Line in Text.splitlines():
        Token = Line.strip()
        if not Token:
            continue
        Base = Token.split("_", 1)[0].split(".", 1)[0]  # drop array/step suffix
        if Base.isdigit():
            Ids.add(Base)
            Ids.add(Token)
    return Ids


def _ParsePbsIds(Text: str) -> Set[str]:
    """Parse `qstat -u` output into a set of base numeric job IDs."""
    Ids: Set[str] = set()
    for Line in Text.splitlines():
        Fields = Line.split()
        if not Fields:
            continue
        First = Fields[0]
        Base = First.split(".", 1)[0]
        if Base.isdigit():
            Ids.add(Base)
            Ids.add(First)
    return Ids


def OutcarHasCompleted(OutcarPath: Path) -> bool:
    """True if the (top-level, in-flight) OUTCAR carries VASP's 'Total CPU' line.

    Tail-reads only the last ~8 KB (the timing line is at the end) so this stays
    cheap when called per-sim every refresh. Empty/missing OUTCAR → not completed.
    """
    try:
        Size = OutcarPath.stat().st_size
    except OSError:
        return False
    if Size == 0:
        return False
    try:
        with OutcarPath.open("rb") as FileObj:
            if Size > 8192:
                FileObj.seek(-8192, os.SEEK_END)
            Tail = FileObj.read()
    except OSError:
        return False
    return b"Total CPU" in Tail


def ParseSubmitEvents(Events: Sequence["StatusLog.LogEvent"]) -> Dict[int, List[Tuple[datetime.datetime, Optional[str]]]]:
    """Map step → [(submit_time, job_id_or_None)] from 'submitted' log events.

    Detail is '<step> <jobid>'. Malformed entries are skipped.
    """
    Result: Dict[int, List[Tuple[datetime.datetime, Optional[str]]]] = {}
    for Event in Events:
        if Event.Event != "submitted":
            continue
        Tokens = Event.Detail.split()
        if not Tokens:
            continue
        try:
            Step = int(Tokens[0])
        except ValueError:
            continue
        JobId = Tokens[1] if len(Tokens) > 1 else None
        Result.setdefault(Step, []).append((Event.Timestamp, JobId))
    return Result


def ParseExitCode(Detail: str) -> int:
    """Parse the rc from an 'exit' event detail; unparseable → non-zero (1)."""
    try:
        return int((Detail or "").split()[0])
    except (ValueError, IndexError):
        return 1


def EstimateWallTime(WorkDir: Path, StepFolders: Sequence[int]) -> str:
    """Estimate wall-clock span from step folders and marker/log timestamps.

    Uses st_ctime (creation time on Windows, inode-change time on POSIX) as an
    additional lower-bound anchor to improve stability on OneDrive-synced paths
    where mtime can be reset on re-sync.
    """
    Times: List[float] = []

    def AddStat(PathFile: Path) -> None:
        try:
            Stat = PathFile.stat()
            Times.append(Stat.st_mtime)
            if Stat.st_ctime < Stat.st_mtime:
                Times.append(Stat.st_ctime)
        except OSError:
            pass

    for Step in StepFolders:
        AddStat(WorkDir / str(Step))

    for Name in ("volsearch_is_done", "sguschi_failed", "RateAnalysis.csv", "jobsub.log",
                 "job.exit", "maxruntime_reached", StatusLog.SIM_LOG):
        PathFile = WorkDir / Name
        if PathFile.exists():
            AddStat(PathFile)

    ParentLog = WorkDir.parent / "log.out"
    if ParentLog.exists():
        AddStat(ParentLog)

    if len(Times) < 2:
        return "-"
    return FormatDuration(max(Times) - min(Times))


def LastUpdateAge(WorkDir: Path) -> str:
    """Time since the most recent simulation activity (best-effort, mtime-based).

    Reuses LatestActivityTime (the same progress-file set that drives the
    'recent activity' RUNNING status) so the Age column stays consistent with
    status detection. Returns '-' when nothing is readable. Shares the OneDrive
    mtime caveat noted on EstimateWallTime.
    """
    Recent = LatestActivityTime(WorkDir)
    if Recent is None:
        return "-"
    return FormatDuration(time.time() - Recent)


# VASP writes its run start time in the OUTCAR header, e.g.:
#   executed on             LinuxIFC date 2024.03.11  09:21:43
_OUTCAR_DATE_RE = re.compile(
    r"executed on.*?date\s+(\d{4})\.(\d{2})\.(\d{2})\s+(\d{2}):(\d{2}):(\d{2})"
)


def ReadOutcarStartTime(OutcarPath: Path) -> Optional[datetime.datetime]:
    """Parse VASP's 'executed on ... date ...' header line as a naive datetime.

    Returns None if the file is missing/unreadable or the line is absent (some
    MLFF/older builds omit it) — the caller then treats that step as having no
    measurable queue time rather than failing.
    """
    try:
        with OutcarPath.open("r", encoding="utf-8", errors="ignore") as File:
            for _ in range(60):
                Line = File.readline()
                if not Line:
                    break
                Match = _OUTCAR_DATE_RE.search(Line)
                if Match:
                    try:
                        return datetime.datetime(*(int(Group) for Group in Match.groups()))
                    except ValueError:
                        return None
    except OSError:
        return None
    return None


def ComputeQueueTime(
    WorkDir: Path,
    StepFolders: Sequence[int],
    Events: Sequence["StatusLog.LogEvent"],
) -> str:
    """Cumulative queue (wait) time across all steps that have both timestamps.

    Per step: VASP start (from the archived OUTCAR header) minus the latest
    `submitted` log event recorded at or before that start, so a resubmitted step
    pairs with the submission that actually launched it. Steps missing either
    timestamp, or with a non-positive delta (clock skew), contribute zero. Returns
    '-' only when no step yields a usable submit/start pair — distinct from '0s'.
    """
    SubmitEvents = ParseSubmitEvents(Events)
    if not SubmitEvents:
        return "-"

    # Archived steps live in <step>/OUTCAR; the in-flight step's OUTCAR is top-level.
    Candidates: List[Tuple[int, Path]] = [
        (Step, WorkDir / str(Step) / "OUTCAR") for Step in StepFolders
    ]
    InFlight = (max(StepFolders) + 1) if StepFolders else 1
    Candidates.append((InFlight, WorkDir / "OUTCAR"))

    TotalSeconds = 0.0
    Found = False
    for Step, OutcarPath in Candidates:
        Submits = SubmitEvents.get(Step)
        if not Submits:
            continue
        Start = ReadOutcarStartTime(OutcarPath)
        if Start is None:
            continue
        Eligible = [Stamp for Stamp, _ in Submits if Stamp <= Start]
        if not Eligible:
            continue
        Found = True
        Delta = (Start - max(Eligible)).total_seconds()
        if Delta > 0:
            TotalSeconds += Delta
    return FormatDuration(TotalSeconds) if Found else "-"


def FormatDuration(Seconds: float) -> str:
    """Format seconds as compact d/h/m/s text."""
    if Seconds < 0:
        return "-"
    SecondsInt = int(round(Seconds))
    Days, Rem = divmod(SecondsInt, 86400)
    Hours, Rem = divmod(Rem, 3600)
    Minutes, Secs = divmod(Rem, 60)
    if Days:
        return f"{Days}d{Hours:02d}h"
    if Hours:
        return f"{Hours}h{Minutes:02d}m"
    if Minutes:
        return f"{Minutes}m{Secs:02d}s"
    return f"{Secs}s"


def DetermineStatus(
    WorkDir: Path,
    StepFolders: Sequence[int],
    Events: Sequence["StatusLog.LogEvent"],
    LiveIds: Optional[Set[str]] = None,
) -> Tuple[str, str, str, str]:
    """Return status, done flag, failed flag, and detail.

    Sources: the engine's authoritative terminal markers via `stat`
    (volsearch_is_done / sguschi_failed / maxruntime_reached), the unified log for
    started/exit/killed/submitted lifecycle, and the scheduler liveness set for
    the stuck verdict. ``LiveIds`` is the per-refresh set of live scheduler job
    IDs, or None when liveness is unknown (off-cluster / query failed) — in which
    case a sim is never flagged STUCK on scheduler grounds.
    """
    if not WorkDir.exists():
        return "MISSING", "N", "Y", "missing"
    if not WorkDir.is_dir():
        return "MISSING", "N", "Y", "not a directory"

    # 1-3: engine terminal markers win (authoritative; cheap stat).
    if (WorkDir / "volsearch_is_done").exists():
        Detail = "max runtime" if (WorkDir / "maxruntime_reached").exists() else "done"
        return "DONE", "Y", "N", Detail
    if (WorkDir / "sguschi_failed").exists():
        return "FAILED", "N", "Y", "sguschi_failed"

    # 4-5: last lifecycle event from the unified log (order-correct across
    # restarts: a fresh 'started' supersedes any earlier 'exit'/'killed').
    Lifecycle = [E for E in Events if E.Event in ("started", "exit", "killed")]
    Last = Lifecycle[-1] if Lifecycle else None
    if Last is not None and Last.Event == "killed":
        return "KILLED", "N", "Y", "killed"
    if Last is not None and Last.Event == "exit":
        Rc = ParseExitCode(Last.Detail)
        if Rc == 0:
            return "DONE", "Y", "N", "exit 0"
        return "FAILED", "N", "Y", "exit {}".format((Last.Detail or "").strip() or "?")

    # 6: scheduler-aware stuck — the current step's job is gone while its OUTCAR
    # has not completed. Only with a known job ID, past the submit grace, and a
    # successful scheduler query (LiveIds is not None).
    VaspState = CheckVaspState(WorkDir, StepFolders)
    InFlightStep = (StepFolders[-1] + 1) if StepFolders else 1
    Submits = ParseSubmitEvents(Events).get(InFlightStep, [])
    CurrentSubmitTs, CurrentJobId = Submits[-1] if Submits else (None, None)
    Incomplete = not OutcarHasCompleted(WorkDir / "OUTCAR")

    if (
        Incomplete
        and CurrentJobId is not None
        and CurrentSubmitTs is not None
        and (time.time() - CurrentSubmitTs.timestamp()) > SUBMIT_GRACE_SECONDS
        and LiveIds is not None
        and CurrentJobId not in LiveIds
    ):
        return "STUCK", "N", "N", "Check .out file and resubmit"

    # 7: still alive / waiting.
    if VaspState == "queued":
        return "RUNNING", "N", "N", "queued"
    if Last is not None and Last.Event == "started":
        return "RUNNING", "N", "N", "started"

    RecentActivity = LatestActivityTime(WorkDir)
    if VaspState == "stuck":
        if RecentActivity and (time.time() - RecentActivity) <= 2 * 3600:
            return "RUNNING", "N", "N", "starting up"
        return "STUCK", "N", "N", "no VASP job; restart master"

    if RecentActivity and (time.time() - RecentActivity) <= 2 * 3600:
        return "RUNNING", "N", "N", "recent activity"
    if not StepFolders:
        return "NOT_STARTED", "N", "N", "no steps"
    return "UNKNOWN", "N", "N", "no marker"


def LatestActivityTime(WorkDir: Path) -> Optional[float]:
    """Return latest mtime for files/folders relevant to the simulation."""
    Times: List[float] = []
    for Name in ("OUTCAR", "RateAnalysis.csv", "jobsub.log"):
        PathFile = WorkDir / Name
        if PathFile.exists():
            try:
                Times.append(PathFile.stat().st_mtime)
            except OSError:
                pass
    if WorkDir.is_dir():
        for StepPath in WorkDir.iterdir():
            if StepPath.name.isdigit():
                try:
                    Times.append(StepPath.stat().st_mtime)
                except OSError:
                    pass
    return max(Times) if Times else None


def CheckVaspState(WorkDir: Path, StepFolders: Sequence[int]) -> Optional[str]:
    """Detect empty-OUTCAR states using the .vasp_submitted_step sentinel.

    Returns 'stuck' if OUTCAR is empty and no valid submission record exists
    for the current nstep (simulation cannot proceed without a restart).
    Returns 'queued' if OUTCAR is empty but the sentinel matches the current
    nstep (VASP job was submitted and is likely waiting in the queue).
    Returns None if OUTCAR is non-empty or no step folders exist yet.
    """
    if not StepFolders:
        return None
    OutcarPath = WorkDir / "OUTCAR"
    if not OutcarPath.exists():
        return None
    try:
        if OutcarPath.stat().st_size > 0:
            return None
    except OSError:
        return None

    ExpectedNstep = StepFolders[-1] + 1
    SentinelPath = WorkDir / ".vasp_submitted_step"
    if SentinelPath.exists():
        try:
            Recorded = int(SentinelPath.read_text(encoding="utf-8").strip())
            if Recorded == ExpectedNstep:
                return "queued"
        except (ValueError, OSError):
            pass
    return "stuck"


def BuildRow(Label: str, WorkDir: Path, LiveIds: Optional[Set[str]] = None) -> SimulationRow:
    """Build one summary row. ``LiveIds`` is the per-refresh scheduler liveness set."""
    StepFolders = NumericStepFolders(WorkDir)
    Events = StatusLog.ReadEvents(WorkDir) if WorkDir.exists() else []
    Status, Done, Failed, Detail = DetermineStatus(WorkDir, StepFolders, Events, LiveIds)
    RateRows, SimTime, TotalO2Added, MoleculesRemoved = ReadRateAnalysis(WorkDir)

    return SimulationRow(
        Simulation=Label,
        Status=Status,
        LastUpdate=LastUpdateAge(WorkDir),
        Folders=str(len(StepFolders)) if WorkDir.exists() else "-",
        RateRows=RateRows,
        SimTime_ps=SimTime,
        TotalO2Added=TotalO2Added,
        MoleculesRemoved=MoleculesRemoved,
        WallTime=EstimateWallTime(WorkDir, StepFolders),
        QueueTime=ComputeQueueTime(WorkDir, StepFolders, Events),
        Done=Done,
        Failed=Failed,
        Detail=Detail,
    )


def BuildSummary(
    RootDir: Path,
    OxParamsPath: Optional[Path] = None,
    LiveIds=_AUTO_LIVE_IDS,
) -> List[SimulationRow]:
    """Scan RootDir and return all summary rows.

    The scheduler is queried once per pass (``LiveJobIds``) and the resulting
    live-ID set is reused for every simulation. Pass an explicit ``LiveIds`` (a
    set, or None for "unknown") to bypass the query — used by tests.
    """
    RootDir = RootDir.resolve()
    Simulations = DiscoverSimulations(RootDir, OxParamsPath)
    if LiveIds is _AUTO_LIVE_IDS:
        LiveIds = LiveJobIds()
    return [BuildRow(Label, WorkDir, LiveIds) for Label, WorkDir in Simulations]


def FormatTable(Rows: Sequence[SimulationRow]) -> str:
    """Return a fixed-width table suitable for terminal inspection."""
    Now = datetime.datetime.now(datetime.timezone.utc)
    Header = f"Generated: {Now.strftime('%Y-%m-%d %H:%M:%S')} UTC   ({len(Rows)} simulations)"

    Values = [[getattr(Row, Field) for Field, _ in SUMMARY_COLUMNS] for Row in Rows]
    Widths = []
    for Index, (_, Display) in enumerate(SUMMARY_COLUMNS):
        Widths.append(max(len(Display), *(len(str(Row[Index])) for Row in Values)) if Values else len(Display))

    Lines = [Header, ""]
    Lines.append("  ".join(Display.ljust(Widths[Index]) for Index, (_, Display) in enumerate(SUMMARY_COLUMNS)))
    Lines.append("  ".join("-" * Width for Width in Widths))
    for RowValues in Values:
        Lines.append("  ".join(str(Value).ljust(Widths[Index]) for Index, Value in enumerate(RowValues)))
    if not Values:
        Lines.append("(no simulations found)")
    return "\n".join(Lines) + "\n"


def WriteOutputs(RootDir: Path, Rows: Sequence[SimulationRow]) -> None:
    """Write text and TSV summaries atomically."""
    TextPath = RootDir / SUMMARY_TXT
    TsvPath = RootDir / SUMMARY_TSV
    TsvPath.parent.mkdir(parents=True, exist_ok=True)

    AtomicWriteText(TextPath, FormatTable(Rows))

    Lines = ["\t".join(Display for _, Display in SUMMARY_COLUMNS)]
    for Row in Rows:
        Lines.append("\t".join(str(getattr(Row, Field)) for Field, _ in SUMMARY_COLUMNS))
    AtomicWriteText(TsvPath, "\n".join(Lines) + "\n")


def AtomicWriteText(PathFile: Path, Text: str) -> None:
    """Write text via temp file then replace."""
    TmpPath = PathFile.with_name(PathFile.name + ".tmp")
    with TmpPath.open("w", encoding="utf-8", newline="\n") as File:
        File.write(Text)
    for _ in range(5):
        try:
            TmpPath.replace(PathFile)
            return
        except PermissionError:
            time.sleep(0.2)

    # Some synced Windows folders can briefly reject atomic replace operations.
    # The summary is advisory, so preserve output with a direct write fallback.
    with PathFile.open("w", encoding="utf-8", newline="\n") as File:
        File.write(Text)
    try:
        TmpPath.unlink()
    except OSError:
        pass


def ResolveOptionalPath(RootDir: Path, RawPath: Optional[str]) -> Optional[Path]:
    """Resolve an optional CLI path relative to RootDir."""
    if RawPath is None:
        return None
    PathValue = Path(RawPath)
    if not PathValue.is_absolute():
        PathValue = RootDir / PathValue
    return PathValue.resolve()


def WriteSummaryOnce(
    RootDir: Path,
    OxParamsPath: Optional[Path] = None,
    Quiet: bool = False,
    PrintOnly: bool = False,
) -> None:
    """Build and write one summary refresh."""
    Rows = BuildSummary(RootDir, OxParamsPath)
    if PrintOnly:
        print(FormatTable(Rows), end="")
        return
    WriteOutputs(RootDir, Rows)
    if not Quiet:
        print(f"Wrote {RootDir / SUMMARY_TXT}")
        print(f"Wrote {RootDir / SUMMARY_TSV}")


def BuildDaemonCommand(
    RootDir: Path,
    OxParamsPath: Optional[Path],
    Interval: float,
    Quiet: bool,
    ParentPid: int,
) -> List[str]:
    """Build the detached watcher command."""
    Command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(RootDir),
        "--watch",
        "--interval",
        FormatInterval(Interval),
        "--parent-pid",
        str(ParentPid),
    ]
    if OxParamsPath is not None:
        Command.extend(["--oxparams", str(OxParamsPath)])
    if Quiet:
        Command.append("--quiet")
    return Command


def FormatInterval(Value: float) -> str:
    """Format interval values without unnecessary .0 suffixes."""
    return str(int(Value)) if float(Value).is_integer() else str(Value)


def StartWatchDaemon(
    RootDir: Path,
    OxParamsPath: Optional[Path],
    Interval: float,
    Quiet: bool,
) -> int:
    """Start a detached watcher process and return immediately."""
    ParentPid = os.getppid()
    Command = BuildDaemonCommand(RootDir, OxParamsPath, Interval, Quiet, ParentPid)
    MetaDir = RootDir / ".simulation_summary"
    try:
        MetaDir.mkdir(exist_ok=True)
        LogFile = (MetaDir / "watcher.log").open("a", encoding="utf-8")
    except OSError as Error:
        print(f"Could not create summary watcher log: {Error}", file=sys.stderr)
        return 1

    PopenKwargs = {
        "cwd": str(RootDir),
        "stdin": subprocess.DEVNULL,
        "stdout": LogFile,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        PopenKwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        PopenKwargs["start_new_session"] = True

    try:
        Process = subprocess.Popen(Command, **PopenKwargs)
    except OSError as Error:
        print(f"Could not start summary watcher: {Error}", file=sys.stderr)
        LogFile.close()
        return 1
    LogFile.close()

    if not Quiet:
        print(f"Started summary watcher PID {Process.pid}")
    return 0


def RunWatchLoop(
    RootDir: Path,
    OxParamsPath: Optional[Path],
    Interval: float,
    Quiet: bool,
    ParentPid: Optional[int],
) -> int:
    """Refresh summaries until interrupted or the monitored parent exits."""
    try:
        WriteSummaryOnce(RootDir, OxParamsPath, Quiet)
        NextWrite = time.monotonic() + Interval
        while ParentPid is None or ProcessIsAlive(ParentPid):
            SleepFor = min(1.0, max(0.0, NextWrite - time.monotonic()))
            time.sleep(SleepFor)
            if time.monotonic() >= NextWrite:
                WriteSummaryOnce(RootDir, OxParamsPath, Quiet)
                NextWrite = time.monotonic() + Interval
        WriteSummaryOnce(RootDir, OxParamsPath, Quiet)
        return 0
    except KeyboardInterrupt:
        WriteSummaryOnce(RootDir, OxParamsPath, Quiet)
        return 130


def ProcessIsAlive(Pid: int) -> bool:
    """Return whether a process id appears to still exist."""
    if Pid <= 0:
        return False
    try:
        os.kill(Pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def ParseArgs(Args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    Parser = argparse.ArgumentParser(description=__doc__)
    Parser.add_argument(
        "RootDir",
        nargs="?",
        default=".",
        help="Root directory containing simulation folders such as 873_2.",
    )
    Parser.add_argument(
        "--oxparams",
        default=None,
        help="Optional OxParams file used to infer expected simulation folders.",
    )
    Parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep refreshing the summary until interrupted.",
    )
    Parser.add_argument(
        "--watch-daemon",
        action="store_true",
        help="Start a detached watcher and return immediately.",
    )
    Parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Refresh interval in seconds for watch modes.",
    )
    Parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress normal status output.",
    )
    Parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the summary table to stdout instead of writing files.",
    )
    Parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return Parser.parse_args(Args)


def Main(Args: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    Parsed = ParseArgs(Args)
    RootDir = Path(Parsed.RootDir).resolve()
    if not RootDir.exists():
        print(f"Root directory does not exist: {RootDir}", file=sys.stderr)
        return 1
    if Parsed.interval <= 0:
        print("Summary refresh interval must be positive.", file=sys.stderr)
        return 1

    OxParamsPath = ResolveOptionalPath(RootDir, Parsed.oxparams)
    if Parsed.watch_daemon:
        return StartWatchDaemon(RootDir, OxParamsPath, Parsed.interval, Parsed.quiet)
    if Parsed.watch:
        return RunWatchLoop(
            RootDir,
            OxParamsPath,
            Parsed.interval,
            Parsed.quiet,
            Parsed.parent_pid,
        )

    WriteSummaryOnce(RootDir, OxParamsPath, Parsed.quiet, Parsed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
