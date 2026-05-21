"""Create a terminal-readable summary of SGUSCHI simulation folders.

The scanner is intentionally independent from the Slurm controller.  It can be
called by the shell master script or imported by a future Python master.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SUMMARY_TXT = Path("SimulationSummary")
SUMMARY_TSV = Path("logs") / "SimulationSummary.tsv"
EXPECTED_PATH = Path(".simulation_summary") / "expected.tsv"
SIMULATION_RE = re.compile(r"^\d+_\d+$")
FATAL_RE = re.compile(r"\bFATAL\b", flags=re.IGNORECASE)


@dataclass
class SimulationRow:
    """One row in the root-level simulation summary."""

    Simulation: str
    Status: str
    Folders: str
    Latest: str
    RateRows: str
    SimTime_ps: str
    TotalO2Added: str
    MoleculesRemoved: str
    WallTime: str
    Done: str
    Failed: str
    Detail: str


# Each entry is (field_name, display_header). Field names must be valid Python
# identifiers and must match the SimulationRow dataclass fields.
SUMMARY_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Simulation",       "Simulation"),
    ("Status",           "Status"),
    ("Folders",          "Folders"),
    ("Latest",           "Latest"),
    ("RateRows",         "RateRows"),
    ("SimTime_ps",       "SimTime_ps"),
    ("TotalO2Added",     "TotalO2Added"),
    ("MoleculesRemoved", "MoleculesRemoved"),
    ("WallTime",         "WallTime"),
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


def NumericStepFolders(WorkDir: Path) -> List[int]:
    """Return numeric step folders found in a Dir_VolSearch directory."""
    if not WorkDir.is_dir():
        return []
    return sorted(
        int(Child.name)
        for Child in WorkDir.iterdir()
        if Child.is_dir() and Child.name.isdigit()
    )


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
    """Count removed molecule entries from the Gas Removed column."""
    if "Gas Removed" not in FieldNames:
        return "-"

    Total = 0
    for Row in Rows:
        Raw = Row.get("Gas Removed")
        if Raw in (None, ""):
            continue
        try:
            Removed = ast.literal_eval(Raw)
        except (SyntaxError, ValueError, TypeError):
            return "-"
        if not isinstance(Removed, (list, tuple)):
            return "-"
        for Molecule in Removed:
            if not isinstance(Molecule, (list, tuple)):
                return "-"
        Total += len(Removed)

    return str(Total)


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


def LatestFatalDetail(Paths: Sequence[Path]) -> Optional[str]:
    """Find the most recent fatal line in small tail windows of log files."""
    LatestLine: Optional[str] = None
    LatestMtime = -1.0

    for PathFile in Paths:
        if not PathFile.exists() or not PathFile.is_file():
            continue
        try:
            Text = ReadTail(PathFile)
            Mtime = PathFile.stat().st_mtime
        except OSError:
            continue
        for Line in Text.splitlines():
            if FATAL_RE.search(Line) and Mtime >= LatestMtime:
                LatestLine = " ".join(Line.strip().split())
                LatestMtime = Mtime

    return Truncate(LatestLine, 58) if LatestLine else None


def ReadTail(PathFile: Path, MaxBytes: int = 32768) -> str:
    """Read the tail of a text file."""
    Size = PathFile.stat().st_size
    with PathFile.open("rb") as File:
        if Size > MaxBytes:
            File.seek(Size - MaxBytes)
        Data = File.read()
    return Data.decode("utf-8", errors="ignore")


def EstimateWallTime(WorkDir: Path, StepFolders: Sequence[int]) -> str:
    """Estimate wall-clock span from step folders and marker/log timestamps."""
    Times: List[float] = []

    for Step in StepFolders:
        StepPath = WorkDir / str(Step)
        try:
            Times.append(StepPath.stat().st_mtime)
        except OSError:
            pass

    for Name in ("volsearch_is_done", "sguschi_failed", "RateAnalysis.csv", "jobsub.log"):
        PathFile = WorkDir / Name
        if PathFile.exists():
            try:
                Times.append(PathFile.stat().st_mtime)
            except OSError:
                pass

    ParentLog = WorkDir.parent / "log.out"
    if ParentLog.exists():
        try:
            Times.append(ParentLog.stat().st_mtime)
        except OSError:
            pass

    if len(Times) < 2:
        return "-"
    return FormatDuration(max(Times) - min(Times))


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
    FatalDetail: Optional[str],
) -> Tuple[str, str, str, str]:
    """Return status, done flag, failed flag, and detail."""
    if not WorkDir.exists():
        return "MISSING", "N", "Y", "Dir_VolSearch missing"
    if not WorkDir.is_dir():
        return "MISSING", "N", "Y", "Dir_VolSearch is not a directory"

    FailedMarker = WorkDir / "sguschi_failed"
    DoneMarker = WorkDir / "volsearch_is_done"
    VaspState = CheckVaspState(WorkDir, StepFolders)

    if DoneMarker.exists():
        return "DONE", "Y", "N", "volsearch_is_done"
    if VaspState == "queued":
        return "RUNNING", "N", "N", "OUTCAR empty, VASP job queued"
    if FailedMarker.exists():
        return "FAILED", "N", "Y", "sguschi_failed"
    if FatalDetail:
        return "FAILED", "N", "Y", FatalDetail

    RecentActivity = LatestActivityTime(WorkDir)
    if VaspState == "stuck":
        if RecentActivity and (time.time() - RecentActivity) <= 2 * 3600:
            return "RUNNING", "N", "N", "OUTCAR empty, VASP starting up"
        return "STUCK", "N", "N", "OUTCAR empty, no VASP job submitted — restart OxidationMaster"

    if RecentActivity and (time.time() - RecentActivity) <= 2 * 3600:
        return "RUNNING", "N", "N", "recent file activity"
    if not StepFolders:
        return "NOT_STARTED", "N", "N", "no step folders"
    return "UNKNOWN", "N", "N", "no done/failure marker"


def LatestActivityTime(WorkDir: Path) -> Optional[float]:
    """Return latest mtime for files/folders relevant to the simulation."""
    Times: List[float] = []
    for Name in ("RateAnalysis.csv", "jobsub.log"):
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


def BuildRow(Label: str, WorkDir: Path) -> SimulationRow:
    """Build one summary row."""
    StepFolders = NumericStepFolders(WorkDir)
    FatalDetail = LatestFatalDetail([WorkDir.parent / "log.out", WorkDir / "jobsub.log"])
    Status, Done, Failed, Detail = DetermineStatus(WorkDir, StepFolders, FatalDetail)
    RateRows, SimTime, TotalO2Added, MoleculesRemoved = ReadRateAnalysis(WorkDir)

    return SimulationRow(
        Simulation=Label,
        Status=Status,
        Folders=str(len(StepFolders)) if WorkDir.exists() else "-",
        Latest=str(StepFolders[-1]) if StepFolders else "-",
        RateRows=RateRows,
        SimTime_ps=SimTime,
        TotalO2Added=TotalO2Added,
        MoleculesRemoved=MoleculesRemoved,
        WallTime=EstimateWallTime(WorkDir, StepFolders),
        Done=Done,
        Failed=Failed,
        Detail=Detail,
    )


def BuildSummary(
    RootDir: Path,
    OxParamsPath: Optional[Path] = None,
) -> List[SimulationRow]:
    """Scan RootDir and return all summary rows."""
    RootDir = RootDir.resolve()
    Simulations = DiscoverSimulations(RootDir, OxParamsPath)
    return [BuildRow(Label, WorkDir) for Label, WorkDir in Simulations]


def Truncate(Value: Optional[str], MaxLen: int) -> str:
    """Truncate text for the fixed-width table."""
    if not Value:
        return "-"
    if len(Value) <= MaxLen:
        return Value
    return Value[: MaxLen - 3] + "..."


def FormatTable(Rows: Sequence[SimulationRow]) -> str:
    """Return a fixed-width table suitable for terminal inspection."""
    Values = [[getattr(Row, Field) for Field, _ in SUMMARY_COLUMNS] for Row in Rows]
    Widths = []
    for Index, (_, Display) in enumerate(SUMMARY_COLUMNS):
        Widths.append(max(len(Display), *(len(str(Row[Index])) for Row in Values)) if Values else len(Display))

    Lines = []
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
) -> None:
    """Build and write one summary refresh."""
    Rows = BuildSummary(RootDir, OxParamsPath)
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

    WriteSummaryOnce(RootDir, OxParamsPath, Parsed.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
