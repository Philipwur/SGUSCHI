"""Create a terminal-readable summary of SGUSCHI simulation folders.

The scanner is intentionally independent from the Slurm controller.  It can be
called by today's shell master script or imported by a future Python master.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


SUMMARY_TXT = "SimulationSummary.txt"
SUMMARY_TSV = "SimulationSummary.tsv"
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
    SimTime_fs: str
    WallTime: str
    Done: str
    Failed: str
    Detail: str


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


def DiscoverSimulations(RootDir: Path) -> List[Tuple[str, Path]]:
    """Find simulation folders under RootDir."""
    Expected = ReadExpectedSimulations(RootDir)
    Found = {Label: WorkDir for Label, WorkDir in Expected}

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


def ReadRateAnalysis(WorkDir: Path) -> Tuple[str, str]:
    """Return row count and last Time (fs) without importing pandas."""
    RatePath = WorkDir / "RateAnalysis.csv"
    if not RatePath.exists():
        return "-", "-"

    try:
        with RatePath.open("r", encoding="utf-8-sig", errors="ignore", newline="") as File:
            Reader = csv.DictReader(File)
            Rows = list(Reader)
    except Exception:
        return "ERR", "-"

    if not Rows:
        return "0", "-"

    TimeValue = "-"
    for Key in ("Time (fs)", "Time", "Time_fs"):
        Raw = Rows[-1].get(Key)
        if Raw not in (None, ""):
            TimeValue = FormatFloat(Raw)
            break
    return str(len(Rows)), TimeValue


def FormatFloat(Value: str) -> str:
    """Format a numeric string compactly, preserving non-numeric values."""
    try:
        Number = float(Value)
    except (TypeError, ValueError):
        return str(Value)
    if Number.is_integer():
        return str(int(Number))
    return f"{Number:.3f}".rstrip("0").rstrip(".")


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

    if FailedMarker.exists():
        return "FAILED", "N", "Y", "sguschi_failed"
    if FatalDetail:
        return "FAILED", "N", "Y", FatalDetail
    if DoneMarker.exists():
        return "DONE", "Y", "N", "volsearch_is_done"

    RecentActivity = LatestActivityTime(WorkDir)
    if RecentActivity and (time.time() - RecentActivity) <= 2 * 3600:
        return "RUNNING", "N", "N", "recent file activity"
    if not StepFolders:
        return "NOT_STARTED", "N", "N", "no step folders"
    return "UNKNOWN", "N", "N", "no done/failure marker"


def LatestActivityTime(WorkDir: Path) -> Optional[float]:
    """Return latest mtime for files/folders relevant to the simulation."""
    Times: List[float] = []
    for Name in ("OUTCAR", "jobsub.log", "RateAnalysis.csv"):
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


def BuildRow(Label: str, WorkDir: Path) -> SimulationRow:
    """Build one summary row."""
    StepFolders = NumericStepFolders(WorkDir)
    FatalDetail = LatestFatalDetail([WorkDir.parent / "log.out", WorkDir / "jobsub.log"])
    Status, Done, Failed, Detail = DetermineStatus(WorkDir, StepFolders, FatalDetail)
    RateRows, SimTime = ReadRateAnalysis(WorkDir)

    return SimulationRow(
        Simulation=Label,
        Status=Status,
        Folders=str(len(StepFolders)) if WorkDir.exists() else "-",
        Latest=str(StepFolders[-1]) if StepFolders else "-",
        RateRows=RateRows,
        SimTime_fs=SimTime,
        WallTime=EstimateWallTime(WorkDir, StepFolders),
        Done=Done,
        Failed=Failed,
        Detail=Detail,
    )


def BuildSummary(RootDir: Path) -> List[SimulationRow]:
    """Scan RootDir and return all summary rows."""
    RootDir = RootDir.resolve()
    Simulations = DiscoverSimulations(RootDir)
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
    Headers = [
        "Simulation",
        "Status",
        "Folders",
        "Latest",
        "RateRows",
        "SimTime_fs",
        "WallTime",
        "Done",
        "Failed",
        "Detail",
    ]
    Values = [[getattr(Row, Header) for Header in Headers] for Row in Rows]
    Widths = []
    for Index, Header in enumerate(Headers):
        Widths.append(max(len(Header), *(len(str(Row[Index])) for Row in Values)) if Values else len(Header))

    Lines = []
    Lines.append("  ".join(Header.ljust(Widths[Index]) for Index, Header in enumerate(Headers)))
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

    AtomicWriteText(TextPath, FormatTable(Rows))

    Headers = list(SimulationRow.__dataclass_fields__.keys())
    Lines = ["\t".join(Headers)]
    for Row in Rows:
        Lines.append("\t".join(str(getattr(Row, Header)) for Header in Headers))
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


def ParseArgs(Args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    Parser = argparse.ArgumentParser(description=__doc__)
    Parser.add_argument(
        "RootDir",
        nargs="?",
        default=".",
        help="Root directory containing simulation folders such as 873_2.",
    )
    return Parser.parse_args(Args)


def Main(Args: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    Parsed = ParseArgs(Args)
    RootDir = Path(Parsed.RootDir).resolve()
    if not RootDir.exists():
        print(f"Root directory does not exist: {RootDir}", file=sys.stderr)
        return 1
    Rows = BuildSummary(RootDir)
    WriteOutputs(RootDir, Rows)
    print(f"Wrote {RootDir / SUMMARY_TXT}")
    print(f"Wrote {RootDir / SUMMARY_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
