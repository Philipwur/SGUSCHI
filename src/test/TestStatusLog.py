"""Tests for the unified per-simulation event log (utils/StatusLog.py)."""

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

from utils import StatusLog  # noqa: E402


@pytest.fixture(name="WorkDir")
def FixtureWorkDir() -> Path:
    """Create a repo-local temp directory."""
    Dir = Path.cwd() / ".test_tmp_statuslog" / uuid.uuid4().hex
    Dir.mkdir(parents=True)
    try:
        yield Dir
    finally:
        shutil.rmtree(Dir, ignore_errors=True)


def TestAppendThenReadRoundTrip(WorkDir: Path) -> None:
    """Events round-trip in order with source/event/detail preserved."""
    StatusLog.Append(WorkDir, "SGUSCHI", "started")
    StatusLog.Append(WorkDir, "volsearch_cont", "submitted", "297 12345")

    Events = StatusLog.ReadEvents(WorkDir)

    assert [E.Event for E in Events] == ["started", "submitted"]
    assert Events[1].Source == "volsearch_cont"
    assert Events[1].Detail == "297 12345"


def TestMalformedLinesAreSkipped(WorkDir: Path) -> None:
    """Bad timestamp / too-few-field lines are skipped; valid lines survive."""
    (WorkDir / StatusLog.SIM_LOG).write_text(
        "not-a-timestamp\tx\ty\n"                      # unparseable timestamp
        "2026-06-03T07:00:00\tSGUSCHI\tstarted\n"      # valid (3 fields, no detail)
        "tooFewFields\n",                              # < 3 fields
        encoding="utf-8",
    )

    Events = StatusLog.ReadEvents(WorkDir)

    assert len(Events) == 1
    assert Events[0].Event == "started"
    assert Events[0].Detail == ""


def TestEmbeddedTabsAndNewlinesAreCollapsed(WorkDir: Path) -> None:
    """Detail with tabs/newlines stays a single parseable line."""
    StatusLog.Append(WorkDir, "SGUSCHI", "killed", "line1\tcol\nline2")

    Events = StatusLog.ReadEvents(WorkDir)

    assert len(Events) == 1
    assert "\t" not in Events[0].Detail
    assert "\n" not in Events[0].Detail


def TestReadEventsMissingFileReturnsEmpty(WorkDir: Path) -> None:
    """No log file yet → empty list (no error)."""
    assert StatusLog.ReadEvents(WorkDir) == []


def TestLastEventReturnsMostRecent(WorkDir: Path) -> None:
    """LastEvent returns the latest matching event."""
    StatusLog.Append(WorkDir, "SGUSCHI", "exit", "0")
    StatusLog.Append(WorkDir, "SGUSCHI", "exit", "1")

    Last = StatusLog.LastEvent(WorkDir, "exit")

    assert Last is not None
    assert Last.Detail == "1"
    assert StatusLog.LastEvent(WorkDir, "nope") is None


def TestAppendToMissingDirDoesNotRaise(WorkDir: Path) -> None:
    """Appending under a non-existent directory is swallowed, never raises."""
    Missing = WorkDir / "does_not_exist"

    StatusLog.Append(Missing, "SGUSCHI", "started")  # must not raise

    assert not (Missing / StatusLog.SIM_LOG).exists()


def TestCliAppend(WorkDir: Path) -> None:
    """The append CLI writes a parseable event."""
    Code = StatusLog.Main(
        ["append", "--workdir", str(WorkDir), "--source", "volsearch_cont",
         "--event", "submitted", "--detail", "5 9090"]
    )

    assert Code == 0
    Events = StatusLog.ReadEvents(WorkDir)
    assert len(Events) == 1
    assert Events[0].Event == "submitted"
    assert Events[0].Detail == "5 9090"
