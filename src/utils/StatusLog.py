"""Unified, append-only per-simulation event log.

One log file lives in each ``Dir_VolSearch``: ``sim_log.tsv``.  Every producer
(SGUSCHI.py, volsearch_cont via the CLI below) appends lifecycle events through
``Append`` so the SimulationSummary scanner has a single source for status and
queue timing instead of several scattered marker files.

Design notes:
- Stdlib only on purpose (same posture as ``utils/FolderUtils.py``): this is a
  leaf module imported by the heavy orchestrator *and* the lightweight, no-pandas
  summary scanner, so it must not pull in third-party deps or create cycles.
- Writes take an exclusive ``flock`` on POSIX (the cluster) so concurrent writers
  never interleave a line.  Platforms without ``fcntl`` (Windows dev/test) fall
  back to a plain append; there is no cross-process contention there in practice.
- Every write is wrapped so a logging failure can **never** raise into the
  caller — the log is advisory and must not affect simulation control flow
  (the same contract the csh layer already applies to ``.submit_times``).

Line format (tab-separated, one event per line)::

    <ISO8601 timestamp>\t<source>\t<event>\t<detail>

``detail`` is optional and may be empty.  Tabs/newlines inside any field are
collapsed to spaces so the one-line-per-event invariant always holds.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

# Filename used in every Dir_VolSearch. Imported by SimulationSummary so the
# producer and consumer never disagree on the path.
SIM_LOG = "sim_log.tsv"


# ---------------------------------------------------------------------------
# Cross-process locking (best-effort, POSIX flock when available)
# ---------------------------------------------------------------------------
try:
    import fcntl

    def _Lock(FileObj) -> None:
        fcntl.flock(FileObj.fileno(), fcntl.LOCK_EX)

    def _Unlock(FileObj) -> None:
        fcntl.flock(FileObj.fileno(), fcntl.LOCK_UN)

except ImportError:  # Windows dev/test — no fcntl; single-process append is fine.

    def _Lock(FileObj) -> None:
        pass

    def _Unlock(FileObj) -> None:
        pass


@dataclass
class LogEvent:
    """One parsed event line."""

    Timestamp: datetime
    Source: str
    Event: str
    Detail: str


def _Clean(Value: object) -> str:
    """Collapse tabs/newlines so a field never breaks the one-line TSV format."""
    return str(Value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def Append(WorkDir, Source: str, Event: str, Detail: str = "") -> None:
    """Append one event to ``WorkDir/sim_log.tsv``. Never raises.

    The timestamp is generated here (write time), which for a ``submitted`` event
    is the submission time used for queue-time accounting.
    """
    try:
        LogPath = Path(WorkDir) / SIM_LOG
        Timestamp = datetime.now().isoformat(timespec="seconds")
        Line = "\t".join(
            (Timestamp, _Clean(Source), _Clean(Event), _Clean(Detail))
        )
        with LogPath.open("a", encoding="utf-8") as FileObj:
            _Lock(FileObj)
            try:
                FileObj.write(Line + "\n")
                FileObj.flush()
            finally:
                _Unlock(FileObj)
    except Exception:
        # Advisory log: a failure must never disturb the caller's control flow.
        pass


def ReadEvents(WorkDir) -> List[LogEvent]:
    """Return ordered events from ``WorkDir/sim_log.tsv``.

    Malformed lines (too few fields, unparseable timestamp) are skipped so a
    partially written or corrupt file degrades to fewer events rather than an
    error. Missing file → empty list.
    """
    LogPath = Path(WorkDir) / SIM_LOG
    Events: List[LogEvent] = []
    try:
        Text = LogPath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return Events

    for Line in Text.splitlines():
        if not Line.strip():
            continue
        Parts = Line.split("\t")
        if len(Parts) < 3:
            continue
        try:
            Timestamp = datetime.fromisoformat(Parts[0])
        except ValueError:
            continue
        Detail = Parts[3] if len(Parts) > 3 else ""
        Events.append(LogEvent(Timestamp, Parts[1], Parts[2], Detail))
    return Events


def LastEvent(WorkDir, Event: str) -> Optional[LogEvent]:
    """Return the most recent event with the given name, or None."""
    Match: Optional[LogEvent] = None
    for Item in ReadEvents(WorkDir):
        if Item.Event == Event:
            Match = Item
    return Match


# ---------------------------------------------------------------------------
# CLI — lets the csh layer append without embedding lock logic.
# ---------------------------------------------------------------------------

def ParseArgs(Args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the ``append`` subcommand."""
    Parser = argparse.ArgumentParser(description=__doc__)
    Sub = Parser.add_subparsers(dest="command")
    AppendParser = Sub.add_parser("append", help="Append one event to the log.")
    AppendParser.add_argument("--workdir", default=".", help="Dir_VolSearch (default: cwd)")
    AppendParser.add_argument("--source", required=True)
    AppendParser.add_argument("--event", required=True)
    AppendParser.add_argument("--detail", default="")
    return Parser.parse_args(Args)


def Main(Args: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    Parsed = ParseArgs(Args)
    if Parsed.command == "append":
        Append(Parsed.workdir, Parsed.source, Parsed.event, Parsed.detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
