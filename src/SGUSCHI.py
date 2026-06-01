"""
SGUSCHI.py — unified workspace setup and simulation orchestration.

Called by OxidationMaster (the user-customised Slurm submission script) on the
compute node. Handles everything from folder creation to running volsearch_cont.

Usage (called by OxidationMaster):
    python /path/to/SGUSCHI/src/SGUSCHI.py [WorkDir] [--dry-run] [--prepare-only]

WorkDir defaults to the current working directory (the simulation workspace).

Re-run behaviour:
    - Folders that already exist are never recreated or overwritten.
    - Simulations marked done (volsearch_is_done or job.exit=0) are skipped.
    - New temperatures/NSims added to OxParams are set up on next submission.
    - Extra folders on disk not in OxParams are never touched.
"""

import argparse
import ast
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path constants — derived from this file's location (src/SGUSCHI.py)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent          # .../src/
REPO_ROOT = SRC_DIR.parent                         # repo root
VOLSEARCH_CONT = REPO_ROOT / "src" / "dependencies" / "SLUSCHI_mod" / "volsearch_cont"
SIMULATION_SUMMARY_SCRIPT = REPO_ROOT / "src" / "utils" / "SimulationSummary.py"
WORKSPACE_SETUP_MODULE = REPO_ROOT / "src" / "preprocessing" / "WorkSpaceSetup.py"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def ParseArgs() -> argparse.Namespace:
    Parser = argparse.ArgumentParser(
        description="SGUSCHI workspace setup and orchestration entry point."
    )
    Parser.add_argument(
        "workdir",
        nargs="?",
        default=".",
        help="Simulation workspace directory (default: current directory)",
    )
    Parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without creating folders or launching jobs",
    )
    Parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create simulation folders only; do not submit VASP jobs or launch "
             "volsearch_cont (safe to run on a login node)",
    )
    return Parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def ReadOxParams(WorkDir: Path) -> dict:
    """Read and validate OxParams from WorkDir. Exits on error."""
    OxParamsPath = WorkDir / "OxParams"
    if not OxParamsPath.exists():
        print("ERROR: OxParams not found in {}".format(WorkDir))
        sys.exit(1)

    # Import VaspIO from the bundled src tree
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from workflow import VaspIO as vio
    except ImportError as E:
        print("ERROR: could not import VaspIO: {}".format(E))
        sys.exit(1)

    RequiredKeys = [
        "Temperatures",
        "NSims",
        "GasRatio",
        "InitO2Count",
        "AtomicRadiusTol",
        "O2Tol",
        "OSmoothing",
    ]
    try:
        Params = vio.ReadKeyValueFile(OxParamsPath, RequiredKeys=RequiredKeys)
    except Exception as E:
        print("ERROR reading OxParams: {}".format(E))
        sys.exit(1)
    return Params


# ---------------------------------------------------------------------------
# Simulation directory helpers
# ---------------------------------------------------------------------------

def GetSimulationDirs(WorkDir: Path, Params: dict) -> List[Tuple[str, Path]]:
    """Return [(label, Dir_VolSearch_path), ...] from Temperatures × NSims."""
    Temperatures = ast.literal_eval(Params["Temperatures"])
    NSims = int(Params["NSims"])
    Result = []
    for Temp in Temperatures:
        for SimIdx in range(1, NSims + 1):
            Label = "{}_{}".format(Temp, SimIdx)
            VolSearchDir = WorkDir / Label / "Dir_VolSearch"
            Result.append((Label, VolSearchDir))
    return Result


def ClassifySimulations(WorkDir: Path, Params: dict) -> Dict[str, str]:
    """Return {label: state} for every expected simulation.

    States:
        'new'     — Dir_VolSearch does not exist; needs setup + initial VASP job
        'pending' — set up but not yet done (includes failed/killed; will be retried)
        'done'    — volsearch_is_done marker or job.exit=0 present
    """
    States: Dict[str, str] = {}
    for Label, VolSearchDir in GetSimulationDirs(WorkDir, Params):
        if not VolSearchDir.exists():
            States[Label] = "new"
            continue
        if (VolSearchDir / "volsearch_is_done").exists():
            States[Label] = "done"
            continue
        ExitFile = VolSearchDir / "job.exit"
        if ExitFile.exists():
            try:
                RC = int(ExitFile.read_text(encoding="utf-8").strip())
                States[Label] = "done" if RC == 0 else "pending"
            except (ValueError, OSError):
                States[Label] = "pending"
            continue
        States[Label] = "pending"
    return States


# ---------------------------------------------------------------------------
# Scheduler command
# ---------------------------------------------------------------------------

def ReadSchedulerCmd(JobInDir: Path) -> List[str]:
    """Return vaspcmd from JobInDir/job.in as a tokenised argv list.

    Falls back to ['sbatch'] if job.in is absent or vaspcmd is not set.
    Example: vaspcmd = sbatch --parsable  →  returns ['sbatch', '--parsable']
    """
    JobInPath = JobInDir / "job.in"
    if not JobInPath.exists():
        return ["sbatch"]
    try:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        from workflow import VaspIO as vio
        Params = vio.ReadKeyValueFile(JobInPath)
        RawCmd = Params.get("vaspcmd", "").strip()
        if RawCmd:
            return shlex.split(RawCmd)
    except Exception:
        pass
    return ["sbatch"]


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def RunSetup(WorkDir: Path, Params: dict, NewLabels: List[str]) -> None:
    """Create simulation folders for NewLabels by calling SetupWorkspace."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    PreprocessingDir = str(SRC_DIR / "preprocessing")
    if PreprocessingDir not in sys.path:
        sys.path.insert(0, PreprocessingDir)

    try:
        from WorkSpaceSetup import SetupWorkspace
    except ImportError as E:
        print("ERROR: could not import SetupWorkspace: {}".format(E))
        sys.exit(1)

    Temperatures = ast.literal_eval(Params["Temperatures"])
    NSims = int(Params["NSims"])

    # Build (Temp, SimIdx) pairs that correspond to NewLabels
    OnlySimIndices: List[Tuple[int, int]] = []
    for Temp in Temperatures:
        for SimIdx in range(1, NSims + 1):
            Label = "{}_{}".format(Temp, SimIdx)
            if Label in NewLabels:
                OnlySimIndices.append((Temp, SimIdx))

    print("Setting up {} new simulation folder(s)...".format(len(OnlySimIndices)))
    try:
        LogLines = SetupWorkspace(WorkDir, Params, OnlySimIndices=OnlySimIndices)
    except Exception as E:
        print("ERROR during workspace setup: {}".format(E))
        sys.exit(1)

    for Line in LogLines:
        print("  " + Line)


# ---------------------------------------------------------------------------
# Initial VASP job submission
# ---------------------------------------------------------------------------

def NeedsInitialVaspJob(VolSearchDir: Path) -> bool:
    """Return True if the initial VASP job should be submitted.

    Uses the same completion signal as volsearch_cont: 'Total CPU' in OUTCAR.
      - No OUTCAR           → job not yet submitted; submit
      - OUTCAR empty        → job submitted or starting; do not re-submit
      - OUTCAR has Total CPU → job finished; volsearch_cont will handle next submission
      - Step folders exist  → volsearch_cont already advanced; do not submit
    """
    # Check for numeric step folders (volsearch_cont has already run at least one cycle)
    try:
        HasStepFolders = any(
            C.name.isdigit() for C in VolSearchDir.iterdir() if C.is_dir()
        )
    except OSError:
        return False
    if HasStepFolders:
        return False

    Outcar = VolSearchDir / "OUTCAR"
    if not Outcar.exists():
        return True                           # fresh folder, nothing submitted yet
    try:
        if Outcar.stat().st_size == 0:
            return False                      # empty OUTCAR = job queued or running
        Content = Outcar.read_text(encoding="utf-8", errors="ignore")
        return "Total CPU" not in Content     # done iff Total CPU line is present
    except OSError:
        return False


def SubmitInitialVaspJobs(WorkDir: Path, Params: dict, NewLabels: List[str]) -> Set[str]:
    """Submit the initial VASP job in each newly-created Dir_VolSearch that needs it.

    Returns the set of labels whose initial submission failed. Failed dirs are
    marked with job.exit=-1 so the caller can exclude them from volsearch_cont
    launch and avoid the master walltime being burnt waiting on a job that was
    never queued.
    """
    Failed: Set[str] = set()
    for Label, VolSearchDir in GetSimulationDirs(WorkDir, Params):
        if Label not in NewLabels:
            continue
        if not VolSearchDir.exists():
            continue
        if not NeedsInitialVaspJob(VolSearchDir):
            print("  [{}] initial VASP job already submitted or done — skipped".format(Label))
            continue
        VaspCmd = ReadSchedulerCmd(VolSearchDir)
        Argv = VaspCmd + ["jobsub"]
        Display = " ".join(VaspCmd)
        try:
            Result = subprocess.run(
                Argv,
                cwd=str(VolSearchDir),
                capture_output=True,
                text=True,
                check=False,
            )
            print("  [{}] {} jobsub → exit {}".format(Label, Display, Result.returncode))
            if Result.stdout.strip():
                print("  [{}]   {}".format(Label, Result.stdout.strip()))
            if Result.returncode != 0:
                if Result.stderr.strip():
                    print("  [{}]   stderr: {}".format(Label, Result.stderr.strip()))
                WriteMarker(VolSearchDir / "job.exit", "-1")
                Failed.add(Label)
        except FileNotFoundError:
            print("  [{}] ERROR: '{}' not on PATH — initial VASP submission failed".format(
                Label, VaspCmd[0]))
            WriteMarker(VolSearchDir / "job.exit", "-1")
            Failed.add(Label)
        except Exception as E:
            print("  [{}] ERROR: initial VASP submission failed: {!r}".format(Label, E))
            WriteMarker(VolSearchDir / "job.exit", "-1")
            Failed.add(Label)
    return Failed


# ---------------------------------------------------------------------------
# Job markers
# ---------------------------------------------------------------------------

def WriteMarker(MarkerPath: Path, Content: str = "") -> None:
    """Write a marker file atomically (write to .tmp then rename)."""
    TmpPath = MarkerPath.with_suffix(".tmp")
    try:
        MarkerPath.parent.mkdir(parents=True, exist_ok=True)
        TmpPath.write_text(Content, encoding="utf-8")
        TmpPath.replace(MarkerPath)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _MakeSigtermHandler(Procs: Dict[str, Tuple[subprocess.Popen, Path]]):
    """Return a SIGTERM handler that cleans up all running volsearch_cont processes."""
    def Handler(SigNum, Frame):
        print("\nSGUSCHI: received SIGTERM — stopping all simulations")
        for Label, (Proc, Vsd) in Procs.items():
            if Proc.poll() is None:
                Proc.terminate()
                WriteMarker(Vsd / "job.killed",
                            "killed by SIGTERM at {}".format(datetime.now().isoformat()))
                print("  [{}] terminated + job.killed written".format(Label))
            else:
                RC = Proc.returncode
                WriteMarker(Vsd / "job.exit", str(RC))
        sys.exit(1)
    return Handler


def RunOrchestration(WorkDir: Path, Params: dict, PendingDirs: List[Tuple[str, Path]]) -> int:
    """Launch volsearch_cont in parallel for all pending simulation dirs.

    Runs on the compute node. Blocks until all processes finish.
    Writes job.started / job.exit / job.killed markers in each Dir_VolSearch.
    """
    # Validate binary
    if not VOLSEARCH_CONT.exists():
        print("ERROR: volsearch_cont not found at {}".format(VOLSEARCH_CONT))
        print("Run 'make' in src/dependencies/SLUSCHI_mod/ first.")
        return 1
    if not os.access(str(VOLSEARCH_CONT), os.X_OK):
        print("ERROR: volsearch_cont is not executable.")
        print("Run: chmod +x {}".format(VOLSEARCH_CONT))
        return 1

    Procs: Dict[str, Tuple[subprocess.Popen, Path]] = {}
    LaunchFailures = 0

    # Tell the bundled volsearch_cont (and its csh helpers) where to find
    # sluschipath, so the flow is self-contained and does not rely on
    # ~/.sluschi.rc pointing at the in-repo SLUSCHI_mod directory.
    Env = os.environ.copy()
    Env["sguschipath"] = str(VOLSEARCH_CONT.parent)

    for Label, Vsd in PendingDirs:
        # Clear stale terminal markers from a previous attempt so
        # SimulationSummary doesn't report the restarted sim as FAILED/KILLED
        # before volsearch_cont rewrites them at exit. (volsearch_cont also
        # removes sguschi_failed at startup; clearing it here closes the brief
        # window before that and keeps the restart cleanup consistent.)
        for Stale in ("job.exit", "job.killed", "sguschi_failed"):
            (Vsd / Stale).unlink(missing_ok=True)
        WriteMarker(Vsd / "job.started", datetime.now().isoformat())
        LogPath = Vsd.parent / "log.out"
        try:
            LogFile = open(str(LogPath), "a", encoding="utf-8")
            Proc = subprocess.Popen(
                [str(VOLSEARCH_CONT)],
                cwd=str(Vsd),
                stdout=LogFile,
                stderr=subprocess.STDOUT,
                env=Env,
            )
            Procs[Label] = (Proc, Vsd)
            print("  [{}] volsearch_cont started (pid {})".format(Label, Proc.pid))
        except OSError as E:
            print("  [{}] ERROR launching volsearch_cont: {}".format(Label, E))
            WriteMarker(Vsd / "job.exit", "-1")
            LaunchFailures += 1

    if not Procs:
        print("No volsearch_cont processes started.")
        return 1

    # Register SIGTERM handler so markers are written on walltime kill
    signal.signal(signal.SIGTERM, _MakeSigtermHandler(Procs))

    # Start SimulationSummary daemon
    if SIMULATION_SUMMARY_SCRIPT.exists():
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    str(SIMULATION_SUMMARY_SCRIPT),
                    str(WorkDir),
                    "--watch-daemon",
                    "--interval", "60",
                    "--oxparams", str(WorkDir / "OxParams"),
                    "--quiet",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass  # daemon is optional; don't abort if it fails

    # Wait for all processes
    AllPassed = LaunchFailures == 0
    for Label, (Proc, Vsd) in Procs.items():
        RC = Proc.wait()
        WriteMarker(Vsd / "job.exit", str(RC))
        Status = "OK" if RC == 0 else "FAILED (exit {})".format(RC)
        print("  [{}] volsearch_cont finished — {}".format(Label, Status))
        if RC != 0:
            AllPassed = False

    return 0 if AllPassed else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # Ensure non-ASCII status characters (em-dashes, arrows) survive on shells
    # whose default encoding is not UTF-8 (notably Windows PowerShell, cp1252).
    for Stream in (sys.stdout, sys.stderr):
        if hasattr(Stream, "reconfigure"):
            try:
                Stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    Args = ParseArgs()
    WorkDir = Path(Args.workdir).resolve()

    if not WorkDir.exists():
        print("ERROR: WorkDir does not exist: {}".format(WorkDir))
        return 1

    print("SGUSCHI — workspace: {}".format(WorkDir))

    Params = ReadOxParams(WorkDir)
    States = ClassifySimulations(WorkDir, Params)

    NewLabels = [L for L, S in States.items() if S == "new"]
    PendingLabels = [L for L, S in States.items() if S in ("new", "pending")]
    DoneLabels = [L for L, S in States.items() if S == "done"]

    print("Simulations: {} new, {} pending, {} done".format(
        len(NewLabels), len(PendingLabels) - len(NewLabels), len(DoneLabels)))

    if Args.dry_run:
        if NewLabels:
            print("Would create folders: {}".format(", ".join(NewLabels)))
        PendingDirsDry = [
            (L, Vsd) for L, Vsd in GetSimulationDirs(WorkDir, Params)
            if L in PendingLabels
        ]
        if PendingDirsDry:
            print("Would start volsearch_cont in: {}".format(
                ", ".join(L for L, _ in PendingDirsDry)))
        return 0

    # Setup new folders
    FailedLabels: Set[str] = set()
    if Args.prepare_only:
        if NewLabels:
            RunSetup(WorkDir, Params, NewLabels)
            print("Prepared {} new simulation folder(s).".format(len(NewLabels)))
        else:
            print("No new simulation folders to prepare.")
        print("--prepare-only: skipping VASP job submission and volsearch_cont launch.")
        return 0

    if NewLabels:
        RunSetup(WorkDir, Params, NewLabels)
        FailedLabels = SubmitInitialVaspJobs(WorkDir, Params, NewLabels)
        if FailedLabels:
            print("Initial VASP submission failed in: {}".format(", ".join(sorted(FailedLabels))))
            print("  → these directories will NOT run volsearch_cont this session.")

    # Collect pending dirs (after setup, new folders are now on disk).
    # Exclude any label whose initial VASP submission failed — otherwise
    # volsearch_cont would sit polling an OUTCAR that never arrives.
    PendingDirs = [
        (L, Vsd) for L, Vsd in GetSimulationDirs(WorkDir, Params)
        if L in PendingLabels and L not in FailedLabels
    ]

    if not PendingDirs:
        if FailedLabels:
            print("All eligible simulations failed initial submission. Aborting.")
            return 1
        print("All simulations are already done. Nothing to run.")
        return 0

    print("Starting volsearch_cont for {} simulation(s)...".format(len(PendingDirs)))
    RC = RunOrchestration(WorkDir, Params, PendingDirs)
    # Treat the overall run as failed if any initial submission also failed.
    return 1 if (RC != 0 or FailedLabels) else 0


if __name__ == "__main__":
    raise SystemExit(main())
