"""
ResumeFromTrajectory.py — rebuild a runnable workspace from ``xyz_files/`` output.

Continue an interrupted oxidation simulation when the ONLY surviving artifact is the
workspace's primary output folder ``xyz_files/`` (one ``{Temp}_{Sim}.xyz`` trajectory
plus its ``RateAnalysis_{Temp}_{Sim}.csv`` mirror per replica), together with the
standard input files (POTCAR, OxParams, CovalentRadii, INCAR, KPOINTS, job.in, jobsub).
The numbered per-step folders inside ``Dir_VolSearch`` and their OUTCAR/WAVECAR are gone.

Full history is preserved (no truncated re-seed): the resumed run keeps the entire
``RateAnalysis`` and keeps *appending* to the existing ``.xyz``. This works because two
independent step-counters are reconstructed in lockstep:

  * ``volsearch_cont`` numbers the next step folder at startup by counting contiguous
    numeric folders on disk (``nstep = highest existing + 1``); their contents are never
    re-read, so EMPTY placeholder folders ``1..N`` suffice.
  * ``OxidationStep.py`` reads folder ``len(RateAnalysis)``; keeping the full N+1-row
    history makes it read the next (real) folder ``N+1``.

No Fortran/csh changes are needed: a rebuilt sim is classified ``pending`` by SGUSCHI,
and ``volsearch_cont``'s startup-recovery branch submits the reconstructed POSCAR as
step ``N+1``. This utility only prepares the workspace; resume with the normal
``sbatch OxidationMaster``.

Usage:
    The simplest way is to ``cd`` into the workspace and run with no path arguments —
    the paths are inferred from the current directory (always dry-run first):

        cd /path/to/workspace                 # the folder containing xyz_files/
        python /path/to/SGUSCHI/src/utils/ResumeFromTrajectory.py --dry-run
        python /path/to/SGUSCHI/src/utils/ResumeFromTrajectory.py   # actually build

    You may instead ``cd`` into the ``xyz_files/`` folder itself; the parent workspace
    is then used for inputs and as the rebuild target. Inference rules:
      * cwd contains ``xyz_files/``  -> cwd is the workspace root (inputs + target)
      * cwd holds the ``*.xyz`` files -> cwd is xyz_files/, its parent is the root
      * neither                       -> pass --target and --inputs explicitly

    Any path can be overridden explicitly (e.g. inputs kept elsewhere):

        python .../ResumeFromTrajectory.py --target WORKDIR --inputs INPUTS_DIR \
            [--xyz-dir DIR] [--only 1273_3,873_1] [--force] [--no-summary] [--dry-run]

    --inputs must hold: POTCAR OxParams CovalentRadii INCAR KPOINTS job.in jobsub.
    By default every {Temp}_{Sim}.xyz found is rebuilt; --only narrows the set.
    This utility only prepares the workspace — resume with ``sbatch OxidationMaster``.

Accepted caveats (physically minor, self-healing):
    * The reconstructed POSCAR is the raw last xyz frame, so it omits the single gas
      removal / O2 addition OxidationStep would apply at the N->N+1 boundary. The next
      cycle re-evaluates gases and self-heals within one segment (<=1 molecule).
    * The POSCAR carries no velocities; VASP re-thermalizes to TEBEG (brief transient).

Limitation (ENFORCED via a ``.resume_seam`` marker + guards in the repair tools):
    After resume, folders ``1..N`` are empty placeholders, so FixXYZ, FixRateAnalysis,
    and RollbackTrajectory cannot operate at or before step N.
"""

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make imports location-independent (mirror the other utils/workflow modules).
sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio  # noqa: E402
from utils.FolderUtils import NumericStepFolders, RESUME_SEAM_MARKER  # noqa: E402

# Files copied into each SimDir and Dir_VolSearch (same set as WorkSpaceSetup).
SIM_INPUT_FILES = ["POTCAR", "KPOINTS", "INCAR", "job.in", "jobsub"]
# Files that live at the workspace root (RootDir), read by OxidationStep.
ROOT_INPUT_FILES = ["OxParams", "CovalentRadii"]
# Stale run-state cleared so volsearch_cont's recovery branch resubmits step N+1.
STALE_RUN_STATE = [
    ".vasp_submitted_step",
    ".vasp_submit_last.out",
    "poscar_built_for_step",
    "volsearch_is_done",
    "maxruntime_reached",
    "job.exit",
    "job.killed",
    "sguschi_failed",
    "awaiting_manual_submission",
    "WAVECAR",
    "CONTCAR",
    "lattice_predict.out",
]

MD_STEPS_PER_CYCLE = 80  # hardcoded in the SLUSCHI binary; one RateAnalysis row per cycle


# ---------------------------------------------------------------------------
# Last-frame XYZ extraction
# ---------------------------------------------------------------------------

@dataclass
class LastFrame:
    """The final trajectory frame, converted to POSCAR-ready fractional coordinates."""
    PositionFrac: pd.DataFrame       # columns ['Element', 'x', 'y', 'z'] (Direct)
    CellDim: pd.DataFrame            # 3x3, columns ['x', 'y', 'z'] (this frame's lattice)
    Step: Optional[int]
    TimeFs: Optional[float]
    NumAtoms: int


def _ParseLatticeMatrix(Comment: str) -> Optional[np.ndarray]:
    """Extract the 3x3 lattice from a Lattice="..." (9 floats) comment line."""
    Match = re.search(r'Lattice="([^"]+)"', Comment)
    if not Match:
        return None
    try:
        Values = [float(X) for X in Match.group(1).split()]
    except ValueError:
        return None
    if len(Values) != 9:
        return None
    return np.array(Values, dtype=float).reshape(3, 3)


def _ReadLastFrameBlock(XyzPath: Path) -> Tuple[str, List[str]]:
    """Return (comment_line, atom_lines) of the final complete frame in an XYZ file.

    A frame is ``natoms`` line, comment line, then ``natoms`` atom lines. Scans the
    whole file once keeping only the last well-formed block (robust and simple; the
    per-frame cost is trivial next to a VASP MD segment).
    """
    LastComment = ""
    LastAtomLines: List[str] = []
    with XyzPath.open("r", encoding="utf-8", errors="ignore") as File:
        while True:
            CountLine = File.readline()
            if not CountLine:
                break
            Stripped = CountLine.strip()
            if not Stripped:
                continue
            try:
                NumAtoms = int(Stripped)
            except ValueError:
                continue
            Comment = File.readline()
            if not Comment:
                break
            AtomLines: List[str] = []
            Truncated = False
            for _ in range(NumAtoms):
                Line = File.readline()
                if not Line:
                    Truncated = True
                    break
                AtomLines.append(Line)
            if Truncated or len(AtomLines) != NumAtoms:
                break  # incomplete trailing frame; keep the last complete one
            LastComment = Comment.strip()
            LastAtomLines = AtomLines
    if not LastAtomLines:
        raise ValueError(f"No complete frame found in {XyzPath}.")
    return LastComment, LastAtomLines


def ParseLastXYZFrame(XyzPath: Path) -> LastFrame:
    """Build a POSCAR-ready fractional-coordinate frame from the LAST xyz frame.

    Uses the final frame's OWN lattice for both the Cartesian->fractional conversion
    and the returned CellDim. (VaspIO.ReadXYZ averages the first+last lattice for its
    CellDim, which must not be reused for a single-frame POSCAR.)
    """
    XyzPath = Path(XyzPath)
    Comment, AtomLines = _ReadLastFrameBlock(XyzPath)

    Lattice = _ParseLatticeMatrix(Comment)
    if Lattice is None:
        raise ValueError(f"Final frame of {XyzPath} has no parseable Lattice=\"...\".")

    Elements: List[str] = []
    Coords: List[List[float]] = []
    for Line in AtomLines:
        Parts = Line.split()
        if len(Parts) < 4:
            raise ValueError(f"Malformed atom line in {XyzPath}: {Line.strip()!r}")
        Elements.append(Parts[0])
        Coords.append([float(Parts[1]), float(Parts[2]), float(Parts[3])])
    CoordArray = np.array(Coords, dtype=float)

    # Coordinate type: honour the Properties tag if present, else auto-detect by
    # magnitude (fractional coords are ~[0,1); Cartesian coords in a real cell exceed
    # 1.2). OxidationStep writes Cartesian ("...:pos:R:3").
    if re.search(r"Properties=\S*\bfrac:R:3", Comment):
        IsCartesian = False
    elif re.search(r"Properties=\S*\bpos:R:3", Comment):
        IsCartesian = True
    else:
        IsCartesian = bool(CoordArray.size and np.max(np.abs(CoordArray)) > 1.2)

    if IsCartesian:
        Frac = CoordArray @ np.linalg.inv(Lattice)
    else:
        Frac = CoordArray

    PositionFrac = pd.DataFrame({
        "Element": Elements,
        "x": Frac[:, 0],
        "y": Frac[:, 1],
        "z": Frac[:, 2],
    })
    CellDim = pd.DataFrame(Lattice, columns=["x", "y", "z"])

    StepMatch = re.search(r"Step\s*=\s*(\d+)", Comment)
    TimeMatch = re.search(r"\bTime(?:fs|_fs|\(fs\))?\s*=\s*([-\d\.Ee+]+)", Comment, re.IGNORECASE)
    Step = int(StepMatch.group(1)) if StepMatch else None
    TimeFs = float(TimeMatch.group(1)) if TimeMatch else None

    return LastFrame(
        PositionFrac=PositionFrac,
        CellDim=CellDim,
        Step=Step,
        TimeFs=TimeFs,
        NumAtoms=len(Elements),
    )


def CountXYZFrames(XyzPath: Path) -> int:
    """Count frames by counting comment lines carrying a Lattice="..." tag."""
    Count = 0
    with Path(XyzPath).open("r", encoding="utf-8", errors="ignore") as File:
        for Line in File:
            if 'Lattice="' in Line:
                Count += 1
    return Count


# ---------------------------------------------------------------------------
# Consistency / precondition checks
# ---------------------------------------------------------------------------

def DetermineN(RatePath: Path, XyzPath: Path, Force: bool = False) -> int:
    """Return N = completed steps = len(RateAnalysis) - 1 (the seed row is step 0).

    Cross-checks against the xyz frame count (~N*80). Aborts on mismatch unless Force.
    """
    RateDf = pd.read_csv(RatePath)
    NRate = len(RateDf) - 1
    if NRate <= 0:
        raise ValueError(
            f"{RatePath} has {len(RateDf)} row(s); nothing to resume (need >= 2). "
            "This looks like a fresh simulation — use WorkSpaceSetup instead."
        )

    Frames = CountXYZFrames(XyzPath)
    NXyz = round(Frames / MD_STEPS_PER_CYCLE)
    if NXyz != NRate:
        Message = (
            f"Step-count mismatch for {XyzPath.name}: RateAnalysis implies N={NRate} "
            f"(len={len(RateDf)}), but the xyz has {Frames} frames (~{NXyz} steps of "
            f"{MD_STEPS_PER_CYCLE}). This usually means a crash mid-cycle / partial write."
        )
        if not Force:
            raise ValueError(Message + " Re-run with --force to proceed using N from RateAnalysis.")
        print(f"WARNING: {Message} Proceeding with N={NRate} (--force).")
    return NRate


def ParsePotcarElements(PotcarPath: Path) -> List[str]:
    """Return the ordered, de-duplicated element list from a POTCAR's TITEL lines.

    Real POTCARs concatenate blocks each beginning ``TITEL  = PAW_PBE Zr_sv 04Jan2005``.
    Element suffixes (_sv, _pv, ...) are stripped, mirroring OutcarParser.
    """
    Elements: List[str] = []
    Seen: set = set()
    with Path(PotcarPath).open("r", encoding="utf-8", errors="ignore") as File:
        for Line in File:
            if "TITEL" not in Line:
                continue
            Payload = Line.split("=", 1)[1] if "=" in Line else Line
            # Skip the pseudopotential flavour token (e.g. PAW_PBE), take the element.
            Tokens = Payload.split()
            El = None
            for Tok in Tokens:
                Match = re.match(r"^([A-Z][a-z]?)(?:_[A-Za-z0-9]+)?$", Tok)
                if Match and Tok not in ("PAW", "PAW_PBE", "PAW_GGA", "US"):
                    El = Match.group(1)
                    break
            if El and El not in Seen:
                Seen.add(El)
                Elements.append(El)
    return Elements


def ReorderPositionToPotcar(Position: pd.DataFrame, PotcarOrder: List[str]) -> pd.DataFrame:
    """Return Position with rows regrouped so species follow the POTCAR element order.

    VASP maps POTCAR blocks to POSCAR species groups positionally, so the POSCAR
    species order must equal the POTCAR order. xyz frames store bare element symbols
    (Zr, O, C); POTCAR TITEL lines may carry suffixes (Zr_sv, O_pv) which
    ParsePotcarElements already strips to bare symbols, so the comparison is on bare
    symbols. Within each species the original relative order is preserved (stable sort;
    keeps appended-O ordering intact). Raises if the frame has an element that has no
    matching POTCAR block (it could not be assigned a pseudopotential).
    """
    FrameElements = list(dict.fromkeys(Position["Element"]))
    Unknown = [El for El in FrameElements if El not in PotcarOrder]
    if Unknown:
        raise ValueError(
            f"Final frame contains element(s) {Unknown} with no matching POTCAR block "
            f"(POTCAR order {PotcarOrder}). Cannot order the POSCAR to the POTCAR."
        )
    Rank = {El: Index for Index, El in enumerate(PotcarOrder)}
    Ordered = Position.copy()
    Ordered["_Rank"] = Ordered["Element"].map(Rank)
    Ordered = (
        Ordered.sort_values("_Rank", kind="stable")
        .drop(columns="_Rank")
        .reset_index(drop=True)
    )
    return Ordered


def ValidatePotcarOrder(SpeciesOrder: List[str], PotcarPath: Path) -> None:
    """Validate the (already POTCAR-ordered) POSCAR species order against the POTCAR.

    The real hard requirement is POSCAR species order == POTCAR species order — VASP
    maps POTCAR blocks to POSCAR groups positionally, so a mismatch corrupts the run.
    Called AFTER ReorderPositionToPotcar, so this confirms consistency and flags a
    POTCAR that lists species absent from the frame. 'O last' is a preprocessing
    convention, NOT a runtime requirement (the O2 add/remove + velocity-insertion paths
    are order-agnostic), so an O-first ordering is allowed with only an informational
    note. If the POTCAR carries no parseable TITEL lines (e.g. the placeholder example
    POTCAR), consistency cannot be checked and a warning is printed.
    """
    PotcarOrder = ParsePotcarElements(PotcarPath)
    if not PotcarOrder:
        print(
            f"WARNING: no TITEL lines found in {PotcarPath}; cannot order/verify species "
            f"against POTCAR. Ensure POTCAR matches the reconstructed POSCAR order "
            f"{SpeciesOrder}."
        )
        return

    # After reordering, the present species must appear in POTCAR order (VASP correctness).
    PresentInPotcarOrder = [El for El in PotcarOrder if El in SpeciesOrder]
    if PresentInPotcarOrder != SpeciesOrder:
        raise ValueError(
            f"Reconstructed POSCAR species order {SpeciesOrder} is not consistent with "
            f"POTCAR order {PotcarOrder}."
        )

    Absent = [El for El in PotcarOrder if El not in SpeciesOrder]
    if Absent:
        print(
            f"WARNING: POTCAR lists {Absent} but the final frame contains none of these "
            f"element(s). VASP will fail unless POTCAR is trimmed to match the POSCAR "
            f"species {SpeciesOrder}."
        )

    if PotcarOrder[-1] != "O":
        print(
            f"NOTE: POTCAR order {PotcarOrder} does not end in 'O' (SGUSCHI's usual "
            "convention is oxygen last). Proceeding — the POSCAR is written to match "
            "the POTCAR, which is what VASP requires."
        )


def ReadThmexpMax(JobInPath: Path) -> Optional[int]:
    """Return thmexp_max from job.in, or None if absent/unparseable (csh default is 5)."""
    try:
        Params = vio.ReadKeyValueFile(Path(JobInPath))
    except (FileNotFoundError, ValueError):
        return None
    Raw = Params.get("thmexp_max", "")
    try:
        return int(float(Raw))
    except (TypeError, ValueError):
        return None


def CheckStopCondition(JobInPath: Path, N: int) -> None:
    """Abort if placeholder folders 1..N would trip volsearch_cont's stop condition.

    volsearch_cont stops when a folder named ``thmexp_max`` exists (default 5 if the
    key is missing). Creating placeholders 1..N with N >= thmexp_max would stop the
    resumed run after a single segment.
    """
    ThmexpMax = ReadThmexpMax(JobInPath)
    if ThmexpMax is None:
        raise ValueError(
            f"{JobInPath} has no 'thmexp_max' key. volsearch_cont would default to 5 "
            "and stop immediately once placeholder folder 5 exists. Set thmexp_max "
            "well above the resume step count before resuming."
        )
    if N >= ThmexpMax:
        raise ValueError(
            f"Resume step N={N} >= thmexp_max={ThmexpMax}: the run would stop after "
            "one segment (folder thmexp_max already exists as a placeholder). Raise "
            "thmexp_max in job.in before resuming."
        )


# ---------------------------------------------------------------------------
# Input templating (mirrors WorkSpaceSetup; PreparePOSCAR intentionally omitted)
# ---------------------------------------------------------------------------

def _MakeFolderTag(FolderName: str) -> str:
    """Convert '873_2' -> '873_s_2' for the scheduler job name."""
    Parts = FolderName.split("_", 1)
    if len(Parts) == 2:
        return "{}_s_{}".format(Parts[0], Parts[1])
    return "{}_s".format(FolderName)


def _UpdateJobName(JobContent: str, FolderTag: str) -> str:
    """Replace the scheduler job-name line (SLURM or PBS) with FolderTag."""
    SlurmPattern = re.compile(r"^(#SBATCH\s+--job-name=).*$", re.MULTILINE)
    if SlurmPattern.search(JobContent):
        return SlurmPattern.sub(r"\1'{}'".format(FolderTag), JobContent)
    PbsPattern = re.compile(r"^(#PBS\s+-N\s*).*$", re.MULTILINE)
    if PbsPattern.search(JobContent):
        return PbsPattern.sub(r"\g<1>{}".format(FolderTag), JobContent)
    return "#SBATCH --job-name='{}'\n{}".format(FolderTag, JobContent)


def _CopyFile(Src: Path, Dst: Path) -> None:
    """Copy Src -> Dst, skipping when they are the same file (in-place resume: the
    root inputs like OxParams/CovalentRadii already live at the target root)."""
    Src, Dst = Path(Src), Path(Dst)
    Dst.parent.mkdir(parents=True, exist_ok=True)
    if Dst.exists() and Src.resolve() == Dst.resolve():
        return
    shutil.copy2(Src, Dst)


def _SetIncarTemperature(IncarPath: Path, Temp: str) -> None:
    """Set TEBEG/TEEND to Temp, preserving inline comments (mirrors WorkSpaceSetup)."""
    with IncarPath.open("r", encoding="utf-8") as File:
        Lines = File.readlines()
    NewLines = []
    for Line in Lines:
        if re.search(r"\bTEBEG\b", Line, re.IGNORECASE):
            Comment = "#" + Line.split("#", 1)[1].strip() if "#" in Line else ""
            NewLines.append("TEBEG = {} {}\n".format(Temp, Comment))
        elif re.search(r"\bTEEND\b", Line, re.IGNORECASE):
            Comment = "#" + Line.split("#", 1)[1].strip() if "#" in Line else ""
            NewLines.append("TEEND = {} {}\n".format(Temp, Comment))
        else:
            NewLines.append(Line)
    with IncarPath.open("w", encoding="utf-8") as File:
        File.writelines(NewLines)


def _SetJobInTemp(JobInPath: Path, Temp: str) -> None:
    """Set 'temp = ...' in job.in to Temp (mirrors WorkSpaceSetup)."""
    with JobInPath.open("r", encoding="utf-8") as File:
        Text = File.read()
    Pattern = re.compile(r"^(\s*#?\s*)(temp)(\s*=\s*)([^#\n]*)(.*)$", re.IGNORECASE | re.MULTILINE)
    New = Pattern.sub(lambda M: "{}{}{}{}{}".format(M.group(1), M.group(2), M.group(3), str(Temp), M.group(5)), Text)
    with JobInPath.open("w", encoding="utf-8") as File:
        File.write(New)


def _SetJobInNavg(JobInPath: Path, Value: str = "10000000") -> None:
    """Set navg so volsearch_cont runs indefinitely (mirrors WorkSpaceSetup)."""
    with JobInPath.open("r", encoding="utf-8") as File:
        Text = File.read()
    Pattern = re.compile(r"^(\s*#?\s*)(navg)(\s*=\s*)([^#\n]*)(.*)$", re.IGNORECASE | re.MULTILINE)
    New = Pattern.sub(lambda M: M.group(1) + M.group(2) + M.group(3) + Value + M.group(5), Text)
    with JobInPath.open("w", encoding="utf-8") as File:
        File.write(New)


# ---------------------------------------------------------------------------
# Workspace reconstruction
# ---------------------------------------------------------------------------

def ParseTrajName(TrajName: str) -> Tuple[str, str]:
    """Split '{Temp}_{Sim}' into (Temp, Sim). Raises if it has no '_' separator."""
    if "_" not in TrajName:
        raise ValueError(
            f"Trajectory name {TrajName!r} is not of the form '{{Temp}}_{{Sim}}'."
        )
    Temp, Sim = TrajName.rsplit("_", 1)
    return Temp, Sim


def DiscoverTrajectories(XyzDir: Path) -> List[Tuple[str, Path, Path]]:
    """Return [(traj_name, xyz_path, rate_path), ...] for every {traj}.xyz that has a
    matching RateAnalysis_{traj}.csv in XyzDir."""
    XyzDir = Path(XyzDir)
    Result: List[Tuple[str, Path, Path]] = []
    for XyzPath in sorted(XyzDir.glob("*.xyz")):
        TrajName = XyzPath.stem
        RatePath = XyzDir / "RateAnalysis_{}.csv".format(TrajName)
        if RatePath.exists():
            Result.append((TrajName, XyzPath, RatePath))
        else:
            print(f"SKIP {XyzPath.name}: no matching {RatePath.name}")
    return Result


def BuildWorkspaceTree(TargetRoot: Path, TrajName: str, InputsDir: Path,
                       Force: bool = False) -> Path:
    """Create {TrajName}/Dir_VolSearch (+ Dir_OptUnitCell) with templated inputs.

    Mirrors WorkSpaceSetup layout but does NOT run OxidationPreprocessing (no vacuum /
    O2 re-placement). Returns the Dir_VolSearch path.
    """
    Temp, _Sim = ParseTrajName(TrajName)
    SimDir = TargetRoot / TrajName
    VolSearchDir = SimDir / "Dir_VolSearch"
    OptUnitDir = SimDir / "Dir_OptUnitCell"

    if VolSearchDir.exists() and any(VolSearchDir.iterdir()) and not Force:
        raise FileExistsError(
            f"{VolSearchDir} already exists and is non-empty. Refusing to clobber a "
            "live run — pass --force to override."
        )

    for FN in SIM_INPUT_FILES:
        Src = InputsDir / FN
        if not Src.exists():
            raise FileNotFoundError(f"Required input file missing: {Src}")
        _CopyFile(Src, SimDir / FN)

    # Root-level inputs read by OxidationStep (OxParams, CovalentRadii).
    for FN in ROOT_INPUT_FILES:
        Src = InputsDir / FN
        if not Src.exists():
            raise FileNotFoundError(f"Required root input file missing: {Src}")
        _CopyFile(Src, TargetRoot / FN)

    _WarnIfNotCovered(TargetRoot / "OxParams", Temp, _Sim, TrajName)

    _SetIncarTemperature(SimDir / "INCAR", Temp)
    _SetJobInTemp(SimDir / "job.in", Temp)

    FolderTag = _MakeFolderTag(TrajName)
    JobsubText = (SimDir / "jobsub").read_text(encoding="utf-8")
    (SimDir / "jobsub").write_text(_UpdateJobName(JobsubText, FolderTag), encoding="utf-8")

    VolSearchDir.mkdir(parents=True, exist_ok=True)
    OptUnitDir.mkdir(parents=True, exist_ok=True)
    (OptUnitDir / "optunitcell_is_done").touch()

    # Copy templated inputs into Dir_VolSearch and set navg for indefinite running.
    for FN in SIM_INPUT_FILES:
        _CopyFile(SimDir / FN, VolSearchDir / FN)
    _SetJobInNavg(VolSearchDir / "job.in")

    return VolSearchDir


def _WarnIfNotCovered(OxParamsPath: Path, Temp: str, Sim: str, TrajName: str) -> None:
    """Warn if (Temp, Sim) is outside OxParams Temperatures/NSims (SGUSCHI won't scan it)."""
    import ast
    try:
        Params = vio.ReadKeyValueFile(Path(OxParamsPath))
        Temperatures = ast.literal_eval(Params["Temperatures"])
        NSims = int(Params["NSims"])
        InRange = (int(Temp) in [int(T) for T in Temperatures]) and (1 <= int(Sim) <= NSims)
    except Exception:
        return  # non-standard OxParams; leave validation to SGUSCHI
    if not InRange:
        print(
            f"WARNING: {TrajName} is outside OxParams Temperatures={Temperatures} / "
            f"NSims={NSims}. SGUSCHI will not scan or launch it until OxParams covers it."
        )


def SeedPlaceholderFolders(VolSearchDir: Path, N: int) -> None:
    """Create empty numeric placeholder folders 1..N (satisfies volsearch_cont's counter)."""
    for Step in range(1, N + 1):
        (VolSearchDir / str(Step)).mkdir(exist_ok=True)


def PlaceCarriedState(VolSearchDir: Path, TargetRoot: Path, TrajName: str,
                      LastFrameData: LastFrame, RatePath: Path, XyzPath: Path) -> None:
    """Write POSCAR + carry the full RateAnalysis and xyz forward (order matters)."""
    # 1. Reconstructed POSCAR (no velocities; VASP re-thermalizes).
    vio.WritePoscar(str(VolSearchDir), LastFrameData.PositionFrac, LastFrameData.CellDim,
                    Velocities=None)

    # 2. Full RateAnalysis (N+1 rows) — what ReadRateAnalysis loads (=> LatestFolder=N+1).
    _CopyFileSafe(RatePath, VolSearchDir / "RateAnalysis.csv")

    # 3. Carry the existing xyz forward BEFORE any launch, so WriteXYZ appends (never
    #    truncates), and keep the RateAnalysis mirror consistent for SimulationSummary.
    XyzOutDir = TargetRoot / "xyz_files"
    XyzOutDir.mkdir(parents=True, exist_ok=True)
    _CopyFileSafe(XyzPath, XyzOutDir / "{}.xyz".format(TrajName))
    _CopyFileSafe(RatePath, XyzOutDir / "RateAnalysis_{}.csv".format(TrajName))


def _CopyFileSafe(Src: Path, Dst: Path) -> None:
    """Alias for _CopyFile (same-file-safe); named for clarity at carry-state sites."""
    _CopyFile(Src, Dst)


def EnsureCleanSubmissionState(VolSearchDir: Path, NextStep: int) -> None:
    """Clear stale run-state and the root OUTCAR so recovery submits step NextStep.

    Also records ``poscar_built_for_step = NextStep`` (= N+1). volsearch_cont's startup
    recovery reads this to decide whether it must rebuild POSCAR from CONTCAR; since the
    reconstructed POSCAR IS the intended geometry for the first continuation step, this
    makes recovery skip that branch — suppressing the misleading "no CONTCAR ... deleted
    WAVECAR" warning (and the no-op WAVECAR deletion) — while still submitting our POSCAR
    as step NextStep. It does not affect the double-submission guard (`.vasp_submitted_step`).
    """
    for FN in STALE_RUN_STATE:
        (VolSearchDir / FN).unlink(missing_ok=True)
    (VolSearchDir / "OUTCAR").unlink(missing_ok=True)
    Pulay = VolSearchDir / "Dir_Pressure_Pulay"
    if Pulay.is_dir():
        shutil.rmtree(Pulay)
    # Written AFTER the STALE_RUN_STATE sweep (which removes any old value).
    (VolSearchDir / "poscar_built_for_step").write_text(str(NextStep), encoding="utf-8")


def WriteResumeSeamMarker(VolSearchDir: Path, N: int) -> None:
    """Record the seam step N so repair/rollback tools refuse to cross it."""
    Timestamp = datetime.now().isoformat(timespec="seconds")
    Content = "{}\n# resumed from trajectory at {}\n".format(N, Timestamp)
    (VolSearchDir / RESUME_SEAM_MARKER).write_text(Content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-trajectory orchestration
# ---------------------------------------------------------------------------

def ResumeTrajectory(TrajName: str, XyzPath: Path, RatePath: Path, TargetRoot: Path,
                     InputsDir: Path, Force: bool = False, DryRun: bool = False) -> Dict[str, object]:
    """Rebuild one trajectory's workspace. Returns a summary dict for logging."""
    Temp, Sim = ParseTrajName(TrajName)

    # --- Validation gates (cheap; run even in dry-run) ---
    N = DetermineN(RatePath, XyzPath, Force=Force)
    LastFrameData = ParseLastXYZFrame(XyzPath)

    # Order the reconstructed POSCAR to the POTCAR element order (bare symbols, so
    # Zr_sv/O_pv-style suffixes are handled by ParsePotcarElements). Skips reordering
    # only when the POTCAR carries no TITEL lines (placeholder POTCAR).
    PotcarOrder = ParsePotcarElements(InputsDir / "POTCAR")
    OriginalElements = list(LastFrameData.PositionFrac["Element"])
    if PotcarOrder:
        LastFrameData.PositionFrac = ReorderPositionToPotcar(
            LastFrameData.PositionFrac, PotcarOrder
        )
        if list(LastFrameData.PositionFrac["Element"]) != OriginalElements:
            print(
                f"  [{TrajName}] NOTE: atom order changed to match POTCAR "
                f"{PotcarOrder}. The continuing xyz will use this order — a one-time "
                "permutation at the restart seam. This is safe for SGUSCHI's per-frame "
                "analysis (gas detection, void tracking) but would affect any across-"
                "seam per-atom tracking (e.g. MSD)."
            )
    SpeciesOrder = list(dict.fromkeys(LastFrameData.PositionFrac["Element"]))
    ValidatePotcarOrder(SpeciesOrder, InputsDir / "POTCAR")
    CheckStopCondition(InputsDir / "job.in", N)

    Summary: Dict[str, object] = {
        "trajectory": TrajName,
        "temp": Temp,
        "sim": Sim,
        "N": N,
        "atoms": LastFrameData.NumAtoms,
        "species_order": SpeciesOrder,
        "last_step": LastFrameData.Step,
        "last_time_fs": LastFrameData.TimeFs,
    }

    if DryRun:
        Summary["action"] = "dry-run (no files written)"
        return Summary

    # --- Reconstruction (order matters) ---
    VolSearchDir = BuildWorkspaceTree(TargetRoot, TrajName, InputsDir, Force=Force)
    SeedPlaceholderFolders(VolSearchDir, N)
    PlaceCarriedState(VolSearchDir, TargetRoot, TrajName, LastFrameData, RatePath, XyzPath)
    EnsureCleanSubmissionState(VolSearchDir, NextStep=N + 1)
    WriteResumeSeamMarker(VolSearchDir, N)

    # Post-build invariant: len(RateAnalysis) == placeholder folders + 1 == N + 1.
    Placeholders = len(NumericStepFolders(VolSearchDir))
    Rows = len(pd.read_csv(VolSearchDir / "RateAnalysis.csv"))
    if not (Placeholders == N and Rows == N + 1):
        raise RuntimeError(
            f"Post-build invariant failed for {TrajName}: placeholders={Placeholders} "
            f"(expected {N}), RateAnalysis rows={Rows} (expected {N + 1})."
        )

    Summary["action"] = "prepared"
    Summary["volsearch_dir"] = str(VolSearchDir)
    return Summary


def WriteStarterSummary(TargetRoot: Path) -> None:
    """Build and write a one-shot 'starter' SimulationSummary for the target root.

    Reuses SimulationSummary so the resumed workspace immediately carries the same
    ``simulation_summary.txt``/``.tsv`` artefact the orchestration daemon would write,
    pre-populated from the carried-forward RateAnalysis (cumulative time, O2 counts).
    Advisory: failures never abort the resume.
    """
    try:
        from utils import SimulationSummary as Summary
        OxParamsPath = TargetRoot / "OxParams"
        Rows = Summary.BuildSummary(TargetRoot, OxParamsPath if OxParamsPath.exists() else None)
        print("\nStarter simulation summary (pre-submission):")
        print(Summary.FormatTable(Rows), end="")
        Summary.WriteOutputs(TargetRoot, Rows)
        print(f"Wrote {TargetRoot / Summary.SUMMARY_TXT}")
    except Exception as Exc:
        print(f"WARNING: could not build starter summary: {Exc!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def InferLayout(Cwd: Path) -> Optional[Tuple[Path, Path, Path]]:
    """Infer (TargetRoot, InputsDir, XyzDir) from the current directory.

    Two supported layouts (so you can just ``cd`` and run):
      * Inside an ``xyz_files/`` folder (cwd holds *.xyz)     -> base = cwd.parent
      * In the workspace root that contains ``xyz_files/``    -> base = cwd
    In both cases the base workspace supplies the inputs and receives the rebuilt
    {Temp}_{Sim} folders. Returns None if neither layout matches.
    """
    Cwd = Path(Cwd).resolve()
    if Cwd.name == "xyz_files" or any(Cwd.glob("*.xyz")):
        Base = Cwd.parent
        return Base, Base, Cwd
    if (Cwd / "xyz_files").is_dir():
        return Cwd, Cwd, Cwd / "xyz_files"
    return None


def ResolvePaths(Args: argparse.Namespace) -> Optional[Tuple[Path, Path, Path]]:
    """Resolve (TargetRoot, InputsDir, XyzDir) from CLI args, filling gaps by inference."""
    Inferred = InferLayout(Path.cwd())

    TargetRoot = Path(Args.target).resolve() if Args.target else (Inferred[0] if Inferred else None)
    InputsDir = Path(Args.inputs).resolve() if Args.inputs else (Inferred[1] if Inferred else None)
    if Args.xyz_dir:
        XyzDir = Path(Args.xyz_dir).resolve()
    elif Inferred:
        XyzDir = Inferred[2]
    elif TargetRoot:
        XyzDir = TargetRoot / "xyz_files"
    else:
        XyzDir = None

    if TargetRoot is None or InputsDir is None or XyzDir is None:
        return None
    return TargetRoot, InputsDir, XyzDir


def ParseArgs(Argv: Optional[List[str]] = None) -> argparse.Namespace:
    Parser = argparse.ArgumentParser(
        description="Rebuild a runnable workspace to continue simulations from xyz_files output."
    )
    Parser.add_argument("--target", default=None,
                        help="RootDir to build {Temp}_{Sim}/Dir_VolSearch under "
                             "(default: inferred from the current directory).")
    Parser.add_argument("--inputs", default=None,
                        help="Directory with POTCAR OxParams CovalentRadii INCAR KPOINTS "
                             "job.in jobsub (default: the inferred workspace root).")
    Parser.add_argument("--xyz-dir", default=None,
                        help="Source xyz_files/ directory (default: <target>/xyz_files, "
                             "or the current directory if it holds the .xyz files).")
    Parser.add_argument("--only", default=None,
                        help="Comma-separated trajectory names to restrict to (default: all).")
    Parser.add_argument("--force", action="store_true",
                        help="Override step-count mismatch and non-empty-dir guards.")
    Parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report; create nothing.")
    Parser.add_argument("--no-summary", action="store_true",
                        help="Skip writing the starter simulation_summary.txt/.tsv after prepare.")
    return Parser.parse_args(Argv)


def main(Argv: Optional[List[str]] = None) -> int:
    for Stream in (sys.stdout, sys.stderr):
        if hasattr(Stream, "reconfigure"):
            try:
                Stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    Args = ParseArgs(Argv)
    Resolved = ResolvePaths(Args)
    if Resolved is None:
        print(
            "ERROR: could not infer paths from the current directory.\n"
            "Either cd into the workspace root (which contains xyz_files/) or into the "
            "xyz_files/ folder itself, or pass --target and --inputs explicitly."
        )
        return 1
    TargetRoot, InputsDir, XyzDir = Resolved

    if not InputsDir.is_dir():
        print(f"ERROR: inputs directory not found: {InputsDir}")
        return 1
    if not XyzDir.is_dir():
        print(f"ERROR: xyz-dir not found: {XyzDir}")
        return 1

    Trajectories = DiscoverTrajectories(XyzDir)
    if Args.only:
        Wanted = {Name.strip() for Name in Args.only.split(",") if Name.strip()}
        Trajectories = [T for T in Trajectories if T[0] in Wanted]
        Missing = Wanted - {T[0] for T in Trajectories}
        for Name in sorted(Missing):
            print(f"WARNING: requested trajectory {Name!r} not found in {XyzDir}.")

    if not Trajectories:
        print("No trajectories to reconstruct.")
        return 1

    Mode = "DRY-RUN" if Args.dry_run else "PREPARE"
    print(f"ResumeFromTrajectory [{Mode}]")
    print(f"  target={TargetRoot}")
    print(f"  inputs={InputsDir}")
    print(f"  {len(Trajectories)} trajectory(ies) from {XyzDir}")

    Failures = 0
    for TrajName, XyzPath, RatePath in Trajectories:
        try:
            Summary = ResumeTrajectory(TrajName, XyzPath, RatePath, TargetRoot,
                                       InputsDir, Force=Args.force, DryRun=Args.dry_run)
            print(
                "  [{trajectory}] N={N} atoms={atoms} species={species_order} "
                "lastStep={last_step} lastTime(fs)={last_time_fs} -> {action}".format(**Summary)
            )
        except Exception as Exc:
            Failures += 1
            print(f"  [{TrajName}] ERROR: {Exc}")

    if Failures:
        print(f"{Failures} trajectory(ies) failed.")
        return 1
    if not Args.dry_run:
        if not Args.no_summary:
            WriteStarterSummary(TargetRoot)
        print("Done. Resume with the normal 'sbatch OxidationMaster' (sims are 'pending').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
