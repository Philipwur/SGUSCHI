# %%
"""
This Code Will Set Up the Work Space according to the OxParams file.
Note, if oxygens are in base structure, code might break. (have not tested)

====================================================
Workplace Requirements for Working Directory Script
====================================================

FILES REQUIRED IN WORKING DIRECTORY
-----------------------------------
The following files must exist before running this script:

1. OxParams        → Contains simulation parameters with keys:
                     - AtomicRadiusTol
                     - TargetPP
                     - PPSmoothing
                     - InitO2Count
                     - GasRatio
                     - Temperatures
                     - NSims
2. POSCAR          → Base structure file (Supercell Expanded)
3. KPOINTS         → VASP KPOINTS input
4. POTCAR          → VASP pseudopotentials (Same order as POSCAR + O Last)
5. INCAR           → VASP INCAR input
6. job.in          → SLUSCHI job file
7. jobsub          → SLURM job submission script
8. CovalentRadii   → File containing Covalent radii for elements for bonding
                     algorithm. Give in Angstroms.
"""

from pathlib import Path
import re
import shutil
import subprocess
import ast
import sys
from typing import List, Optional, Tuple

# --- Make imports location-independent ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio
import OxidationPreprocessing as opp


def MakeFolderTag(folder_name: str) -> str:
    """Convert '873_2' → '873_s_2' for job name."""
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}_s_{parts[1]}"
    return f"{folder_name}_s"


def UpdateJobName(job_content: str, folder_tag: str) -> str:
    """Replace the scheduler job name line with the folder tag (SLURM or PBS)."""
    slurm_pattern = re.compile(r"^(#SBATCH\s+--job-name=).*$", re.MULTILINE)
    if slurm_pattern.search(job_content):
        return slurm_pattern.sub(r"\1'{}'".format(folder_tag), job_content)
    pbs_pattern = re.compile(r"^(#PBS\s+-N\s*).*$", re.MULTILINE)
    if pbs_pattern.search(job_content):
        return pbs_pattern.sub(r"\g<1>{}".format(folder_tag), job_content)
    return "#SBATCH --job-name='{}'\n{}".format(folder_tag, job_content)


def EnsureFilesExist(workdir: Path, filenames: List[str]):
    """Check if required input files exist in workdir."""
    missing = [f for f in filenames if not (workdir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

def CopyFile(src: Path, dst: Path):
    """Copy a single file while ensuring destination directories exist."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def WriteTextFile(FilePath: Path, Text: str):
    FilePath.parent.mkdir(parents=True, exist_ok=True)
    with FilePath.open("w", encoding="utf-8") as F:
        F.write(Text)


def SetupWorkspace(
    WorkDir: Path,
    Params: dict,
    OnlySimIndices: Optional[List[Tuple[int, int]]] = None,
) -> List[str]:
    """Set up simulation folders from OxParams. Returns log lines.

    OnlySimIndices: optional list of (Temp, SimIdx) pairs to set up.
    If None, sets up all combinations from Params.
    Per-folder idempotent: folders where Dir_VolSearch already exists are skipped.
    """
    RequiredFiles = ["POSCAR", "KPOINTS", "POTCAR", "INCAR", "job.in", "jobsub"]
    EnsureFilesExist(WorkDir, RequiredFiles)

    Position, CellDim = vio.ReadPoscar(WorkDir)

    JobsubBasePath = WorkDir / "jobsub"
    with JobsubBasePath.open("r", encoding="utf-8") as F:
        JobsubBaseContent = F.read()

    LogLines = []
    LogLines.append("Working Directory Prepared Successfully")

    Temperatures = ast.literal_eval(Params["Temperatures"])
    NSims = int(Params["NSims"])
    GasRatio = float(Params["GasRatio"])
    InitO2 = int(Params["InitO2Count"])

    ForCopy = ["POTCAR", "job.in", "KPOINTS", "INCAR"]

    for Temp in Temperatures:
        for SimIdx in range(1, NSims + 1):
            if OnlySimIndices is not None and (Temp, SimIdx) not in OnlySimIndices:
                continue

            FolderName = "{}_{}".format(Temp, SimIdx)
            SimDir = WorkDir / FolderName
            VolSearchDir = SimDir / "Dir_VolSearch"

            if VolSearchDir.exists():
                LogLines.append("[{}] already exists — skipped".format(FolderName))
                continue

            SimDir.mkdir(parents=True, exist_ok=True)

            # Copy files into sim folder
            for FN in ForCopy:
                CopyFile(WorkDir / FN, SimDir / FN)

            # Update TEBEG / TEEND in INCAR to match folder temp
            IncarPath = SimDir / "INCAR"
            if IncarPath.exists():
                with IncarPath.open("r", encoding="utf-8") as F:
                    Lines = F.readlines()

                NewLines = []
                for Line in Lines:
                    if re.search(r"\bTEBEG\b", Line, re.IGNORECASE):
                        Comment = ""
                        if "#" in Line:
                            Comment = "#" + Line.split("#", 1)[1].strip()
                        NewLines.append("TEBEG = {} {}\n".format(Temp, Comment))
                    elif re.search(r"\bTEEND\b", Line, re.IGNORECASE):
                        Comment = ""
                        if "#" in Line:
                            Comment = "#" + Line.split("#", 1)[1].strip()
                        NewLines.append("TEEND = {} {}\n".format(Temp, Comment))
                    else:
                        NewLines.append(Line)

                with IncarPath.open("w", encoding="utf-8") as F:
                    F.writelines(NewLines)

            # Update 'temp = ...' in job.in to match folder temp
            JobInPath = SimDir / "job.in"
            if JobInPath.exists():
                with JobInPath.open("r", encoding="utf-8") as F:
                    JobInText = F.read()

                TempPattern = re.compile(
                    r"^(\s*#?\s*)(temp)(\s*=\s*)([^#\n]*)(.*)$",
                    re.IGNORECASE | re.MULTILINE
                )

                def _ReplTemp(M):
                    return "{}{}{}{}{}".format(
                        M.group(1), M.group(2), M.group(3), str(Temp), M.group(5)
                    )

                JobInNew = TempPattern.sub(_ReplTemp, JobInText)

                with JobInPath.open("w", encoding="utf-8") as F:
                    F.write(JobInNew)

            # Prepare and write jobsub with updated job name
            FolderTag = MakeFolderTag(FolderName)
            JobsubContent = UpdateJobName(JobsubBaseContent, FolderTag)
            WriteTextFile(SimDir / "jobsub", JobsubContent)

            # Create subfolders
            OptUnitDir = SimDir / "Dir_OptUnitCell"
            VolSearchDir.mkdir(parents=True, exist_ok=True)
            OptUnitDir.mkdir(parents=True, exist_ok=True)

            # Create optunitcell_is_done file
            (OptUnitDir / "optunitcell_is_done").touch()

            # Prepare new POSCAR (unique per folder)
            NewPosition, NewCellDim = opp.PreparePOSCAR(
                Position, CellDim, GasRatio=GasRatio, InitO2=InitO2
            )
            vio.WritePoscar(str(SimDir), NewPosition, NewCellDim)
            print("Prepared POSCAR for {}".format(FolderName))

            # Copy files into Dir_VolSearch
            CopyFile(SimDir / "POSCAR", VolSearchDir / "POSCAR")
            CopyFile(SimDir / "jobsub", VolSearchDir / "jobsub")
            for FN in ["KPOINTS", "POTCAR", "job.in", "INCAR"]:
                CopyFile(SimDir / FN, VolSearchDir / FN)

            # Set navg = 10000000 in Dir_VolSearch/job.in so volsearch_cont runs indefinitely
            VolSearchJobIn = VolSearchDir / "job.in"
            if VolSearchJobIn.exists():
                with VolSearchJobIn.open("r", encoding="utf-8") as F:
                    VsJobInText = F.read()

                NavgPat = re.compile(
                    r"^(\s*#?\s*)(navg)(\s*=\s*)([^#\n]*)(.*)$",
                    re.IGNORECASE | re.MULTILINE
                )
                VsJobInNew = NavgPat.sub(
                    lambda m: m.group(1) + m.group(2) + m.group(3) + "10000000" + m.group(5),
                    VsJobInText
                )
                with VolSearchJobIn.open("w", encoding="utf-8") as F:
                    F.write(VsJobInNew)

            # Submit initial VASP job using vaspcmd from Dir_VolSearch/job.in
            try:
                JobInParams = vio.ReadKeyValueFile(VolSearchJobIn)
                VaspCmd = JobInParams.get("vaspcmd", "").strip().split()[0] if JobInParams.get("vaspcmd", "").strip() else ""
            except Exception:
                VaspCmd = ""

            if VaspCmd:
                try:
                    Result = subprocess.run(
                        [VaspCmd, "jobsub"],
                        cwd=str(VolSearchDir),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    Status = "exit {}".format(Result.returncode)
                    if Result.returncode != 0:
                        Status = "ERROR ({})".format(Status)
                    LogLines.append("[{}] {} jobsub → {}".format(FolderName, VaspCmd, Status))
                    if Result.stdout:
                        LogLines.append("[{}] stdout: {}".format(FolderName, Result.stdout.strip()))
                    if Result.stderr:
                        LogLines.append("[{}] stderr: {}".format(FolderName, Result.stderr.strip()))
                except FileNotFoundError:
                    LogLines.append("[{}] '{}' not on PATH — skipped initial VASP submission".format(FolderName, VaspCmd))
                except Exception as E:
                    LogLines.append("[{}] initial VASP submission failed: {!r}".format(FolderName, E))
            else:
                LogLines.append("[{}] vaspcmd not set in job.in — skipped initial VASP submission".format(FolderName))

            LogLines.append("[{}] set up successfully".format(FolderName))

    # Create xyz_files folder
    (WorkDir / "xyz_files").mkdir(parents=True, exist_ok=True)

    return LogLines


def PrepareWorkingDirectory():

    WorkDir = Path.cwd()

    RequiredKeys = [
        'Temperatures',
        'NSims',
        'GasRatio',
        'InitO2Count'
    ]

    OxParamsPath = WorkDir / "OxParams"
    Params = vio.ReadKeyValueFile(OxParamsPath, RequiredKeys=RequiredKeys)

    LogLines = SetupWorkspace(WorkDir, Params)

    WriteTextFile(WorkDir / "log.out", "\n".join(LogLines) + "\n")


if __name__ == "__main__":
    PrepareWorkingDirectory()
