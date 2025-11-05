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
9. jobsub_master   → Master process slurm job submission script
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import List  

# --- Make imports location-independent ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio
import OxidationPreprocessing as opp
# from workflow import OxidationAnalysis as an


def MakeFolderTag(folder_name: str) -> str:
    """Convert '873_2' → '873_s_2' for job name."""
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}_s_{parts[1]}"
    return f"{folder_name}_s"


def UpdateJobName(job_content: str, folder_tag: str) -> str:
    """Replace the #SBATCH job name line with the folder tag."""
    pattern = re.compile(r"^(#SBATCH\s+--job-name=).*$", re.MULTILINE)
    replacement = r"\1'{}'".format(folder_tag)
    if pattern.search(job_content):
        return pattern.sub(replacement, job_content)
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
        

def PrepareWorkingDirectory():
    
    WorkDir = Path.cwd()

    #K required in OxParams
    RequiredKeys = [
        'Temperatures',
        'NSims',
        'GasRatio',
        'InitO2Count'
    ]
    
    # 1) Read and validate OxParams and input files
    OxParamsPath = WorkDir / "OxParams"
    Params = vio.ReadKeyValueFile(OxParamsPath, RequiredKeys = RequiredKeys)

    RequiredFiles = ["POSCAR", "KPOINTS", "POTCAR", "INCAR", "job.in", "jobsub"]
    EnsureFilesExist(WorkDir, RequiredFiles)

    # Load the baseline POSCAR once (Position, CellDim will be mutated per sim)
    Position, CellDim = vio.ReadPOSCAR(WorkDir)

    # Read the base jobsub content once
    JobsubBasePath = WorkDir / "jobsub"
    with JobsubBasePath.open("r", encoding="utf-8") as F:
        JobsubBaseContent = F.read()

    LogLines = []
    LogLines.append("Working Directory Prepared Sucessfully")

    Temperatures = Params[RequiredKeys[0]]
    NSims = Params[RequiredKeys[1]]
    GasRatio = Params[RequiredKeys[2]]
    InitO2 = Params[RequiredKeys[3]]

    ForCopy = ["POTCAR", "job.in", "KPOINTS", "INCAR"]

    for Temp in Temperatures:
        for SimIdx in range(1, NSims + 1):
            FolderName = "{}_{}".format(Temp, SimIdx)
            SimDir = WorkDir / FolderName
            SimDir.mkdir(parents=True, exist_ok=True)

            # 2) Copy files into sim folder
            for FN in ForCopy:
                CopyFile(WorkDir / FN, SimDir / FN)

            # --- Update TEBEG / TEEND in INCAR to match folder temp ---
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

            # --- Update 'temp = ...' in job.in to match folder temp ---
            JobInPath = SimDir / "job.in"
            if JobInPath.exists():
                with JobInPath.open("r", encoding="utf-8") as F:
                    JobInText = F.read()

                Pattern = re.compile(
                    r"^(\s*#?\s*)(temp)(\s*=\s*)([^#\n]*)(.*)$",
                    re.IGNORECASE | re.MULTILINE
                )

                def _Repl(M):
                    Prefix = M.group(1)
                    Key = M.group(2)  # preserve original key case
                    Eq = M.group(3)
                    Trailing = M.group(5)
                    return "{}{}{}{}{}".format(Prefix, Key, Eq, str(Temp), Trailing)

                JobInNew = Pattern.sub(_Repl, JobInText)

                with JobInPath.open("w", encoding="utf-8") as F:
                    F.write(JobInNew)

            # Prepare and write jobsub with updated job name
            FolderTag = MakeFolderTag(FolderName)
            JobsubContent = UpdateJobName(JobsubBaseContent, FolderTag)
            WriteTextFile(SimDir / "jobsub", JobsubContent)

            # 3) Create subfolders
            VolSearchDir = SimDir / "Dir_VolSearch"
            OptUnitDir = SimDir / "Dir_OptUnitCell"
            VolSearchDir.mkdir(parents=True, exist_ok=True)
            OptUnitDir.mkdir(parents=True, exist_ok=True)

            # 4) Create optunitcell_is_done file
            (OptUnitDir / "optunitcell_is_done").touch()

            # 5–7) Prepare new POSCAR (unique per folder)
            NewPosition, NewCellDim = opp.PreparePOSCAR(
                Position, CellDim, GasRatio=GasRatio, InitO2=InitO2
            )
            vio.WritePOSCAR(str(SimDir), NewPosition, NewCellDim)
            print("Prepared POSCAR for {}".format(FolderName))

            # 8) Copy files into Dir_VolSearch
            CopyFile(SimDir / "POSCAR", VolSearchDir / "POSCAR")
            CopyFile(SimDir / "jobsub", VolSearchDir / "jobsub")
            for FN in ["KPOINTS", "POTCAR", "job.in", "INCAR"]:
                CopyFile(SimDir / FN, VolSearchDir / FN)

            '''
            # 9) Run sbatch jobsub
            try:
                Proc = subprocess.run(
                    ["sbatch", "jobsub"],
                    cwd=str(VolSearchDir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                Status = "OK (exit {})".format(Proc.returncode)
                if Proc.returncode != 0:
                    Status = "ERROR (exit {})".format(Proc.returncode)
                LogLines.append("[{}] sbatch jobsub -> {}".format(FolderName, Status))
                if Proc.stdout:
                    LogLines.append("[{}] stdout: {}".format(FolderName, Proc.stdout.strip()))
                if Proc.stderr:
                    LogLines.append("[{}] stderr: {}".format(FolderName, Proc.stderr.strip()))
            except FileNotFoundError:
                LogLines.append("[{}] sbatch not found on PATH; skipped submission.".format(FolderName))
            except Exception as E:
                LogLines.append("[{}] sbatch execution failed: {!r}".format(FolderName, E))
            '''
            
    # 10) Create xyz_files folder
    (WorkDir / "xyz_files").mkdir(parents=True, exist_ok=True)

    # 11) Write log.out
    WriteTextFile(WorkDir / "log.out", "\n".join(LogLines) + "\n")


if __name__ == "__main__":
    PrepareWorkingDirectory()
