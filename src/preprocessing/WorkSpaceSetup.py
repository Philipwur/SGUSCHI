# This Code Will Set Up the Work Space according to the OxParams file.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
#import os
import re
import shutil
import subprocess
import sys

sys.path.append('..')

from workflow import VaspIO as vio
from workflow import OxidationAnalysis as an

import OxidationPrepocessing as opp


def ReadOxParams(FilePath: Path) -> dict:
    """
    Parse OxParams key=value lines. Accepts comma- or space-separated temperature lists.
    Required keys for this workflow: Temperatures, NSims, GasRatio, InitO2.
    """
    if not FilePath.exists():
        raise FileNotFoundError(f"OxParams not found at: {FilePath}")

    Params = {}
    with FilePath.open("r", encoding="utf-8") as F:
        for Line in F:
            Line = Line.strip()
            if not Line or Line.startswith("#") or "=" not in Line:
                continue
            Key, Val = [x.strip() for x in Line.split("=", 1)]
            Params[Key] = Val

    # Normalize & type-cast the inputs we need
    if "Temperatures" not in Params or not Params["Temperatures"]:
        raise ValueError("Temperatures must be provided in OxParams (comma- or space-separated).")
    TempsRaw = Params["Temperatures"]
    if "," in TempsRaw:
        Temperatures = [t.strip() for t in TempsRaw.split(",") if t.strip()]
    else:
        Temperatures = [t for t in TempsRaw.split() if t]

    if "NSims" not in Params or not Params["NSims"]:
        raise ValueError("NSims must be provided in OxParams.")
    try:
        NSims = int(Params["NSims"])
    except ValueError as E:
        raise ValueError("NSims must be an integer.") from E

    if "GasRatio" not in Params or not Params["GasRatio"]:
        raise ValueError("GasRatio must be provided in OxParams.")
    try:
        GasRatio = float(Params["GasRatio"])
    except ValueError as E:
        raise ValueError("GasRatio must be a float.") from E

    if "InitO2" not in Params or not Params["InitO2"]:
        raise ValueError("InitO2 must be provided in OxParams.")
    try:
        InitO2 = int(Params["InitO2"])
    except ValueError as E:
        raise ValueError("InitO2 must be an integer.") from E

    ParamsOut = {
        "Temperatures": Temperatures,
        "NSims": NSims,
        "GasRatio": GasRatio,
        "InitO2": InitO2,
    }
    return ParamsOut


def MakeFolderTag(FolderName: str) -> str:
    """
    Convert '{Temperature}_{NSim}' -> '{Temperature}_s_{NSim}' for job name.
    """
    Parts = FolderName.split("_", 1)
    if len(Parts) == 2:
        return f"{Parts[0]}_s_{Parts[1]}"
    # Fallback if unexpected format
    return f"{FolderName}_s"


def UpdateJobNameInContent(JobsubContent: str, FolderTag: str) -> str:
    """
    Replace a '#SBATCH --job-name=...' line with one matching the folder tag.
    Keeps single-quote style as in your example.
    """
    Pattern = re.compile(r"^(#SBATCH\s+--job-name=).*$", flags=re.MULTILINE)
    Replacement = rf"\1'{FolderTag}'"
    if Pattern.search(JobsubContent):
        return Pattern.sub(Replacement, JobsubContent)
    # If not found, append a job-name line near the top to be safe.
    Lines = JobsubContent.splitlines()
    InsertAt = 0
    Lines.insert(InsertAt, f"#SBATCH --job-name='{FolderTag}'")
    return "\n".join(Lines) + ("\n" if not JobsubContent.endswith("\n") else "")


def ReadPOSCARWithVaspIO(PoscarPath: Path):
    """
    Load POSCAR using a VaspIO function. Tries common function names.
    Expected return: (Position, CellDim)
    """
    if hasattr(vio, "ReadPOSCAR"):
        return vio.ReadPOSCAR(str(PoscarPath))
    if hasattr(vio, "LoadPOSCAR"):
        return vio.LoadPOSCAR(str(PoscarPath))
    if hasattr(vio, "GetPOSCAR"):
        return vio.GetPOSCAR(str(PoscarPath))
    raise AttributeError(
        "Could not find a POSCAR read function in VaspIO. Tried ReadPOSCAR, LoadPOSCAR, GetPOSCAR."
    )


def EnsureFilesExist(WorkDir: Path, RequiredFiles: list):
    Missing = [f for f in RequiredFiles if not (WorkDir / f).exists()]
    if Missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(Missing)}")


def WriteTextFile(FilePath: Path, Text: str):
    FilePath.parent.mkdir(parents=True, exist_ok=True)
    with FilePath.open("w", encoding="utf-8") as F:
        F.write(Text)


def CopyFile(Src: Path, Dst: Path):
    Dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(Src), str(Dst))


def PrepareWorkingDirectory():
    WorkDir = Path.cwd()

    # 1) Read and validate OxParams and input files
    OxParamsPath = WorkDir / "OxParams"
    Params = ReadOxParams(OxParamsPath)

    RequiredFiles = ["POSCAR", "KPOINTS", "POTCAR", "INCAR", "job.in", "jobsub"]
    EnsureFilesExist(WorkDir, RequiredFiles)

    # Load the baseline POSCAR once (Position, CellDim will be mutated per sim)
    BasePoscarPath = WorkDir / "POSCAR"
    Position, CellDim = ReadPOSCARWithVaspIO(BasePoscarPath)

    # Read the base jobsub content once
    JobsubBasePath = WorkDir / "jobsub"
    with JobsubBasePath.open("r", encoding="utf-8") as F:
        JobsubBaseContent = F.read()

    LogLines = []
    LogLines.append("Working Directory Prepared Sucessfully")  # exact spelling as requested

    Temperatures = Params["Temperatures"]
    NSims = Params["NSims"]
    GasRatio = Params["GasRatio"]
    InitO2 = Params["InitO2"]

    ForCopy = ["POTCAR", "job.in", "KPOINTS", "INCAR"]  # POSCAR handled separately after opp.PreparePOSCAR

    for Temp in Temperatures:
        for SimIdx in range(1, NSims + 1):
            FolderName = f"{Temp}_{SimIdx}"
            SimDir = WorkDir / FolderName
            SimDir.mkdir(parents=True, exist_ok=True)

            # 2) Copy POTCAR, jobsub, job.in, KPOINTS, INCAR into the sim folder
            for FN in ForCopy:
                CopyFile(WorkDir / FN, SimDir / FN)

            # Prepare and write jobsub with updated job name for this folder
            FolderTag = MakeFolderTag(FolderName)  # '{Temp}_s_{Sim}'
            JobsubContent = UpdateJobNameInContent(JobsubBaseContent, FolderTag)
            WriteTextFile(SimDir / "jobsub", JobsubContent)

            # 3) Create subfolders
            VolSearchDir = SimDir / "Dir_VolSearch"
            OptUnitDir = SimDir / "Dir_OptUnitCell"
            VolSearchDir.mkdir(parents=True, exist_ok=True)
            OptUnitDir.mkdir(parents=True, exist_ok=True)

            # 4) Create file optunitcell_is_done in Dir_OptUnitCell
            (OptUnitDir / "optunitcell_is_done").touch()

            # 5–7) Prepare new POSCAR (unique per folder) and save into the sim folder
            # Each call should (quasi)randomize O2 placement per your module's behavior.
            NewPosition, NewCellDim = opp.PreparePOSCAR(
                Position, CellDim, GasRatio=GasRatio, InitO2=InitO2
            )
            # Save new POSCAR into the sim folder
            vio.WritePOSCAR(str(SimDir), NewPosition, NewCellDim)

            # 8) Copy POSCAR, jobsub, KPOINTS, POTCAR, job.in into Dir_VolSearch
            # Use the freshly written POSCAR from the sim folder.
            CopyFile(SimDir / "POSCAR", VolSearchDir / "POSCAR")
            CopyFile(SimDir / "jobsub", VolSearchDir / "jobsub")
            for FN in ["KPOINTS", "POTCAR", "job.in", "INCAR"]:
                CopyFile(SimDir / FN, VolSearchDir / FN)

            # 9) Run 'sbatch jobsub' inside Dir_VolSearch
            try:
                Proc = subprocess.run(
                    ["sbatch", "jobsub"],
                    cwd=str(VolSearchDir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                Status = f"OK (exit {Proc.returncode})"
                if Proc.returncode != 0:
                    Status = f"ERROR (exit {Proc.returncode})"
                LogLines.append(f"[{FolderName}] sbatch jobsub -> {Status}")
                if Proc.stdout:
                    LogLines.append(f"[{FolderName}] stdout: {Proc.stdout.strip()}")
                if Proc.stderr:
                    LogLines.append(f"[{FolderName}] stderr: {Proc.stderr.strip()}")
            except FileNotFoundError:
                LogLines.append(f"[{FolderName}] sbatch not found on PATH; skipped submission.")
            except Exception as E:
                LogLines.append(f"[{FolderName}] sbatch execution failed: {E!r}")

    # 10) Create xyz_files folder in the starting directory
    (WorkDir / "xyz_files").mkdir(parents=True, exist_ok=True)

    # 11) Create log file 'log.out' (top line already added)
    WriteTextFile(WorkDir / "log.out", "\n".join(LogLines) + "\n")


def Main():
    PrepareWorkingDirectory()


if __name__ == "__main__":
    Main()
