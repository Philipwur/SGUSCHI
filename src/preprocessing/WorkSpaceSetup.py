# %%
"""
This Code Will Set Up the Work Space according to the OxParams file.

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
                     - InitO2
                     - GasRatio
                     - Temperatures
                     - NSims
2. POSCAR          → Base structure file (Supercell Expanded)
3. KPOINTS         → VASP KPOINTS input
4. POTCAR          → VASP pseudopotentials (Same order as POSCAR + O Last)
5. INCAR           → VASP INCAR input
6. job.in          → SLUSCHI job file
7. jobsub          → SLURM job submission script
8. IonicRadii      → File containing ionic radii for elements
9. jobsub_master   → Master process slurm job submission script
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, List  # ✅ for 3.9-safe typing

# --- Make imports location-independent ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio
import OxidationPreprocessing as opp
# from workflow import OxidationAnalysis as an


def ReadOxParams(FilePath: Path) -> Dict[str, object]:
    """
    Parse OxParams key=value lines.

    Supports Temperatures specified either with or without brackets, e.g.:
      Temperatures = [873, 973, 1073, 1273]
      Temperatures = 873, 973, 1073, 1273

    Returns:
      {
        "Temperatures": ["873", "973", ...],  # strings (safe for folder names)
        "NSims": int,
        "GasRatio": float,
        "InitO2": int,
        # Optional passthroughs if present:
        "AtomicRadiusTol": float,
        "TargetPP": int,
        "PPSmoothing": float,
      }
    """
    if not FilePath.exists():
        raise FileNotFoundError("OxParams not found at: {}".format(FilePath))

    # --- Read raw key/value pairs ---
    RawParams: Dict[str, str] = {}
    with FilePath.open("r", encoding="utf-8") as File:
        for Line in File:
            Line = Line.strip()
            if not Line or Line.startswith("#") or "=" not in Line:
                continue
            Key, Value = [x.strip() for x in Line.split("=", 1)]
            RawParams[Key] = Value

    # --- Helper functions ---
    def ParseTemperatures(Value: str) -> List[str]:
        """Extract numeric temperature tokens from comma/space-separated lists with or without brackets."""
        Cleaned = Value.strip().strip("[](){}")
        Tokens = re.split(r"[,\s]+", Cleaned)
        Temperatures: List[str] = []
        for Token in Tokens:
            if not Token:
                continue
            Match = re.search(r"-?\d+(?:\.\d+)?", Token)
            if Match:
                Temperatures.append(Match.group(0))
        if not Temperatures:
            raise ValueError("Could not parse Temperatures from: {!r}".format(Value))
        return Temperatures

    def Require(Key: str) -> str:
        if Key not in RawParams or RawParams[Key] == "":
            raise ValueError("{} must be provided in OxParams.".format(Key))
        return RawParams[Key]

    # --- Parse required fields ---
    Temperatures = ParseTemperatures(Require("Temperatures"))

    try:
        NSims = int(Require("NSims"))
    except ValueError as Err:
        raise ValueError("NSims must be an integer.") from Err

    try:
        GasRatio = float(Require("GasRatio"))
    except ValueError as Err:
        raise ValueError("GasRatio must be a float.") from Err

    try:
        InitO2 = int(Require("InitO2"))
    except ValueError as Err:
        raise ValueError("InitO2 must be an integer.") from Err

    # --- Optional passthroughs ---
    ParamsOut: Dict[str, object] = {
        "Temperatures": Temperatures,
        "NSims": NSims,
        "GasRatio": GasRatio,
        "InitO2": InitO2,
    }

    if "AtomicRadiusTol" in RawParams and RawParams["AtomicRadiusTol"]:
        try:
            ParamsOut["AtomicRadiusTol"] = float(RawParams["AtomicRadiusTol"])
        except ValueError:
            pass  # ignore if not numeric

    if "TargetPP" in RawParams and RawParams["TargetPP"]:
        try:
            ParamsOut["TargetPP"] = int(float(RawParams["TargetPP"]))
        except ValueError:
            pass

    if "PPSmoothing" in RawParams and RawParams["PPSmoothing"]:
        try:
            ParamsOut["PPSmoothing"] = float(RawParams["PPSmoothing"])
        except ValueError:
            pass

    return ParamsOut


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


def PrepareWorkingDirectory():
    """Main setup routine to prepare simulation directories."""
    workdir = Path.cwd()
    oxparams = ReadOxParams(workdir / "OxParams")

    # Verify input files
    required_files = ["POSCAR", "KPOINTS", "POTCAR", "INCAR", "job.in", "jobsub"]
    EnsureFilesExist(workdir, required_files)

    # Load base POSCAR via the provided function
    position, celldim = vio.ReadPOSCAR(workdir=str(workdir), filename="POSCAR")

    # Load base jobsub content
    jobsub_base = (workdir / "jobsub").read_text(encoding="utf-8")

    log_lines = ["Working Directory Prepared Sucessfully"]  # exact spelling

    for temp in oxparams["Temperatures"]:
        for sim_idx in range(1, oxparams["NSims"] + 1):
            folder_name = "{}_{}".format(temp, sim_idx)
            sim_dir = workdir / folder_name
            sim_dir.mkdir(exist_ok=True)

            # Copy common files
            for fn in ["POTCAR", "job.in", "KPOINTS", "INCAR"]:
                CopyFile(workdir / fn, sim_dir / fn)

            # Modify jobsub job name
            tag = MakeFolderTag(folder_name)
            jobsub_text = UpdateJobName(jobsub_base, tag)
            (sim_dir / "jobsub").write_text(jobsub_text, encoding="utf-8")

            # Create Dir_VolSearch and Dir_OptUnitCell
            vol_dir = sim_dir / "Dir_VolSearch"
            opt_dir = sim_dir / "Dir_OptUnitCell"
            vol_dir.mkdir(exist_ok=True)
            opt_dir.mkdir(exist_ok=True)

            # optunitcell_is_done marker
            (opt_dir / "optunitcell_is_done").touch()

            # Prepare unique POSCAR
            new_pos, new_cell = opp.PreparePOSCAR(
                position, celldim,
                GasRatio=oxparams["GasRatio"],
                InitO2=oxparams["InitO2"]
            )

            # Save it
            vio.WritePOSCAR(str(sim_dir), new_pos, new_cell)

            # Copy into Dir_VolSearch
            for fn in ["POSCAR", "jobsub", "KPOINTS", "POTCAR", "job.in", "INCAR"]:
                CopyFile(sim_dir / fn, vol_dir / fn)

            # --- Job submission (currently disabled for safety) ---
            '''
            try:
                proc = subprocess.run(
                    ["sbatch", "jobsub"],
                    cwd=str(vol_dir),
                    capture_output=True,
                    text=True
                )
                log_lines.append("[{}] sbatch exit {}".format(folder_name, proc.returncode))
                if proc.stdout:
                    log_lines.append("stdout: {}".format(proc.stdout.strip()))
                if proc.stderr:
                    log_lines.append("stderr: {}".format(proc.stderr.strip()))
            except FileNotFoundError:
                log_lines.append("[{}] sbatch not found; skipped.".format(folder_name))
            '''

    # xyz_files folder
    (workdir / "xyz_files").mkdir(exist_ok=True)

    # log.out
    (workdir / "log.out").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    PrepareWorkingDirectory()
