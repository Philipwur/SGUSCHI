import os
import sys
from pathlib import Path
from typing import Union

try:
    from tqdm import tqdm as tqdm
except ImportError:
    tqdm = None

sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio


'''
-------------------------------------------------------------------------------
How to use this script:

From any given directory you can run:
> python path/to/SGUSCHI/src/utils/FixXYZ.py /path/to/Dir_VolSearch

Or if you’re already in Dir_VolSearch:
> python src/utils/FixXYZ.py

There's a lot of IO required for this script, so expect about 1 minute per 50
VolSearch folders.

tqdm functionality is optional, other imports arent.
-------------------------------------------------------------------------------
'''


def FixXYZ(WorkDir: Union[str, Path] = None) -> Path:
    """
    Rebuild the trajectory XYZ file for a Dir_VolSearch-like working directory.

    Behaviour:
        - Detects RootDir as two levels above WorkDir.
        - Determines TrajectoryName from the parent folder of WorkDir.
        - Collects all numbered subfolders (1, 2, 3, ...) in WorkDir.
        - Safety Check: Verifies all folders contain POSCAR AND OUTCAR.
        - For each step folder, reads OUTCAR data via vio.OutcarParser.
        - Calls vio.WriteXYZ in ascending step order to reconstruct the XYZ.
        - Overwrites RootDir / 'xyz_files' / f'{TrajectoryName}.xyz'.

    Notes:
        - Only the XYZ file is created/overwritten.
        - No POSCAR / WAVECAR / RateAnalysis or other files are touched.
    """
    if WorkDir is None:
        WorkDir = os.getcwd()

    WorkDir = Path(WorkDir).resolve()
    RootDir = WorkDir.parents[1]
    TrajectoryName = WorkDir.parent.name

    XYZDir = RootDir / "xyz_files"
    XYZDir.mkdir(parents=True, exist_ok=True)
    XYZPath = XYZDir / f"{TrajectoryName}.xyz"

    # Collect numbered step folders
    StepFolders = sorted(
        int(Directory.name)
        for Directory in WorkDir.iterdir()
        if Directory.is_dir() and Directory.name.isdigit()
    )

    if not StepFolders:
        raise FileNotFoundError(
            f"No numbered step folders (e.g. '1', '2', ...) found in {WorkDir}"
        )

    # --- SAFETY CHECK: Check for incomplete folders before deleting anything ---
    IncompleteFolders = []
    for Step in StepFolders:
        StepPath = WorkDir / str(Step)
        PoscarPath = StepPath / 'POSCAR'
        OutcarPath = StepPath / 'OUTCAR'
        
        # Check both files
        if (not PoscarPath.exists()) or (not OutcarPath.exists()):
            IncompleteFolders.append(Step)

    if IncompleteFolders:
        print(f"\nError: Incomplete MD folders found (Steps: {IncompleteFolders}).")
        print("Missing POSCAR or OUTCAR. Potential loss of data.")
        print("Cancelling XYZ rebuild. Best solution is stitching new and old XYZ files together.\n")
        sys.exit(1)
    # -------------------------------------------------------------------------

    # Remove any existing XYZ so we can rebuild from scratch
    # Only done AFTER the safety check passed
    if XYZPath.exists():
        XYZPath.unlink()

    StepIterable = StepFolders
    if tqdm is not None:
        StepIterable = tqdm(StepFolders, desc="FixXYZ", unit="step")

    # Rebuild XYZ in chronological order
    for StepIndex in StepIterable:
        OutcarPath = WorkDir / str(StepIndex)
        OutcarData = vio.OutcarParser(OutcarPath)
        # Assumes vio.WriteXYZ handles append / create semantics internally
        vio.WriteXYZ(OutcarData, FilePath=XYZPath)

    return XYZPath


if __name__ == "__main__":
    if len(sys.argv) > 1:
        WorkDirArgument = sys.argv[1]
    else:
        WorkDirArgument = os.getcwd()

    XYZFilePath = FixXYZ(WorkDirArgument)
    print(f"Rebuilt XYZ file at: {XYZFilePath}")