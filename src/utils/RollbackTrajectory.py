import os
import sys
import shutil
import datetime
from pathlib import Path
from typing import Union

# Add 'src' to path so we can import from workflow and utils
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import the provided utility scripts
# Assumes FixXYZ.py and FixRateAnalysis.py are in the same package (src.utils)
from utils.FixXYZ import FixXYZ
from utils.FixRateAnalysis import FixRateAnalysis
from workflow import VaspIO as vio

'''
-------------------------------------------------------------------------------
How to use this script:

From any given SGUSCHI working (Dir_VolSearch) directory you can run:
> python src/utils/RollbackTrajectory.py [TargetStep]
> python src/utils/RollbackTrajectory.py [WorkDir] [TargetStep]

Examples:
> python src/utils/RollbackTrajectory.py 200
  (Rolls back current directory to step 200)

> python src/utils/RollbackTrajectory.py /path/to/Dir_VolSearch 150
  (Rolls back specified directory to step 150)

This script:
1. Deletes all step folders > TargetStep.
2. Copies TargetStep/POSCAR to WorkDir/POSCAR (Resetting geometry).
3. Updates log.out in the parent directory.
4. Runs FixXYZ and FixRateAnalysis to clean up the data files.
-------------------------------------------------------------------------------
'''

def RollbackTrajectory(WorkDir: Union[str, Path] = None, TargetStep: int = 0) -> None:
    """
    Reverts the simulation state to a specific step.

    Args:
        WorkDir (Path): The working directory (Dir_VolSearch).
        TargetStep (int): The step number to revert to. This folder will be KEPT,
                          and its POSCAR will become the new starting point.
    """
    
    if WorkDir is None:
        WorkDir = os.getcwd()
        
    WorkDir = Path(WorkDir).resolve()
    
    # 1. Validate Target Folder
    TargetFolder = WorkDir / str(TargetStep)
    if not TargetFolder.exists():
        raise FileNotFoundError(f"Target folder {TargetStep} does not exist in {WorkDir}.")
    
    # 2. Identify and Delete Future Folders
    StepFolders = [
        int(d.name) for d in WorkDir.iterdir() 
        if d.is_dir() and d.name.isdigit()
    ]
    
    FoldersToRemove = [step for step in StepFolders if step > TargetStep]
    
    if not FoldersToRemove:
        print(f"No folders found after step {TargetStep}. Nothing to delete.")
    else:
        print(f"Deleting {len(FoldersToRemove)} future folders...")
        for step in sorted(FoldersToRemove):
            FolderParam = WorkDir / str(step)
            print(f"  - Removing {FolderParam}")
            shutil.rmtree(FolderParam)

    # 3. Reset POSCAR
    # "Uses its [TargetFolder] POSCAR as the starting point"
    SourcePoscar = TargetFolder / 'POSCAR'
    DestPoscar = WorkDir / 'POSCAR'
    
    if SourcePoscar.exists():
        print(f"Resetting root POSCAR from {SourcePoscar}...")
        shutil.copy(SourcePoscar, DestPoscar)
    else:
        raise FileNotFoundError(f"POSCAR not found in target folder {TargetFolder}. Cannot reset simulation.")

    # 4. Update Log (One folder above WorkDir)
    LogPath = WorkDir.parent / 'log.out'
    Timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LogMessage = f"[{Timestamp}] Simulation rolled back to folder {TargetStep}.\n"
    
    try:
        with open(LogPath, 'a') as f:
            f.write(LogMessage)
        print(f"Logged rollback to {LogPath}")
    except Exception as e:
        print(f"Warning: Could not write to log.out: {e}")

    # 5. Repair Data Files
    print("Running FixXYZ...")
    FixXYZ(WorkDir)
    
    print("Running FixRateAnalysis...")
    FixRateAnalysis(WorkDir)

    print("Rollback complete.")


if __name__ == "__main__":
    
    # Argument Parsing Logic
    Args = sys.argv[1:]
    
    WorkDirArg = None
    TargetStepArg = None
    
    if len(Args) == 1:
        # Case: python RollbackTrajectory.py 200
        if Args[0].isdigit():
            WorkDirArg = os.getcwd()
            TargetStepArg = int(Args[0])
        else:
            print("Error: Single argument must be the Target Step integer.")
            sys.exit(1)
            
    elif len(Args) == 2:
        # Case: python RollbackTrajectory.py /path/to/dir 200
        WorkDirArg = Args[0]
        if Args[1].isdigit():
            TargetStepArg = int(Args[1])
        else:
            print("Error: Second argument must be the Target Step integer.")
            sys.exit(1)
    else:
        print("Usage: python RollbackTrajectory.py [Optional: WorkDir] [TargetStep]")
        sys.exit(1)
        
    # Execute
    RollbackTrajectory(WorkDirArg, TargetStepArg)