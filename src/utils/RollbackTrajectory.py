import os
import sys
import shutil
import datetime
from pathlib import Path
from typing import Union

# Add 'src' to path so we can import from workflow and utils
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.FixXYZ import FixXYZ
from utils.FixRateAnalysis import FixRateAnalysis
from utils.FolderUtils import NumericStepFolders

'''
-------------------------------------------------------------------------------
How to use this script:

From any given Dir_VolSearch directory you can run:
> python src/utils/RollbackTrajectory.py [TargetStep]
> python src/utils/RollbackTrajectory.py [WorkDir] [TargetStep]

Example:
> python src/utils/RollbackTrajectory.py 200

Actions taken:
1. TargetStep (200) is considered the LAST GOOD STEP.
2. The script looks for the start of the NEXT step (201/POSCAR) to use as the 
   continuation point (since 200/CONTCAR is not saved).
3. 201/POSCAR is copied to WorkDir/POSCAR.
4. All folders > 200 (e.g., 201, 202) are DELETED.
5. WorkDir/WAVECAR is DELETED.
6. RateAnalysis and XYZ files are repaired.

Notes: 
1. If 201/POSCAR does not exist (e.g., 200 crashed before creating it),
you must rollback to 199.
2. Since we use the next folders's poscar as starting point, gasses which are 
added or removed in this step won't show up in the rateanalysis since oxidation 
step has already adjusted its poscar in the failed simulation.
To prevent this ensure you select a frame in which no gasses are added/removed 
between folder n and n+1.
-------------------------------------------------------------------------------
'''

def RollbackTrajectory(WorkDir: Union[str, Path] = None, TargetStep: int = 0) -> None:
    
    if WorkDir is None:
        WorkDir = os.getcwd()
        
    WorkDir = Path(WorkDir).resolve()
    
    # 1. Validate Target Folder
    TargetFolder = WorkDir / str(TargetStep)
    if not TargetFolder.exists():
        raise FileNotFoundError(f"Target folder {TargetStep} does not exist in {WorkDir}.")
    
    # 2. Secure the POSCAR from the Next Step (TargetStep + 1)
    # Since CONTCARs are not saved, 201/POSCAR represents the end of 200. (post-oxidationstep)
    NextStep = TargetStep + 1
    SourcePoscar = WorkDir / str(NextStep) / 'POSCAR'
    DestPoscar = WorkDir / 'POSCAR'
    
    if not SourcePoscar.exists():
        print(f"\nCRITICAL ERROR: To rollback to folder {TargetStep}, the poscar from {NextStep} is required as new starting point.")
        print(f"Please rollback further (e.g. to {TargetStep - 1}) or put a suitable poscar in folder {NextStep}.\n")
        # Exit script safely without deleting anything
        sys.exit(1)
        
    print(f"Securing geometry from {SourcePoscar} to {DestPoscar}...")
    shutil.copy(SourcePoscar, DestPoscar)

    # 3. Identify and Delete Future Folders
    StepFolders = NumericStepFolders(WorkDir)
    
    # We keep TargetStep, we delete everything strictly greater than it
    FoldersToRemove = [step for step in StepFolders if step > TargetStep]
    
    if not FoldersToRemove:
        print(f"No folders found after step {TargetStep} to delete.")
    else:
        print(f"Deleting {len(FoldersToRemove)} future folders (Steps > {TargetStep})...")
        for step in sorted(FoldersToRemove):
            FolderParam = WorkDir / str(step)
            print(f"  - Removing {FolderParam}")
            shutil.rmtree(FolderParam)

    # 4. Remove WAVECAR
    # The existing WAVECAR belongs to a later step and will cause mismatches.
    WavecarPath = WorkDir / 'WAVECAR'
    if WavecarPath.exists():
        print(f"Removing incompatible WAVECAR at {WavecarPath}...")
        os.remove(WavecarPath)

    # 5. Update Log (One folder above WorkDir)
    LogPath = WorkDir.parent / 'log.out'
    Timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LogMessage = f"\n[{Timestamp}] Simulation rolled back to end of step {TargetStep} (using geometry from {NextStep}).\n"
    
    try:
        with open(LogPath, 'a') as f:
            f.write('\n' + '-' * 80 + '\n')
            f.write(LogMessage)
            f.write('\n' + '-' * 80 + '\n')
        print(f"Logged rollback to {LogPath}")
    except Exception as e:
        print(f"Warning: Could not write to log.out: {e}")

    # 6. Repair Data Files
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