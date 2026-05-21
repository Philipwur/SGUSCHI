#%%
# Primary Workflow for SGUSCHI. 
# This Code runs every 80 MD steps.

import os
import numpy as np
import pandas as pd
#import shutil
import sys
from pathlib import Path
from typing import Union, Sequence

sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio
from workflow import OxidationAnalysis as an


def ExponentialSmoothing(f1: Union[float, int], f2: Union[float, int], 
                          alpha: float = 0.001) -> float:
    """
    Apply exponential smoothing between two scalar values.

    Args:
        f1 (float or int): The new or current value (latest observation).
        f2 (float or int): The previous smoothed value.
        alpha (float, optional): The smoothing factor in the range [0, 1].

    Returns:
        float: The exponentially smoothed value.
    """
    return f1 * alpha + f2 * (1 - alpha)


def InsertNewVelocities(Velocities: pd.DataFrame,
                                 NewVelocityVectors: Union[np.ndarray, Sequence[Sequence[float]]],
                                 ElementSymbol: str = "O") -> pd.DataFrame:
    """
    Insert new velocity rows immediately after the last existing row of the given element.

    This mirrors the positional insertion policy where new O atoms are appended at the end
    of the 'O' section in `Position`. Assumes the same pre-insertion order as `Position`.

    Parameters
    ----------
    Velocities : pandas.DataFrame
        Existing velocities with columns ['Element', 'vx', 'vy', 'vz'].
    NewVelocityVectors : array-like of shape (K, 3)
        New velocity vectors to insert (e.g., output of MaxwellBoltzmannVelocities).
    ElementSymbol : str, optional
        Element label to insert after (default 'O').

    Returns
    -------
    pandas.DataFrame
        Updated velocities DataFrame with new rows inserted in the correct position.

    """
    RequiredCols = ["Element", "vx", "vy", "vz"]
    if not all(C in Velocities.columns for C in RequiredCols):
        raise ValueError("Velocities must contain columns %r." % RequiredCols)

    NewArr = np.asarray(NewVelocityVectors, dtype=float)
    if NewArr.ndim != 2 or NewArr.shape[1] != 3:
        raise ValueError("NewVelocityVectors must be shape (K, 3).")

    # Build DF for new rows
    NewVelDF = pd.DataFrame(NewArr, columns=["vx", "vy", "vz"])
    NewVelDF.insert(0, "Element", ElementSymbol)

    # Determine insertion index: right after the last existing 'O' (or append if none)
    ElementMask = (Velocities["Element"].values == ElementSymbol)
    if ElementMask.any():
        LastIdx = int(np.where(ElementMask)[0].max())
        InsertPos = LastIdx + 1
    else:
        InsertPos = len(Velocities)

    # Insert by slicing with iloc and reindexing cleanly
    Updated = pd.concat(
        [Velocities.iloc[:InsertPos], NewVelDF, Velocities.iloc[InsertPos:]],
        ignore_index=True
    )
    return Updated


def CreateGassesRemovedStr(Gasses) -> str:
    """
    Return a repr string of all non-O2 molecules found in the frame, e.g.:
    "[('C', 'O', 'O'), ('H', 'O')]"
    """
    if isinstance(Gasses, tuple):
        Gasses = Gasses[0]
    if Gasses is None or Gasses.empty:
        return "[]"

    NonO2 = [tuple(M) for M in Gasses["Molecule"].tolist() if tuple(M) != ("O", "O")]
    
    return repr(NonO2)


def CheckSimulationEnvironment(WorkDir: Path) -> None:
    
    #Function which checks if:
    #RateAnalysis.csv matches the number of folders in WorkDir.
        #If not, runs FixRateAnalysis and FixXYZ to repair the folder
    #Any Folders are missing.
        #Produces error, reccomends rollback to earlis.

    StepFolders = [
        int(d.name)
        for d in WorkDir.iterdir() 
        if d.is_dir() and d.name.isdigit()]
    
    #Check for missing folders
    if len(StepFolders) != 1:
        
        SortedSteps = sorted(set(StepFolders))
        
        StepSet = set(SortedSteps)
       
        ExpectedRange = range(SortedSteps[0], SortedSteps[-1] + 1)
        MissingSteps = [Step for Step in ExpectedRange if Step not in StepSet]
        
        IsConsecutive = len(MissingSteps) == 0
        
        if not IsConsecutive:
            print(f'Missing Step Folders in {WorkDir}: {MissingSteps}.\nRollBack to {min(MissingSteps) - 1} for complete RateAnalysis and XYZ.')
        
    LatestFolder = max(StepFolders)
    
    try:
        RateAnalysis = vio.ReadRateAnalysis(WorkDir / 'RateAnalysis.csv')
        RateAnalysisSize = len(RateAnalysis)
    except:
        RateAnalysisSize = 1
    
    # Check if RateAnalysis size matches folder count
    if RateAnalysisSize != LatestFolder:
        
        print('RateAnalysis.csv entries do not match Dir_VolSearch.')
        
        #from utils.FixRateAnalysis import FixRateAnalysis
        #from utils.FixXYZ import FixXYZ
        #
        #print('Running FixRateAnalysis...')
        #FixRateAnalysis(WorkDir)
        #print('Done.\nRunning FixXYZ...')
        #FixXYZ(WorkDir)
        #print('Done.')
        #
        #RateAnalysis = vio.ReadRateAnalysis(WorkDir / 'RateAnalysis.csv')
        #RateAnalysisSize = len(RateAnalysis)
    #
        #if RateAnalysisSize != LatestFolder:
        #    raise ValueError('RateAnalysis.csv entries still do not match Dir_VolSearch after Fix utilities.\nFATAL: RollBack required.')
    #
 

def ValidateNoUnknownElements(Position: pd.DataFrame, Context: str) -> None:
    """Fail before bonding logic sees unresolved element labels."""
    if "Element" not in Position.columns:
        raise ValueError(f"{Context} is missing an Element column.")

    Elements = Position["Element"].astype(str).str.strip()
    Unknown = sorted(set(Elements[(Elements == "") | (Elements == "X")].tolist()))
    if Unknown:
        raise ValueError(f"{Context} contains unresolved element labels: {Unknown}.")


def ValidateOutcarData(OutcarData: dict, OutcarPath: Path) -> None:
    """Validate parsed OUTCAR data before any workflow files are mutated."""
    Positions = OutcarData.get("Positions")
    TimesFs = OutcarData.get("TimesFs")

    if not Positions:
        raise ValueError(f"No positions parsed from {OutcarPath}.")
    if not TimesFs:
        raise ValueError(f"No ionic step times parsed from {OutcarPath}.")
    if len(TimesFs) != len(Positions):
        raise ValueError(
            f"Parsed OUTCAR times/positions mismatch for {OutcarPath}: "
            f"{len(TimesFs)} times vs {len(Positions)} frames."
        )

    for FrameIndex, FramePosition in enumerate(Positions, start=1):
        ValidateNoUnknownElements(
            FramePosition,
            f"OUTCAR frame {FrameIndex} in {OutcarPath}"
        )


def main(WorkDir = None, TestCase = False):
    
    """
    Main Function, gets called every 80 molecular dynamics steps. 
    
    Searches POSCAR file for O2, CO and CO2.
    Exponentially smoothes O2.  
    
    Checks POSCAR file for the existence of Carbon gas each SLUSCHI step.
    Prefroms rate analysis to estimate rate and condifdence intervals.
    Keeps track of parameters in outputted RateAnalysis.csv. 
    Gets called in Dir_VolSearch. Partial pressure for OxRate calculated by 
    target O2

    Parameters:
        WorkDir (string): The location for Dir_Volsearch, where oxidation calculations are taking place
        TestCase (boolean): Option to run main without updating Files 
                            
    """
    
    if WorkDir == None:
        WorkDir = os.getcwd()
    
    WorkDir = Path(WorkDir).resolve()
    
    #Location of Radii, Oxparams, Results/, xyz_files/ etc
    RootDir = WorkDir.parents[1]
    TrajectoryName = WorkDir.parent.name
    
    #Verify no missing folders, RateAnalysis coherence
    #CheckSimulationEnvironment(WorkDir)
    
    #------------------------- Gather Hyperparameters -------------------------
        
    OxParamsPath = RootDir / 'OxParams'
    
    if not OxParamsPath.exists():
        raise FileNotFoundError(f'OxParams file not found in {OxParamsPath}.')
    
    OxParams = vio.ReadKeyValueFile(OxParamsPath, 
                                    RequiredKeys = ['AtomicRadiusTol', 
                                                    'O2Tol',
                                                    'OSmoothing',
                                                    'GasRatio',
                                                    'InitO2Count'])
    
    AtomicRadiusTol = float(OxParams['AtomicRadiusTol'])
    O2Tol = float(OxParams['O2Tol'])
    OxygenSmoothing = float(OxParams['OSmoothing'])
    GasRatio = float(OxParams['GasRatio'])
    InitO2Count = int(OxParams['InitO2Count'])
        
    #Collect Radii for Bond Algo
    
    CovalentRadiiPath = RootDir / 'CovalentRadii'
    
    if not CovalentRadiiPath.exists():
        raise FileNotFoundError(f'CovalentRadii file not found in {RootDir}.')
    
    CovalentRadii = vio.ReadKeyValueFile(CovalentRadiiPath)
    CovalentRadii = {k: float(v) for k, v in CovalentRadii.items()}
    
    #Read RateAnalysis
    RateAnalysisPath = WorkDir / 'RateAnalysis.csv'
    
    try:
        RateAnalysis = vio.ReadRateAnalysis(RateAnalysisPath)

    except:
        RateAnalysis = pd.DataFrame([{'Time (fs)': 0,
                                     'O2 Count': InitO2Count,
                                     'Smoothed O2 Count': InitO2Count,
                                     'O2 Added': InitO2Count,
                                     'Gas Removed': '[]',
                                     'Free Gas Fraction': 1
                                     }])
    
    LatestFolder = len(RateAnalysis) #Prudent to add manual check of folder count
    LatestFolderPath = WorkDir / str(LatestFolder)
    if not LatestFolderPath.is_dir():
        raise FileNotFoundError(
            f"Expected completed job folder {LatestFolderPath} does not exist."
        )
    print(f'Expecting to read folder {LatestFolder}')
    #------------------------- Gather Run Information -------------------------


    #Read POSCAR of last jobs (in working directory)
    #Edited by SLUSCHI, build next POSCAR from this one
    PoscarData = vio.ReadPoscar(WorkDir, GiveVelocities = True)
    if len(PoscarData) != 3:
        raise ValueError(f"POSCAR in {WorkDir} does not contain velocities.")
    Position, CellDim, Velocity = PoscarData
    
    #Rename Elements in case of corruption
    Position = vio.FixElementFormatting(Position)
    ValidateNoUnknownElements(Position, f"POSCAR in {WorkDir}")

    OutcarData = vio.OutcarParser(LatestFolderPath)
    ValidateOutcarData(OutcarData, LatestFolderPath / "OUTCAR")

    Temperature = OutcarData['Temperature']
    if Temperature is None:
        raise ValueError(f"No target temperature parsed from {LatestFolderPath / 'OUTCAR'}.")
    SimTime = OutcarData['TimesFs'][-1]
    
    
    #----------------------------- Analysis Steps -----------------------------
    
    
    GasFraction = an.CalculateGasFraction(Position, GasRatio)
    
    Gasses = an.FindGases(Position, 
                          CellDim, 
                          CovalentRadii = CovalentRadii,
                          AtomicRadiusTol = AtomicRadiusTol, 
                          MinimumComplexity = 2,
                          MaximumComplexity = 3,
                          ReturnBondMatrix = False)
    
    #Ensure the Molecule column is tuple-typed so equality checks behave
    if (Gasses is not None) and (not Gasses.empty) and ("Molecule" in Gasses.columns):
        Gasses = Gasses.copy()
        Gasses["Molecule"] = Gasses["Molecule"].apply(lambda M: tuple(M) if not isinstance(M, tuple) else M)

    Position, Velocity = an.RemoveNonO2Gasses(Position,
                                                Velocity,
                                                Gasses)
    
    O2Tol = O2Tol * GasFraction
    
    SmoothedO2Count = RateAnalysis['Smoothed O2 Count'].iloc[-1]
    
    #Smooth O for each frame of outcar
    for OutcarPosition in OutcarData['Positions']:
        
        FrameGasses = an.FindGases(OutcarPosition, 
                                   CellDim, 
                                   CovalentRadii = CovalentRadii,
                                   AtomicRadiusTol = AtomicRadiusTol, 
                                   MinimumComplexity = 2,
                                   MaximumComplexity = 3,
                                   ReturnBondMatrix = False)
        
        # Explicitly handle empty dataframes and use .apply for safe tuple comparison
        if FrameGasses is None or FrameGasses.empty:
            O2Count = 0
        else:
            # .apply(lambda x: ...) avoids Pandas broadcasting errors when comparing tuples
            IsO2 = FrameGasses['Molecule'].apply(lambda x: tuple(x) == ('O', 'O'))
            O2Count = len(FrameGasses[IsO2].index)
    
        SmoothedO2Count = ExponentialSmoothing(O2Count, 
                                               SmoothedO2Count,
                                               alpha = OxygenSmoothing)
    
    #Condition met to add 1 O2
    if SmoothedO2Count <= O2Tol and O2Count < O2Tol:
        print(f'Adding O2 Molecule: {SmoothedO2Count} <= {O2Tol}')
        
        OptimalSite = an.FindOptimalCoords(Position,
                                           CellDim,
                                           N = 1)
        
        Position = an.PlaceO2Molecules(Position,
                                    CellDim,
                                    OptimalSite,
                                    BondLength = 1.2)
        
        NewVelocity = an.MaxwellBoltzmannVelocities(['O', 'O'],
                                                    Temperature)

        Velocity = InsertNewVelocities(Velocity,
                                       NewVelocity,
                                       ElementSymbol = 'O')
        
        #WAVECAR no longer good starting point for calculation
        #Only delete if this is NOT a test case
        if not TestCase and os.path.exists(f'{WorkDir}/WAVECAR'):
            os.remove(f'{WorkDir}/WAVECAR')
            
        O2Count += 1
        O2Added = 1
    else:
        O2Added = 0
        
    #----------------------------- File Management ----------------------------
    
    GasRemovedStr = CreateGassesRemovedStr(Gasses)
    
    NewRateRow = [
        SimTime + RateAnalysis['Time (fs)'].iloc[-1],
        O2Count,
        SmoothedO2Count,
        O2Added + RateAnalysis['O2 Added'].iloc[-1],
        GasRemovedStr,
        GasFraction
    ]
    
    NewRateRow = pd.DataFrame([NewRateRow], columns=RateAnalysis.columns)
    
    if not TestCase:
        
        # Update xyz
        #PathToXYZ = os.path.join(RootDir, 'xyz_files', f'{TrajectoryName}.xyz')
        
        XYZPath = RootDir / 'xyz_files' / f'{TrajectoryName}.xyz'

        vio.WriteXYZ(OutcarData, FilePath = XYZPath)
        
        # Update RateAnalysis
        RateAnalysis = pd.concat([RateAnalysis, NewRateRow],
                                ignore_index = True)
        RateAnalysis.to_csv(RateAnalysisPath, index = False)
        
        PathToResults = RootDir / 'xyz_files' / f'RateAnalysis_{TrajectoryName}.csv'
        
        RateAnalysis.to_csv(PathToResults, index = False)
        
        # Update POSCAR
        vio.WritePoscar(WorkDir, Position, CellDim, Velocity)
        
        # Could come up with some solution to also include other outcar data 
        # but meh, we probably wont use it anyways.
        
    # Prevents any creation of new files and dumps variables
    if TestCase:
        
        TestOutPath = WorkDir / 'test.out'
        print(f'Running in Test Mode. Writing variables to {TestOutPath}...')
        
        with open(TestOutPath, 'w') as TestFile:
            TestFile.write(f'--- Test Run Output ---\n')
            TestFile.write(f'WorkDir: {WorkDir}\n')
            TestFile.write(f'LatestFolder (Step): {LatestFolder}\n')
            TestFile.write(f'TrajectoryName: {TrajectoryName}\n\n')
            
            TestFile.write(f'--- Simulation Parameters ---\n')
            TestFile.write(f'Temperature: {Temperature}\n')
            TestFile.write(f'SimTime: {SimTime}\n')
            TestFile.write(f'GasFraction: {GasFraction}\n\n')
            
            TestFile.write(f'--- Oxygen Analysis ---\n')
            TestFile.write(f'SmoothedO2Count: {SmoothedO2Count}\n')
            TestFile.write(f'O2Count (Actual): {O2Count}\n')
            TestFile.write(f'O2Tol (Adjusted): {O2Tol}\n')
            TestFile.write(f'O2Added (Bool): {O2Added}\n\n')
            
            TestFile.write(f'--- Gas Removal ---\n')
            TestFile.write(f'GasRemovedStr: {GasRemovedStr}\n\n')
            
            TestFile.write(f'--- Pending DataFrame Update ---\n')
            TestFile.write(NewRateRow.to_string(index=False))
            TestFile.write('\n')
            
        print('Test run complete.')
 
    
if __name__ == '__main__':
    
    WorkDir = os.getcwd() #use in prod 
    
    # Check for 'test' argument for file changes
    TestMode = 'test' in [Arg.lower() for Arg in sys.argv]
    
    try:
        main(WorkDir, TestCase = TestMode)
    except Exception as Exc:
        print(f"FATAL OxidationStep failed in {WorkDir}: {Exc}", file=sys.stderr)
        raise
    
    #Trial Fixing RateAnalysis
    #WorkDir = 'SLUSCHI_Oxidation_Test_25_1273K_10O_1/Dir_VolSearch' #use for demos     
    #FixRateAnalysis(WorkDir)
    

    
# %%
