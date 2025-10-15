
#%%
#python script for Managing oxidation workflow within SLUSCHI.

import os
import numpy as np
import pandas as pd
import shutil

import VaspIO as vio
import OxidationAnalysis as an

def ReadOxParams(WorkDir = None):
    
    '''
    Function which allows for the setting of different oxidation hyperparamers
    across different oxidation simulations. 
    Currently only AtomicRadiusTolerance, O2tolerance and O2 Smoothing factor 
    can be set. Uses same format as INCAR. 
    '''
    
    if WorkDir == None:
        WorkDir = os.getcwd()
    
    OxParamsPath = os.path.join(WorkDir, '..', 'OxParams')
    
    AtomicRadiusTol, O2Tol, O2Smoothing = vio.INCARParser(Parameters = ['AtomicRadiusTol', 
                                                                        'O2Tol', 
                                                                        'O2Smoothing'],
                                                          FilePath = OxParamsPath)

    return AtomicRadiusTol, O2Tol, O2Smoothing
    
    
def FixElementFormatting(Position, ReturnPrevNames = False):
    
    '''
    A function which fixes the element names in a position/velocity DataFrame so
    elements can be comprehended by bond finding algorithms later. 
    
    Parameters:
        Position (DataFrame): Atom positions with 'Element' and fractional 
                              coordinates 'x', 'y', 'z'. Can also be Velocity.
        ReturnPrevNames (boolean): Condition on whether to output old names, 
                                   should be set to T if renaming back is needed.
    '''
    
    if ReturnPrevNames == True:
        PrevNames = Position['Element'].unique()
            
    for i in Position['Element'].unique():
        if '_' in i:
            FixedName = i.split('_')[0]
            Position.loc[Position['Element'] == i, 'Element'] = FixedName
        elif '/' in i:
            FixedName = i.split('/')[0]
            Position.loc[Position['Element'] == i, 'Element'] = FixedName

    if ReturnPrevNames == True:
        return Position, PrevNames
    else:
        return Position


def RemoveCO(Position, CCoords):
    
    '''
    Function which takes the indices of identified carbon gases and removes
    them. Can be used on Velocity and Position file.
    
    Parameters:
        Position (DataFrame): Atom 3D positions or velocity with 'Element' column
        CCoords (DataFrame): A dataframe with 'Molecule' compositions and 'Indices'
                             of atoms in molecule. Both columns are composed of tuples.
    
    Returns:
        Position (DataFrame): Position or Velocity DataFrame with removed indicies.
    '''

    for _, row in CCoords.iterrows():
        Position = Position.drop(list(row['Indices']))
        
    return Position.reset_index(drop = True)


def UnFixElementFormatting(Position, PrevNames):
    
    '''
    A function to rename the elements in the position and AtomInfo dataframes 
    back to their old names.
    
    Parameters:
        Position (DataFrame): DataFrame with the 'Element' Column to rename
        PrevNames (DataFrame): DataFrame with previous 'Element' names
        
    Returns: 
        Position (DataFrame): DataFrame with Elements reverted to previous names
                              renamed by the FixElementFormatting function.
    '''
    
    NewNames = sorted(Position['Element'].unique())
    PrevNames = sorted(PrevNames)
    
    for i in range(len(NewNames)):
        
        NewName = NewNames[i]
        PrevName = PrevNames[i]
        
        Position.loc[Position['Element'] == NewName, 'Element'] = PrevName 
        
    return Position
      

def ExponentialSmoothing(f1, f2, alpha = 0.001):
    #Exponential smoothing function
    return f1 * alpha + f2 * (1 - alpha)


def ReadRateAnalysis(WorkDir = None):

    '''
    A function which returns a parsed RateAnalysis file with the total time and 
    temperature. 
    
    Parameters:

    '''
    
    if WorkDir == None:
        WorkDir = os.getcwd()
    
    if os.path.isfile(os.path.join(WorkDir, 'RateAnalysis.csv')):
        #Read Previous RateAnalysis steps
        RatePath = os.path.join(WorkDir, 'RateAnalysis.csv')
        RateAnalysis = pd.read_csv(RatePath)
        i = len(RateAnalysis.index)
        
        #Obtain MD StepSize from INCAR in numbered folder from calculation just completed
        Temperature, StepSize, FolderSize = vio.INCARParser(f'{WorkDir}/{i + 1}')
        
        #Obtain total passed time and total carbon molecules removed
        TimePassed = RateAnalysis['Time (fs)'].iloc[i - 1]
        TotalTime = TimePassed + (StepSize * FolderSize)
    
    else:
        #Create RateAnalysis DataFrame
        RateAnalysis = pd.DataFrame(columns=['Time (fs)',
                                              'CO Count', 
                                              'CO2 Count',
                                              'O2 Added',
                                              'Total C Removed',
                                              'Total O2 Added',
                                              'O2 Count',
                                              'Smoothed O2 Count',
                                              'Oxidation Rate',
                                              'Lower 95% CI',
                                              'Upper 95% CI'])
        
        Temperature, StepSize, FolderSize = vio.INCARParser(f'{WorkDir}/1')
        
        #Set first timestep and carbon molecule removed
        TotalTime = StepSize * FolderSize
        
    return RateAnalysis, TotalTime, Temperature

    
    #Temperature, StepSize, FolderSize = vio.INCARParser(WorkDir = INCARPath)
    
    
def PlaceO2(OptimalCoords, Positions, CellDim):
    
    '''
    Returns a position dataframe with 2 new Os added at the loc. 
    Os will be alligned along x-axis.
    '''
    
    O2BondLength = 1.50
    
    #Convert coords to Cartesian
    CellDim = CellDim.to_numpy()
    OptimalCoords = OptimalCoords @ CellDim
    Ox1 = OptimalCoords + [O2BondLength/2, 0, 0] #Allign Os along x-axis
    Ox2 = OptimalCoords - [O2BondLength/2, 0, 0]

    #Revert coords back to Direct/Fractional
    InvCellDim = np.linalg.inv(CellDim)  
    Ox1 = Ox1 @ InvCellDim 
    Ox2 = Ox2 @ InvCellDim
    
    NewOx = pd.DataFrame({"Element": ["O", "O"],
                          "x": [Ox1[0], Ox2[0]],
                          "y": [Ox1[1], Ox2[1]],
                          "z": [Ox1[2], Ox2[2]]
                          })
    
    #Add in new Oxygen atoms (after final Ox)
    IndexLastO = len(Positions.loc[Positions['Element'] == 'O'].index)
    Positions = pd.concat([Positions.iloc[:IndexLastO], 
                           NewOx,
                           Positions.iloc[IndexLastO:]]).reset_index(drop = True)
    
    return Positions
    
    
def AddO2(Positions, Velocities, CellDim, Temperature):
    
    '''
    Function to add single O2 molecule to position and Velocity dataframes.
    
    Location of atom will be furthest away from any nearest neighbours, and velocities
    will be initialised via the MaxwellVoltzmann distribution.
    
    Args:
        Position (pd.DataFrame): Atom positions with 'Element' and cartesian 
            coordinates ('x', 'y', 'z').
            Args:
        Velocities (pd.DataFrame): Atom positions with 'Element' and cartesian 
            velocities in Å/fs ('vx', 'vy', 'vz').
        CellDim (pd.DataFrame): A 3x3 DataFrame defining cell dimensions in angstroms.
        Temperature (float): Temperature of the system in Kelvins
            
    Returns:
        Position (pd.DataFrame): Atom positions with 'Element' and cartesian 
            coordinates ('x', 'y', 'z'). Now with 2 added oxygen atoms.
        Velocities (pd.DataFrame): Atom positions with 'Element' and cartesian 
            velocities in Å/fs ('vx', 'vy', 'vz'). Now with 2 added oxygen atoms.
    '''
    
    OptimalCoords = an.FindNewCoord(Positions, CellDim)
    Positions = PlaceO2(OptimalCoords, Positions, CellDim)
 
    NewOxVel = an.MaxwellBoltzmannVelocities(['O', 'O'], Temperature)
    
    #Starting velocities will always be heading towards negative surface 
    NewOxVel = pd.DataFrame({'Element': ['O', 'O'],
                             'vx': [NewOxVel[0][0], NewOxVel[1][0]],
                             'vy': [NewOxVel[0][1], NewOxVel[1][1]],
                             'vz': [NewOxVel[0][2], NewOxVel[1][2]]
                             })
    
    #Add New Velocities
    IndexLastO = len(Velocities.loc[Velocities['Element'] == 'O'].index)
    Velocities = pd.concat([Velocities.iloc[:IndexLastO], 
                            NewOxVel,
                            Velocities.iloc[IndexLastO:]]).reset_index(drop = True)
    
    return Positions, Velocities


def CalculateVolumeRatio(WorkDir, CurrentGasVolume):
    #Calculate the ratio of starting volume (For which OTol was set) to current 
    #gas volume, to keep partialpressure constant.
    
    Position, _, CellDim = vio.ContcarParser(WorkDir = f'{WorkDir}/1', ReadPOSCAR = True)
    InitialGasVolume = an.CalculateGasVolume(Position, CellDim)
    
    return CurrentGasVolume / InitialGasVolume

    
def main(WorkDir = None, FreezePOSCAR = False):
    
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
        FreezePOSCAR (boolean): Option to run main without updating POSCAR, useful 
                                when fixing/updating rateanalysis.
    """
    
    #Hyperparameters
    if os.path.exists(os.path.join(WorkDir, '..', 'OxParams')):
        AtomicRadiusTol, O2Tol, OxygenSmoothing = ReadOxParams(WorkDir)
    
    else:
        #Presets if no Oxdiation Parameters file exists in one folder above workdir
        AtomicRadiusTol = 1.75 #Factor for bond finding (1.75 means 75% of bond length extra tolerance)
        O2Tol = 0.9 #Number of O2 atoms present before another gets added. Exponential smoothing factor of 0.001 steps so always set this below the actual target O2 -> think of it as a "lower bound" for the system. Gets adjusted downwards as available volume decreases (to maintain constant partial pressure).
        OxygenSmoothing = 0.001 #Oxygen Expoenential smoothign parameter. Lower = More smoothing, less responsive.
    
    #--------------------------- Gather Information ---------------------------

    if WorkDir == None:
        WorkDir = os.getcwd()
    
    #Check to see if MLFF is used for OUTCAR parsing
    MLFF = vio.CheckForMLFF(WorkDir)

    #Read POSCAR of last jobs (in working directory)
    Positions, AtomInfo, CellDim, Velocities = vio.ContcarParser(WorkDir, 
                                                                GiveVelocities = True,
                                                                ReadPOSCAR = True)
        
    #Rename Elements such that algos can work with them
    Positions, PrevNames = FixElementFormatting(Positions, ReturnPrevNames = True)
    AtomInfo = FixElementFormatting(AtomInfo, ReturnPrevNames = False)
    
    #Read RateAnalysis.csv
    RateAnalysis, TotalTime, Temperature = ReadRateAnalysis(WorkDir)
    
    #------------------------ Calculate Oxidation Rate ------------------------

    #If No RateAnalysis.csv exists yet only use current values
    if len(RateAnalysis.index) == 0:
              
        PrevCCount = 0
        TotalO2Added = 0
        #Maybe put this in a new function
        i = 0
        AllPositions, _ = vio.OUTCARParser(f'{WorkDir}/{1}', MLFF)
        #Go over all 80 steps 
        for Position in AllPositions:
            
            Position = FixElementFormatting(Position)
            Count, GasIndices, _ = an.FindGases(Position, 
                                                CellDim, 
                                                AtomicRadiusTol = AtomicRadiusTol)

            O2Count = int(Count.loc[Count['Molecule'] == ('O', 'O'), 'Count'].values[0])
            
            #Absolute first positions in sim
            if i == 0:
                SmoothedO2Count = O2Count #Value for next smoothing step
                i += 1
            #nth step of first 80 steps
            else:
                SmoothedO2Count = ExponentialSmoothing(O2Count, 
                                                       SmoothedO2Count, 
                                                       alpha = OxygenSmoothing)
            

    else:
        PrevCCount = RateAnalysis['Total C Removed'].iloc[-1]
        SmoothedO2Count = RateAnalysis['Smoothed O2 Count'].iloc[-1]
        TotalO2Added = RateAnalysis['Total O2 Added'].iloc[-1]
        
        i = len(RateAnalysis.index)
        AllPositions, _ = vio.OUTCARParser(f'{WorkDir}/{i+1}', MLFF)
        
        for Position in AllPositions:
            
            Position = FixElementFormatting(Position)
            Count, GasIndices, _ = an.FindGases(Position, 
                                                CellDim, 
                                                AtomicRadiusTol = AtomicRadiusTol)

            O2Count = int(Count.loc[Count['Molecule'] == ('O', 'O'), 'Count'].values[0])
            
            #read OUTCAR here and do actual smoothing on all 80 steps.
            SmoothedO2Count = ExponentialSmoothing(O2Count, 
                                                   SmoothedO2Count, 
                                                   alpha = OxygenSmoothing)

    #Obtain counts for final Position in file
    O2Count = int(Count.loc[Count['Molecule'] == ('O', 'O'), 'Count'].values[0])
    COCount = int(Count.loc[Count['Molecule'] == ('C', 'O'), 'Count'].values[0])
    CO2Count = int(Count.loc[Count['Molecule'] == ('C', 'O', 'O'), 'Count'].values[0])
    TotalCCount = CO2Count + COCount + PrevCCount

    GasVolume = an.CalculateGasVolume(Position, CellDim)
    
    #Adjust O2 Tolerance for decrease in Volume
    GasRatio = CalculateVolumeRatio(WorkDir, GasVolume)
    if GasRatio < 1:
        print(f'Adjusted OTol from {O2Tol} to {O2Tol * GasRatio}')
        O2Tol *= GasRatio
            
    #Calculate Partial Pressure based on target PP
    PartialPressure = an.CalculatePartialPressure(O2Tol,  
                                                  Temperature,
                                                  GasVolume)
    
    #Calculate Oxidation Rate and 95% confidence intervals
    OxRate, LowerCI, UpperCI = an.CalculateOxidationRate(N = TotalCCount,
                                                         t = TotalTime, 
                                                         CellDim = CellDim,
                                                         PartialPressure = PartialPressure)
    

    #---------------------- Prepare for next simulation -----------------------  
    
    #If carbon gasses have been found remove them from next simulation
    if (COCount != 0 or CO2Count != 0):
        
        #Remove Gasses
        OnlyCCIndices = GasIndices.loc[GasIndices['Molecule'] != ('O', 'O')]
        Positions = RemoveCO(Positions, OnlyCCIndices)
        Velocities = RemoveCO(Velocities, OnlyCCIndices)
            
        #Update AtomInfo
        AtomInfo.loc[AtomInfo["Element"] == 'O', "Number"] -= (COCount + 2 * CO2Count)
        AtomInfo.loc[AtomInfo["Element"] == 'C', "Number"] -= (COCount + CO2Count)
    
    #Add O2 if smoothed curve below 0.9, this can be customised parameters
    #Second condition is so that we don't add more O2 if Smoothed curve is already increasing
    if SmoothedO2Count <= O2Tol and O2Count < O2Tol:

        Positions, Velocities = AddO2(Positions, Velocities, CellDim, Temperature)
        AtomInfo.loc[AtomInfo["Element"] == 'O', "Number"] += 2
        
        O2Count += 1
        O2Added = 1
        TotalO2Added += 1        
        #WAVECAR no longer good starting point for calculation as atoms have been added
        if os.path.exists(f'{WorkDir}/WAVECAR'):
            os.remove(f'{WorkDir}/WAVECAR')
    else:
        O2Added = 0
    
    #Write to POSCAR for next job (and return old names)
    Positions = UnFixElementFormatting(Positions, PrevNames)
    AtomInfo = UnFixElementFormatting(AtomInfo, PrevNames)
    
    #Useful for FixRateAnalysis, file changes in this block
    if not FreezePOSCAR:
        vio.WritePOSCAR(WorkDir, Positions, CellDim, AtomInfo, Velocities)
    
        #If MLFF is running, prepare ABN for next run
        if os.path.exists(f'{WorkDir}/ML_ABN'):
            shutil.copy(f'{WorkDir}/ML_ABN', f'{WorkDir}/ML_AB')
        
    #Append new information and save RateAnalysis.csv
    RateAnalysis.loc[len(RateAnalysis.index)] = [round(TotalTime, 4),
                                                 COCount, 
                                                 CO2Count,
                                                 O2Added,
                                                 TotalCCount,
                                                 TotalO2Added,
                                                 O2Count,
                                                 SmoothedO2Count,
                                                 OxRate,
                                                 LowerCI,
                                                 UpperCI
                                                 ]
                                                 
    RateAnalysis.to_csv(f'{WorkDir}/RateAnalysis.csv', index = False)
    
    '''
    'Time (fs)',
    'CO Count', 
    'CO2 Count',
    'O2 Added',
    'Total C Removed',
    'Total O2 Added',
    'O2 Count',
    'Smoothed O2 Count',
    'Oxidation Rate',
    'Lower 95% CI',
    'Upper 95% CI'
    '''

def FixRateAnalysis(WorkDir):
    #Function that rewrites RateAnalysis if there have been issues in run
    #or rateanalysis.csv structure has been updated
    
    if os.path.exists(f'{WorkDir}/RateAnalysis.csv'):
        os.remove(f'{WorkDir}/RateAnalysis.csv')
    
    i = 1
    while True:
        if os.path.exists(f'{WorkDir}/{i}'):
            main(WorkDir, FreezePOSCAR = True)
            i += 1
        else:
            break
    
if __name__ == '__main__':
    
    WorkDir = os.getcwd() #use in prod 
    #WorkDir = 'Dir_VolSearch' #use for demos 
    
    main(WorkDir)
    
    #FixRateAnalysis(WorkDir = 'Dir_VolSearch_MLFF')
    
# %%

