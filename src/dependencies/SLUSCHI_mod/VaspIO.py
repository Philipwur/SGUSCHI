#%%
#Suite of functions for reading and creating VASP related files

import pandas as pd
import numpy as np
import os

import OxidationAnalysis as an



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

def AddElementsToPos(Position, AtomInfo):
    
    """
    Adds an 'Element' column to an atomic position or velocity DataFrame.

    This function assigns element labels to atomic positions based on the 
    provided AtomInfo DataFrame. It is a prerequisite for performing analysis 
    or further processing on atomic frames.

    Args:
        Position (pd.DataFrame): DataFrame containing atomic positions with 
            fractional coordinates 'x', 'y', and 'z'. It should not yet have 
            an 'Element' column.
        AtomInfo (pd.DataFrame): 2xN DataFrame containing element types and 
            their respective counts in the simulation cell. Order matters.

    Returns:
        pd.DataFrame: The input Position DataFrame with an added 'Element' column.
    """
    
    Elements = np.repeat(AtomInfo['Element'].values, AtomInfo['Number'].values)
    Position.insert(0, 'Element', Elements)

    return Position


def ContcarParser(WorkDir = None, GiveVelocities = False, ReadPOSCAR = False):
    
    '''
    Parses a CONTCAR or POSCAR file and extracts structural information.

    Args:
        WorkDir (str, optional): Absolute path to the directory containing the POSCAR.
            If None, the current working directory is used.
        GiveVelocities (bool, optional): Whether to return the Velocities DataFrame.
        ReadPOSCAR (bool, optional): If True, reads a POSCAR file instead of CONTCAR.

    Returns:
        list: A list containing the following elements:
            1. pd.DataFrame: Atom positions with 'Element' and fractional coordinates ('x', 'y', 'z').
            2. pd.DataFrame: AtomInfo, a 2xN DataFrame showing elements and their counts.
            3. pd.DataFrame: CellDim, a 3x3 DataFrame defining cell dimensions in angstroms.
            4. (Optional) pd.DataFrame: Velocities, with 'Element', 'vx', 'vy', and 'vz'.

    Notes:
        The order of returned elements depends on the boolean flags.
    '''
    
    #Helper function to find the next non-empty line in the POS/CONTCAR
    def NextNonEmpty(idx, Lines):
        while idx < len(Lines) and not Lines[idx].strip():
            idx += 1
        return idx
    
    #Assume current dir is workdir
    if WorkDir == None:
        WorkDir = os.getcwd()
        
    #Create file path ot CONT or POS
    FilePath = os.path.join(WorkDir, 'CONTCAR')
    if ReadPOSCAR == True:
        try:
            FilePath = os.path.join(WorkDir, 'POSCAR')
        except:
            FilePath = os.path.join(WorkDir, 'CONTCAR')

    #Read lines and remove trailing spaces and well as any inital blank spaces
    with open(FilePath, 'r') as f:
        Lines = [Line.rstrip() for Line in f] 
    Lines = Lines[NextNonEmpty(0, Lines):]

    #Read Scale Factor and CellDim
    idx = NextNonEmpty(1, Lines)
    ScaleFactor = float(Lines[idx].strip())
    idx = NextNonEmpty(idx + 1, Lines)
    CellDim = []
    for _ in range(3):
        Dimension = [float(x) * ScaleFactor for x in Lines[idx].split()]
        CellDim.append(Dimension)
        idx += 1
    CellDim = pd.DataFrame(CellDim, columns=['x', 'y', 'z'])

    #Read element and number lines, put into AtomInfo
    idx = NextNonEmpty(idx, Lines)
    ElemLine = Lines[idx].split()
    idx = NextNonEmpty(idx + 1, Lines)
    NumLine = Lines[idx].split()
    AtomInfo = pd.DataFrame([ElemLine, NumLine], index=['Element', 'Number']).T
    AtomInfo['Number'] = AtomInfo['Number'].astype(int)
    
    #Read line with coordinate system ("Direct" or "Cartesian"). If Cartesian flag for conversion later.
    idx = NextNonEmpty(idx + 1, Lines)
    if Lines[idx].lower()[0] == 'c':
        CartesianFlag = True
    else:
        CartesianFlag = False

    #Read atomic coordinates
    idx = NextNonEmpty(idx + 1, Lines)
    Positions = []
    while idx < len(Lines) and Lines[idx].strip():
        Positions.append([float(x) for x in Lines[idx].split()[:3]])
        idx += 1
    Positions = pd.DataFrame(Positions, columns = ['x', 'y', 'z'])
    Positions = AddElementsToPos(Positions, AtomInfo)

    #Convert Cartesian to Direct Coordinates
    if CartesianFlag:
        Positions = an.ConvertCartesianToDirect(Positions, CellDim)
    
    returns = [Positions, AtomInfo, CellDim]
    
    #If requested, attempt to read velocities
    if GiveVelocities:
        
        idx = NextNonEmpty(idx, Lines)
        Velocities = []
        
        #Keep reading velocity lines until empty line is encountered or end of file is reached.
        while idx < len(Lines) and Lines[idx].strip():
            Velocities.append([float(x) for x in Lines[idx].split()[:3]])
            idx = idx + 1

        #If Positions match velocities, great. Else something has gone wrong. Omit velocities.
        if len(Velocities) == len(Positions):
            Velocities = pd.DataFrame(Velocities, columns = ['vx', 'vy', 'vz'])
            Velocities = AddElementsToPos(Velocities, AtomInfo)
        else:
            print('Number of Velocity lines do not match Position lines')
            Velocities = None

        returns.append(Velocities)

    return returns


def INCARParser(WorkDir = None, Parameters = ['TEBEG', 'POTIM', 'NSW'], FilePath = None):
    """
    Parses the INCAR file and extracts specified parameters in the given order.

    This function reads an INCAR file in the given working directory and retrieves 
    the values of specified VASP input parameters. Can also be used on the OxParams
    file to pass on any hyperparameters.

    Args:
        WorkDir (str, optional): Path to the directory containing the INCAR file. 
            Defaults to the current working directory.
        Parameters (list, optional): List of parameter names to extract. 
            Defaults to ['TEBEG', 'POTIM', 'NSW'].

    Returns:
        list: A list of values corresponding to the requested parameters, 
              in the same order as `Parameters`. If a parameter is missing, `None` is returned.
    """

    # Use current directory if WorkDir is not specified
    if WorkDir is None:
        WorkDir = os.getcwd()
    
    #if FilePath:
        
    if not FilePath:
        FilePath = os.path.join(WorkDir, 'INCAR')

    # Initialize dictionary to store parameters with None as default
    INCARValues = {param: None for param in Parameters}

    with open(FilePath, 'r') as f:
        for line in f:
            # Strip whitespace and split at '='
            parts = line.strip().split('=')

            if len(parts) == 2:
                Key = parts[0].strip()
                Value = parts[1].strip()

                # Store only requested parameters
                if Key in INCARValues:
                    # Attempt to convert value to float or int if possible
                    try:
                        if '.' in Value:
                            INCARValues[Key] = float(Value)
                        else:
                            INCARValues[Key] = int(Value)
                    except ValueError:
                        INCARValues[Key] = Value  # Store as string if conversion fails

    #Return values in the same order as requested
    return [INCARValues[param] for param in Parameters]
 
 
def CheckForMLFF(WorkDir):
    #Small function to check if MLFF was used in calcuation. 
    #Can be passed on to OUTCAR parser.
    
    try:
        MLFF = INCARParser(WorkDir, ['ML_LMLFF'])
        if MLFF[0] == '.TRUE.':
            MLFF = True
        else:
            MLFF = False
    except:
        MLFF = False
    
    return MLFF
 

def OUTCARParser(WorkDir = None, MLFF = False):
    
    """
    Parses the OUTCAR file from a VASP simulation to extract atomic positions and energies.

    This function reads atomic positions and total energies from the OUTCAR file while 
    ensuring memory efficiency. It assumes constant volume (V) and requires the 
    corresponding INCAR and CONTCAR files for necessary metadata.

    Args:
        WorkDir (str, optional): Path to the directory containing the OUTCAR file.
            Defaults to the current working directory.
        MLFF (boolean): Boolean for whether the simulation was done using Vasp's
            built in MLFF potential.

    Returns:
        tuple:
            - AllPositions (pd.DataFrame): A list of atomic position DataFrames, one per simulation step.
              Each DataFrame contains columns ['Element', 'x', 'y', 'z'] in **direct coordinates**.
            - AllEnergies (pd.DataFrame): A DataFrame containing time evolution of total energies with columns:
                - **'Time (fs)'**: Simulation time in femtoseconds.
                - **'Energy (eV)'**: Total energy of the system at each step.

    Notes:
        - Requires INCAR and POSCAR to be in same directory as OUTCAR
    """
    
    #Set WorkDir
    if WorkDir == None:
        WorkDir = os.getcwd()
    
    #First get relevant info from smaller files to simplify parser
    _, StepSize, TotalSteps = INCARParser(WorkDir)
    _, AtomInfo, CellDim = ContcarParser(WorkDir, ReadPOSCAR = True)
    
    NumAtoms = AtomInfo['Number'].sum()
    
    #Current objectives get energies and positions
    AllEnergies = []
    AllPositions = []
    
    #Lines for parser in case MLFF is used
    if MLFF == False:
        PositionTag = 'POSITION'
        EnergyTag = 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)'
    
    elif MLFF == True:
        PositionTag = 'POSITION                                       TOTAL-FORCE (eV/Angst) (ML)'
        EnergyTag = 'ML FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)'
        
    with open(os.path.join(WorkDir, 'OUTCAR')) as f:
        
        #Iterating over lines
        for line in f:
            
            #Collecting Positions    
            if PositionTag in line:
                
                next(f) #Skip -------
                
                Position = [[float(x) #Converts to flaot
                              for x in next(f).strip().split()[0:3]] #Takes first 3 position coords 
                             for _ in range(NumAtoms)] #Iterates over all atoms
                
                #Ensures Position Format is the same as other code (direct with elements)
                Position = pd.DataFrame(Position, columns = ['x', 'y', 'z'])
                Position = AddElementsToPos(Position, AtomInfo)
                Position = an.ConvertCartesianToDirect(Position, CellDim)
                AllPositions.append(Position)  # Store for later analysis
                
            #Collecting Energies
            if EnergyTag in line:
                next(f) #Skip -------
                next_line = next(f).strip().split()  # Read and split the next line

                # Extract the energy value (second last item in the split list)
                Energy = float(next_line[-2])  # Get the numeric value before "eV"
                AllEnergies.append(Energy)
    
    #Formatting for AllEnergy values
    Times = [(i + 1) * StepSize for i in range(TotalSteps)]
    AllEnergies = pd.DataFrame({'Time (fs)': Times, 'Energy (eV)': AllEnergies})
    
    return AllPositions, AllEnergies
    

def VolSearchParser(WorkDir = None):
    
    '''
    Parses the entire OUTCAR trajctory of an Oxidation SLUSCHI run located within
    a Dir_VolSearch folder. 
    '''
    
    #Set WorkDir
    if WorkDir == None:
        WorkDir = os.getcwd()
    
    #Initialise arrays
    Folder = 0
    AllPositions = []
    AllEnergies = []
    TimeOffset = 0 #Variable for combining AllEnergy
    
    MLFF = CheckForMLFF(WorkDir)
    #print(MLFF)
    while True:

        #prepare folder directory
        Folder += 1
        FolderDir = os.path.join(WorkDir, f'{Folder}')
        if not os.path.isdir(FolderDir):
            break

        #Gather Position and Energies
        Positions, Energies = OUTCARParser(WorkDir = FolderDir, MLFF = MLFF)
        AllPositions.extend(Positions)
        AllEnergies.append(Energies)
    
    
    #Combine all Energies into a single DataFrame
    FlattenedEnergies = []
    TimeOffset = 0
    for Energies in AllEnergies:
        
        #Offset time by previous final time
        Energies['Time (fs)'] += TimeOffset
        FlattenedEnergies.append(Energies)
        TimeOffset = Energies['Time (fs)'].iloc[-1]
        
    AllEnergies = pd.concat(FlattenedEnergies, ignore_index=True)
    
    return AllPositions, AllEnergies
    
    
def WritePOSCAR(WorkDir, Position, CellDim, AtomInfo, Velocities = None):

    """
    Creates a POSCAR file for VASP calculations in the specified directory.

    This function generates a POSCAR file using the provided atomic positions, 
    cell dimensions, and atomic composition. If velocities are provided, they 
    will also be included in the output file.

    Args:
        WorkDir (str): Path to the directory where the new POSCAR file must be saved.
        Position (pd.DataFrame): DataFrame containing atom positions with columns: 
            'Element', 'x', 'y', and 'z' (fractional coordinates).
        CellDim (pd.DataFrame): 3x3 DataFrame defining cell dimensions in angstroms.
        AtomInfo (pd.DataFrame): 2xN DataFrame listing element types and their counts 
            in the simulation cell. Order matters.
        Velocities (pd.DataFrame, optional): DataFrame containing atomic velocities 
            with columns: 'Element', 'vx', 'vy', and 'vz'.

    Returns:
        None: The function writes the POSCAR file directly to disk.

    Raises:
        FileNotFoundError: If the specified WorkDir does not exist and cannot be created.
        ValueError: If the input DataFrames do not conform to the expected structure.
    """
    
    #Check for Directory
    if not os.path.exists(WorkDir):
        os.makedirs(WorkDir)
    
    #Boilerplate
    FileName = os.path.join(WorkDir, "POSCAR")
    Title = 'Structure (CO(2) Removed) by SLUSCHI'
    ScaleFactor = 1.0
    
    #Reorder Position to match the element order given in AtomInfo 
    #(which was set in first POSCAR read)
    OrderedPositions = []
    for _, row in AtomInfo.iterrows():
        Element = row['Element']
        Count = row['Number']
        Subset = Position[Position['Element'] == Element].iloc[:Count]
        OrderedPositions.append(Subset)
    OrderedPositions = pd.concat(OrderedPositions, ignore_index=True)

    #Prepare formatting for Element and Number of element line.
    MaxLenElement = max(len(e) for e in AtomInfo['Element'])
    MaxLenNumber = max(len(str(n)) for n in AtomInfo['Number'])
    ElementLine = "    ".join(e.ljust(MaxLenElement) for e in AtomInfo['Element'])
    NumberLine =  "    ".join(str(n).rjust(MaxLenNumber) for n in AtomInfo['Number'])

    #Start Writing the POSCAR file
    with open(FileName, 'w') as f:
        
        #Boilerplate
        f.write(f"{Title}\n") #Title
        f.write(f"{ScaleFactor}\n") #Scale
        for i in range(3): #Cell Dimensions
            f.write(f"{CellDim.iloc[i]['x']:22.16f} {CellDim.iloc[i]['y']:22.16f} {CellDim.iloc[i]['z']:22.16f}\n")
        f.write(f"    {ElementLine}\n") #Element
        f.write(f"    {NumberLine}\n") #Number of Atoms per element
        f.write("Direct\n")
        
        #Write Atomic positions
        for _, Atom in OrderedPositions.iterrows():
            f.write(f"{Atom['x']:22.16f} {Atom['y']:22.16f} {Atom['z']:22.16f}\n")

        if Velocities is not None:
            
            #Reorder Velocities to match the element order given in AtomInfo
            OrderedVelocities = []
            for _, row in AtomInfo.iterrows():
                Element = row['Element']
                Count = row['Number']
                Subset = Velocities[Velocities['Element'] == Element].iloc[:Count]
                OrderedVelocities.append(Subset)
            OrderedVelocities = pd.concat(OrderedVelocities, ignore_index=True)
            
            #Write Velocities
            f.write("\n")
            for _, v in OrderedVelocities.iterrows():
                f.write(f"{v['vx']:22.16f} {v['vy']:22.16f} {v['vz']:22.16f}\n")


# %%


#if __name__ == '__main__':
    
    
    
    #WorkDir = 'Dir_VolSearch_MLFF/1'
    #WorkDir = os.getcwd()
    #Position, _, CellDim = ContcarParser(WorkDir, False, True)
    #Position = FixElementFormatting(Position)
    #-------
    
    '''
    WIP ScaleLength Determiner
    
    BondMatrix = an.CalculateScaleThickness(Position, CellDim, 1.75)
    
    #Filter to only Zr and O elements
    OIndices = Position[Position['Element'] == 'O'].index
    ZrIndices = Position[Position['Element'] == 'Zr'].index
    
    #Make mask for all Os bonded to Zrs
    ScaleOIndex = []
    for OIndex in OIndices:
        for ZrIndex in ZrIndices:
            if OIndex not in ScaleOIndex:
                if BondMatrix[OIndex, ZrIndex] == 1:
                    ScaleOIndex.append(OIndex)
                
    #Apply mask
    ScaleOs = Position.loc[ScaleOIndex].sort_values(by = 'x')
    
    def FindScaleWidths(ScaleOs, Threshold = 0.05):
    #how do we ensure consistency in the output of scale widths
    #- Option 1: one scale is always close to x=1/0 and the other x=0.5-0.4
        #Option 2: wait until 20 have absorbed and then do KMeans
        #Too few oxygens to measure scale properly
        if len(ScaleOs) <= 3:
            return [0, 0] 
        
        
        return 
    '''
    
# %%

