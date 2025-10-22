#%%
#Suite of functions for reading and creating VASP related files

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import sys
from pathlib import Path
from typing import Optional, Tuple
sys.path.append(str(Path(__file__).resolve().parents[1]))
from workflow import OxidationAnalysis as an





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



def XYZTrajectoryParser(FilePath=None,
                        WorkDir=None,
                        AssumeStaticCell=True,
                        CellChangeTolerance=1e-10,
                        ShowProgress=False,
                        ReadFirstAndLastOnly=False):
    """
    Efficiently parse an extended XYZ trajectory (VASP-style).

    Modes:
    - Full trajectory parsing (with optional tqdm progress bar)
    - Fast first/last frame extraction without scanning full file

    Returns:
        Positions : list[pd.DataFrame]
        Energies  : pd.DataFrame
        CellDim   : pd.DataFrame (first frame)
    """
    # --------------------------------------------------------------------------
    # === Setup ===
    if FilePath is None:
        if WorkDir is None:
            WorkDir = os.getcwd()
        FilePath = os.path.join(WorkDir, "trajectory.xyz")

    # --- Regex patterns ---
    LatticeRegex = re.compile(r'Lattice="([^"]+)"')
    TimeRegex = re.compile(r'Time_fs=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
    EnergyRegex = re.compile(r'Energy_eV=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

    # --------------------------------------------------------------------------
    # === Helper functions ===
    def BuildCell(LatticeVals):
        """Return both ndarray and DataFrame forms of the lattice."""
        Mat = np.array([LatticeVals[0:3], LatticeVals[3:6], LatticeVals[6:9]], dtype=float)
        return Mat, pd.DataFrame(Mat, columns=["x", "y", "z"])

    def CartesianToDirect(Coords, Mat):
        """Convert Cartesian coordinates to fractional using the lattice matrix."""
        return Coords @ np.linalg.inv(Mat.T)

    def ParseXYZFrame(Lines):
        """Parse one XYZ frame (header + atoms)."""
        NumAtoms = int(Lines[0].strip())
        Header = Lines[1].strip()

        # Metadata
        Lattice = [float(x) for x in LatticeRegex.search(Header).group(1).split()]
        Mat, CellDf = BuildCell(Lattice)
        Time = float(TimeRegex.search(Header).group(1)) if TimeRegex.search(Header) else np.nan
        Energy = float(EnergyRegex.search(Header).group(1)) if EnergyRegex.search(Header) else np.nan

        # Atomic positions
        AtomLines = Lines[2:2 + NumAtoms]
        Elements, Coords = zip(*[(ln.split()[0], list(map(float, ln.split()[1:4]))) for ln in AtomLines])
        Coords = np.array(Coords, dtype=float)
        Frac = CartesianToDirect(Coords, Mat)

        Pos = pd.DataFrame({"Element": pd.Categorical(Elements),
                            "x": Frac[:, 0], "y": Frac[:, 1], "z": Frac[:, 2]})
        return Pos, [Time, Energy], CellDf, Mat

    def CountFrames(Path):
        """Quickly count number of frames for tqdm."""
        Count = 0
        with open(Path) as f:
            for line in f:
                if line.strip().isdigit():
                    Count += 1
        return Count

    # --------------------------------------------------------------------------
    # === Fast path: first and last frame only ===
    if ReadFirstAndLastOnly:
        def ReadFirstFrame(Path):
            with open(Path) as f:
                N = int(f.readline().strip())
                Lines = [str(N), f.readline()] + [f.readline() for _ in range(N)]
            return ParseXYZFrame(Lines)

        def ReadLastFrame(Path):
            CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
            Lines = []
            with open(Path, "rb") as f:
                f.seek(0, os.SEEK_END)
                Size = f.tell()
                Offset = 0
                while Offset < Size:
                    Offset = min(Size, Offset + CHUNK_SIZE)
                    f.seek(Size - Offset)
                    Chunk = f.read(CHUNK_SIZE)
                    LinesChunk = Chunk.splitlines()
                    Lines = LinesChunk + Lines
                    for i, ln in enumerate(Lines):
                        txt = ln.decode(errors="ignore").strip()
                        if txt.isdigit():
                            N = int(txt)
                            if len(Lines) - i >= N + 2:
                                Block = [L.decode(errors="ignore") for L in Lines[i:i + N + 2]]
                                return ParseXYZFrame(Block)
            raise ValueError("Last frame not found.")

        FirstFrame, FirstE, CellDim, _ = ReadFirstFrame(FilePath)
        LastFrame, LastE, _, _ = ReadLastFrame(FilePath)

        Energies = pd.DataFrame([FirstE, LastE], columns=["Time (fs)", "Energy (eV)"])
        return [FirstFrame, LastFrame], Energies, CellDim

    # --------------------------------------------------------------------------
    # === Full parsing path ===
    TotalFrames = CountFrames(FilePath) if ShowProgress else None
    Positions, EnergiesList = [], []
    CellDim, FirstCell = None, None

    with open(FilePath) as f, tqdm(total=TotalFrames, disable=not ShowProgress, desc="Parsing XYZ") as bar:
        FrameIndex = 0
        while True:
            Line = f.readline()
            if not Line:
                break
            if not Line.strip().isdigit():
                continue

            N = int(Line.strip())
            Header = f.readline()
            FrameLines = [Line, Header] + [f.readline() for _ in range(N)]
            if len(FrameLines) < N + 2:
                break

            Pos, Energy, CellDf, Mat = ParseXYZFrame(FrameLines)

            # Manage static cell logic
            if FrameIndex == 0:
                CellDim, FirstCell = CellDf, Mat
            else:
                if not AssumeStaticCell and np.linalg.norm(Mat - FirstCell) > CellChangeTolerance:
                    FirstCell = Mat 

            Positions.append(Pos)
            EnergiesList.append(Energy)
            FrameIndex += 1
            if ShowProgress:
                bar.update(1)

    Energies = pd.DataFrame(EnergiesList, columns=["Time (fs)", "Energy (eV)"])
    return Positions, Energies, CellDim





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
    Times = [(i + 1) * StepSize for i in range(len(AllEnergies))]
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
   
   
   
#%% New Gen functions here
 
def ReadPOSCAR(
    workdir: Optional[str] = None,
    filename: Optional[str] = None,
    give_velocities: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read a VASP POSCAR/CONTCAR-like file and return:
      - Position: DataFrame with columns ['Element', 'x', 'y', 'z'] (fractional)
      - CellDim:  DataFrame with columns ['x', 'y', 'z'] (cartesian lattice vectors)
    If give_velocities=True, also returns Velocities: DataFrame with ['Element','vx','vy','vz'].

    Args:
        workdir: Directory to search in (defaults to current working directory).
        filename: Explicit filename to read. If None, uses 'POSCAR' in workdir,
                  and if that does not exist, falls back to 'CONTCAR'.
        give_velocities: Try to parse N velocity lines (after positions). Default False.

    Returns:
        (Position, CellDim) or (Position, CellDim, Velocities)
    """
    # -------- helpers --------
    def next_nonempty(idx: int, lines: list[str]) -> int:
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        return idx

    def parse_three_floats(line: str) -> list[float]:
        vals = []
        for tok in line.split():
            try:
                vals.append(float(tok))
                if len(vals) == 3:
                    break
            except ValueError:
                # ignore flags like T/F or comments
                continue
        if len(vals) != 3:
            raise ValueError(f"Could not parse 3 numeric coords from line: {line!r}")
        return vals

    def expand_elements(elem_line: list[str], num_line: list[str]) -> list[str]:
        out = []
        for el, n in zip(elem_line, num_line):
            out.extend([el] * int(n))
        return out

    def cart_to_frac(cart: np.ndarray, cell: np.ndarray) -> np.ndarray:
        # cell is 3x3 with rows = lattice vectors as given in POSCAR
        # r_frac satisfies r_cart = r_frac @ cell  =>  r_frac = r_cart @ inv(cell)
        inv_cell = np.linalg.inv(cell)
        return cart @ inv_cell

    # -------- locate file --------
    base = Path(workdir) if workdir is not None else Path.cwd()
    if filename is None:
        path = base / "POSCAR"
        if not path.exists():
            # gentle fallback for convenience
            alt = base / "CONTCAR"
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(f"No POSCAR or CONTCAR found in {base}")
    else:
        p = Path(filename)
        path = p if p.is_absolute() else (base / p)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # -------- read & trim --------
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]
    start = next_nonempty(0, lines)
    lines = lines[start:]

    # Title (ignore), scale, cell
    # title:
    idx = next_nonempty(1, lines)  # scale is usually line 1
    scale = float(lines[idx].split()[0])
    idx = next_nonempty(idx + 1, lines)

    cell_rows = []
    for _ in range(3):
        cell_rows.append([float(x) * scale for x in lines[idx].split()[:3]])
        idx += 1
    cell = np.array(cell_rows, dtype=float)  # 3x3
    CellDim = pd.DataFrame(cell_rows, columns=["x", "y", "z"])

    # Element symbols + counts
    idx = next_nonempty(idx, lines)
    elem_line = lines[idx].split()
    idx = next_nonempty(idx + 1, lines)
    num_line = lines[idx].split()
    counts = [int(x) for x in num_line]
    n_atoms = sum(counts)
    elements_expanded = expand_elements(elem_line, num_line)

    # Coordinate header (Selective Dynamics optional)
    idx = next_nonempty(idx + 1, lines)
    header = lines[idx].strip().lower()
    if header.startswith("s"):  # "Selective dynamics"
        idx = next_nonempty(idx + 1, lines)
        header = lines[idx].strip().lower()

    if header.startswith("c"):  # Cartesian
        is_cart = True
    else:                        # "Direct" assumed if not Cartesian
        is_cart = False

    # Positions: read exactly n_atoms lines (more robust than 'until blank')
    idx = next_nonempty(idx + 1, lines)
    coords = []
    for _ in range(n_atoms):
        if idx >= len(lines):
            raise ValueError("Unexpected end of file while reading positions.")
        coords.append(parse_three_floats(lines[idx]))
        idx += 1
    coords = np.array(coords, dtype=float)

    # Convert to fractional if needed
    if is_cart:
        frac = cart_to_frac(coords, cell)
    else:
        frac = coords

    Position = pd.DataFrame(frac, columns=["x", "y", "z"])
    Position.insert(0, "Element", elements_expanded)

    if not give_velocities:
        return Position, CellDim

    # ---- optional velocities (try to parse next n_atoms lines of 3 floats) ----
    idx = next_nonempty(idx, lines)
    vels = []
    for _ in range(n_atoms):
        if idx >= len(lines) or not lines[idx].strip():
            break
        try:
            vels.append(parse_three_floats(lines[idx]))
        except ValueError:
            break
        idx += 1

    Velocities = None
    if len(vels) == n_atoms:
        Velocities = pd.DataFrame(vels, columns=["vx", "vy", "vz"])
        Velocities.insert(0, "Element", elements_expanded)

    return (Position, CellDim, Velocities) if Velocities is not None else (Position, CellDim)


def WritePOSCAR(
    WorkDir: str,
    Position: pd.DataFrame,
    CellDim: pd.DataFrame,
    Velocities: pd.DataFrame = None,
    FileName: str = None,
    Title: str = "Structure generated by SLUSCHI and SGUSCHI",
):
    """
    Writes a VASP-format POSCAR (or CONTCAR-like) file using Position and CellDim.

    Args:
        WorkDir (str): Directory where the POSCAR should be written.
        Position (pd.DataFrame): DataFrame with columns ['Element', 'x', 'y', 'z']
            in fractional coordinates.
        CellDim (pd.DataFrame): 3x3 DataFrame defining lattice vectors (in Å).
        Velocities (pd.DataFrame, optional): DataFrame with columns
            ['Element', 'vx', 'vy', 'vz'].
        FileName (str, optional): Output file name. Defaults to "POSCAR".
        Title (str, optional): Title line for the POSCAR header.

    Returns:
        None
    """
    # ---- Setup and validation ----
    if not os.path.exists(WorkDir):
        os.makedirs(WorkDir)

    if FileName is None:
        FileName = "POSCAR"
    filepath = os.path.join(WorkDir, FileName)

    required_cols = {"Element", "x", "y", "z"}
    if not required_cols.issubset(Position.columns):
        raise ValueError(f"Position DataFrame must contain columns: {required_cols}")

    if not {"x", "y", "z"}.issubset(CellDim.columns) or len(CellDim) != 3:
        raise ValueError("CellDim must be a 3x3 DataFrame with columns ['x','y','z'].")

    # ---- Infer composition ----
    atom_counts = Position["Element"].value_counts(sort=False)
    element_order = list(atom_counts.index)

    # ---- Write POSCAR ----
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{Title}\n")
        f.write("1.0\n")  # scale factor

        # lattice vectors
        for i in range(3):
            f.write(
                f"{CellDim.iloc[i]['x']:22.16f} "
                f"{CellDim.iloc[i]['y']:22.16f} "
                f"{CellDim.iloc[i]['z']:22.16f}\n"
            )

        # element and counts lines
        f.write("   " + "   ".join(element_order) + "\n")
        f.write("   " + "   ".join(str(atom_counts[e]) for e in element_order) + "\n")
        f.write("Direct\n")

        # positions, grouped by element for readability
        for elem in element_order:
            subset = Position[Position["Element"] == elem]
            for _, atom in subset.iterrows():
                f.write(
                    f"{atom['x']:22.16f} "
                    f"{atom['y']:22.16f} "
                    f"{atom['z']:22.16f}\n"
                )

        # optional velocities section
        if Velocities is not None and not Velocities.empty:
            f.write("\n")
            for elem in element_order:
                subset = Velocities[Velocities["Element"] == elem]
                for _, v in subset.iterrows():
                    f.write(
                        f"{v['vx']:22.16f} "
                        f"{v['vy']:22.16f} "
                        f"{v['vz']:22.16f}\n"
                    )

    return

# %%

