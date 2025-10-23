#%%
#Suite of functions for reading and creating VASP related files

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))
#from workflow import OxidationAnalysis as an


def FixElementFormatting(Position, ReturnPrevNames = False):
    '''
    A function which fixes the element names in a position/velocity DataFrame so
    elements can be comprehended by bond finding algorithms later. 
    '''
    if ReturnPrevNames:
        PrevNames = Position['Element'].unique()

    for i in Position['Element'].unique():
        if '_' in i:
            FixedName = i.split('_')[0]
            Position.loc[Position['Element'] == i, 'Element'] = FixedName
        elif '/' in i:
            FixedName = i.split('/')[0]
            Position.loc[Position['Element'] == i, 'Element'] = FixedName

    if ReturnPrevNames:
        return Position, PrevNames
    else:
        return Position




def XYZTrajectoryParser(FilePath=None,
                        WorkDir=None,
                        AssumeStaticCell=True,
                        CellChangeTolerance=1e-10,
                        ShowProgress=False,
                        ReadFirstAndLastOnly=False):
    """
    Efficiently parse an extended XYZ trajectory (VASP-style).
    """
    if FilePath is None:
        if WorkDir is None:
            WorkDir = os.getcwd()
        FilePath = os.path.join(WorkDir, "trajectory.xyz")

    LatticeRegex = re.compile(r'Lattice="([^"]+)"')
    TimeRegex = re.compile(r'Time_fs=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
    EnergyRegex = re.compile(r'Energy_eV=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

    def BuildCell(LatticeVals):
        Mat = np.array([LatticeVals[0:3], LatticeVals[3:6], LatticeVals[6:9]], dtype=float)
        return Mat, pd.DataFrame(Mat, columns=["x", "y", "z"])

    def CartesianToDirect(Coords, Mat):
        return Coords @ np.linalg.inv(Mat.T)

    def ParseXYZFrame(Lines):
        NumAtoms = int(Lines[0].strip())
        Header = Lines[1].strip()
        Lattice = [float(x) for x in LatticeRegex.search(Header).group(1).split()]
        Mat, CellDf = BuildCell(Lattice)
        Time = float(TimeRegex.search(Header).group(1)) if TimeRegex.search(Header) else np.nan
        Energy = float(EnergyRegex.search(Header).group(1)) if EnergyRegex.search(Header) else np.nan
        AtomLines = Lines[2:2 + NumAtoms]
        Elements, Coords = zip(*[(ln.split()[0], list(map(float, ln.split()[1:4]))) for ln in AtomLines])
        Coords = np.array(Coords, dtype=float)
        Frac = CartesianToDirect(Coords, Mat)
        Pos = pd.DataFrame({"Element": pd.Categorical(Elements),
                            "x": Frac[:, 0], "y": Frac[:, 1], "z": Frac[:, 2]})
        return Pos, [Time, Energy], CellDf, Mat

    def CountFrames(Path):
        Count = 0
        with open(Path) as f:
            for line in f:
                if line.strip().isdigit():
                    Count += 1
        return Count

    if ReadFirstAndLastOnly:
        def ReadFirstFrame(Path):
            with open(Path) as f:
                N = int(f.readline().strip())
                Lines = [str(N), f.readline()] + [f.readline() for _ in range(N)]
            return ParseXYZFrame(Lines)

        def ReadLastFrame(Path):
            CHUNK_SIZE = 2 * 1024 * 1024
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


def INCARParser(WorkDir=None, Parameters=None, FilePath=None):
    if Parameters is None:
        Parameters = ['TEBEG', 'POTIM', 'NSW']
    if WorkDir is None:
        WorkDir = os.getcwd()
    if not FilePath:
        FilePath = os.path.join(WorkDir, 'INCAR')
    INCARValues = {param: None for param in Parameters}
    with open(FilePath, 'r') as f:
        for line in f:
            parts = line.strip().split('=')
            if len(parts) == 2:
                Key = parts[0].strip()
                Value = parts[1].strip()
                if Key in INCARValues:
                    try:
                        if '.' in Value:
                            INCARValues[Key] = float(Value)
                        else:
                            INCARValues[Key] = int(Value)
                    except ValueError:
                        INCARValues[Key] = Value
    return [INCARValues[param] for param in Parameters]



'''
#NOT WORKING
def OUTCARParser(WorkDir=None, MLFF=False):
    if WorkDir is None:
        WorkDir = os.getcwd()
    _, StepSize, TotalSteps = INCARParser(WorkDir)
    _, AtomInfo, CellDim = ContcarParser(WorkDir, ReadPOSCAR=True)
    NumAtoms = AtomInfo['Number'].sum()
    AllEnergies = []
    AllPositions = []
    if not MLFF:
        PositionTag = 'POSITION'
        EnergyTag = 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)'
    else:
        PositionTag = 'POSITION                                       TOTAL-FORCE (eV/Angst) (ML)'
        EnergyTag = 'ML FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)'

    with open(os.path.join(WorkDir, 'OUTCAR')) as f:
        for line in f:
            if PositionTag in line:
                next(f)
                Position = [[float(x) for x in next(f).strip().split()[0:3]]
                            for _ in range(NumAtoms)]
                Position = pd.DataFrame(Position, columns=['x', 'y', 'z'])
                Position = AddElementsToPos(Position, AtomInfo)
                Position = an.ConvertCartesianToDirect(Position, CellDim)
                AllPositions.append(Position)
            if EnergyTag in line:
                next(f)
                next_line = next(f).strip().split()
                Energy = float(next_line[-2])
                AllEnergies.append(Energy)

    Times = [(i + 1) * StepSize for i in range(len(AllEnergies))]
    AllEnergies = pd.DataFrame({'Time (fs)': Times, 'Energy (eV)': AllEnergies})
    return AllPositions, AllEnergies

#NOT WORKIGN BUT MAYBE DEFUNCT
def VolSearchParser(WorkDir=None):
    if WorkDir is None:
        WorkDir = os.getcwd()
    Folder = 0
    AllPositions = []
    AllEnergies = []
    TimeOffset = 0
    MLFF = CheckForMLFF(WorkDir)
    while True:
        Folder += 1
        FolderDir = os.path.join(WorkDir, f'{Folder}')
        if not os.path.isdir(FolderDir):
            break
        Positions, Energies = OUTCARParser(WorkDir=FolderDir, MLFF=MLFF)
        AllPositions.extend(Positions)
        AllEnergies.append(Energies)
    FlattenedEnergies = []
    TimeOffset = 0
    for Energies in AllEnergies:
        Energies['Time (fs)'] += TimeOffset
        FlattenedEnergies.append(Energies)
        TimeOffset = Energies['Time (fs)'].iloc[-1]
    AllEnergies = pd.concat(FlattenedEnergies, ignore_index=True)
    return AllPositions, AllEnergies

# In theory we dont need this anymore
def AddElementsToPos(Position, AtomInfo):
    """
    Adds an 'Element' column to an atomic position or velocity DataFrame.
    """
    Elements = np.repeat(AtomInfo['Element'].values, AtomInfo['Number'].values)
    Position.insert(0, 'Element', Elements)
    return Position
'''

#%% New Gen functions here

def CheckForMLFF(WorkDir):
    try:
        MLFF = INCARParser(WorkDir, ['ML_LMLFF'])
        if MLFF[0] == '.TRUE.':
            MLFF = True
        else:
            MLFF = False
    except Exception:
        MLFF = False
    return MLFF

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

def ReadPOSCAR(
    workdir=None,
    filename=None,
    give_velocities=False
):
    """
    Read a VASP POSCAR/CONTCAR-like file.
    """
    def next_nonempty(idx, lines):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        return idx

    def parse_three_floats(line):
        vals = []
        for tok in line.split():
            try:
                vals.append(float(tok))
                if len(vals) == 3:
                    break
            except ValueError:
                continue
        if len(vals) != 3:
            raise ValueError("Could not parse 3 numeric coords from line: %r" % line)
        return vals

    def expand_elements(elem_line, num_line):
        out = []
        for el, n in zip(elem_line, num_line):
            out.extend([el] * int(n))
        return out

    def cart_to_frac(cart, cell):
        inv_cell = np.linalg.inv(cell)
        return cart @ inv_cell

    base = Path(workdir) if workdir is not None else Path.cwd()
    if filename is None:
        path = base / "POSCAR"
        if not path.exists():
            alt = base / "CONTCAR"
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError("No POSCAR or CONTCAR found in %s" % base)
    else:
        p = Path(filename)
        path = p if p.is_absolute() else (base / p)
    if not path.exists():
        raise FileNotFoundError("File not found: %s" % path)

    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]
    start = next_nonempty(0, lines)
    lines = lines[start:]

    idx = next_nonempty(1, lines)
    scale = float(lines[idx].split()[0])
    idx = next_nonempty(idx + 1, lines)

    cell_rows = []
    for _ in range(3):
        cell_rows.append([float(x) * scale for x in lines[idx].split()[:3]])
        idx += 1
    cell = np.array(cell_rows, dtype=float)
    CellDim = pd.DataFrame(cell_rows, columns=["x", "y", "z"])

    idx = next_nonempty(idx, lines)
    elem_line = lines[idx].split()
    idx = next_nonempty(idx + 1, lines)
    num_line = lines[idx].split()
    counts = [int(x) for x in num_line]
    n_atoms = sum(counts)
    elements_expanded = expand_elements(elem_line, num_line)

    idx = next_nonempty(idx + 1, lines)
    header = lines[idx].strip().lower()
    if header.startswith("s"):
        idx = next_nonempty(idx + 1, lines)
        header = lines[idx].strip().lower()

    is_cart = header.startswith("c")

    idx = next_nonempty(idx + 1, lines)
    coords = []
    for _ in range(n_atoms):
        if idx >= len(lines):
            raise ValueError("Unexpected end of file while reading positions.")
        coords.append(parse_three_floats(lines[idx]))
        idx += 1
    coords = np.array(coords, dtype=float)
    frac = cart_to_frac(coords, cell) if is_cart else coords

    Position = pd.DataFrame(frac, columns=["x", "y", "z"])
    Position.insert(0, "Element", elements_expanded)

    if not give_velocities:
        return Position, CellDim

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

    if Velocities is not None:
        return Position, CellDim, Velocities
    else:
        return Position, CellDim


def WritePOSCAR(WorkDir,
                Position,
                CellDim,
                Velocities=None,
                FileName=None,
                Title="Structure generated by SLUSCHI and SGUSCHI"):
    """
    Writes a VASP-format POSCAR file.
    """
    if not os.path.exists(WorkDir):
        os.makedirs(WorkDir)
    if FileName is None:
        FileName = "POSCAR"
    filepath = os.path.join(WorkDir, FileName)
    required_cols = {"Element", "x", "y", "z"}
    if not required_cols.issubset(Position.columns):
        raise ValueError("Position DataFrame must contain columns: %s" % required_cols)
    if not {"x", "y", "z"}.issubset(CellDim.columns) or len(CellDim) != 3:
        raise ValueError("CellDim must be a 3x3 DataFrame with columns ['x','y','z'].")

    atom_counts = Position["Element"].value_counts(sort=False)
    element_order = list(atom_counts.index)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("%s\n" % Title)
        f.write("1.0\n")
        for i in range(3):
            f.write("%22.16f %22.16f %22.16f\n" % (
                CellDim.iloc[i]['x'], CellDim.iloc[i]['y'], CellDim.iloc[i]['z']))
        f.write("   " + "   ".join(element_order) + "\n")
        f.write("   " + "   ".join(str(atom_counts[e]) for e in element_order) + "\n")
        f.write("Direct\n")
        for elem in element_order:
            subset = Position[Position["Element"] == elem]
            for _, atom in subset.iterrows():
                f.write("%22.16f %22.16f %22.16f\n" % (
                    atom['x'], atom['y'], atom['z']))
        if Velocities is not None and not Velocities.empty:
            f.write("\n")
            for elem in element_order:
                subset = Velocities[Velocities["Element"] == elem]
                for _, v in subset.iterrows():
                    f.write("%22.16f %22.16f %22.16f\n" % (
                        v['vx'], v['vy'], v['vz']))
    return

# %%
