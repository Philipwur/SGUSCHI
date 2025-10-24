#%%
#Suite of functions for reading and creating VASP related files

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
from pathlib import Path
from typing import Optional, Any, List, Dict, Union, Tuple

import sys

#sys.path.append(str(Path(__file__).resolve().parents[1]))
#from workflow import OxidationAnalysis as an


#Old equations, need another pass

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


'''
need an xyz writer, with most relevatn data (time, energy, lattice vectors, positions)
Probably need to think about atom tracking carefully. 

#NOT WORKIGN BUT MAYBE DEFUNCT (MAYBE NOT THOUGH)
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
'''

# New Gen functions here

#Think this might be defunct
def FixElementFormatting(
    Position: pd.DataFrame, ReturnPrevNames: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    """
    Clean element names in a DataFrame to ensure consistency for bond-finding 
    algorithms and post processing. Useful to deal with corrupted element names.

    Removes suffixes such as "_x" or "/y" from element symbols in the "Element" column.
    Optionally returns the list of original unique element names for renaming later.

    Args:
        Position: DataFrame containing an "Element" column to clean.
        ReturnPrevNames: If True, also return the list of previous element names.

    Returns:
        Position if ReturnPrevNames is False,
        (Position, PrevNames) if True.
    """
    if ReturnPrevNames:
        PrevNames = Position["Element"].unique().tolist()

    for Elem in Position["Element"].unique():
        if "_" in Elem:
            FixedName = Elem.split("_")[0]
            Position.loc[Position["Element"] == Elem, "Element"] = FixedName
        elif "/" in Elem:
            FixedName = Elem.split("/")[0]
            Position.loc[Position["Element"] == Elem, "Element"] = FixedName

    if ReturnPrevNames:
        return Position, PrevNames
    else:
        return Position
    
    
def CheckForMLFF(WorkDir: Union[str, Path]) -> bool:
    """
    Check whether a VASP INCAR file enables the machine-learning force field (MLFF).
    Checks for the parameter `ML_LMLFF`.
    The parameter is expected to be either `.TRUE.` or `.FALSE.` (VASP-style logicals).

    Args:
        WorkDir (Union[str, Path]):
            Path to the INCAR file. 
            Must reference an existing INCAR file, not just a directory.

    Returns:
        bool:
            True if `ML_LMLFF = .TRUE.` is present in the INCAR file.

    Raises:
        ValueError:
            If `WorkDir` does not point to a file named 'INCAR'.
        FileNotFoundError:
            If the INCAR file cannot be found.
    """
    # --- Validate path ---
    if str(WorkDir).split("/")[-1] != "INCAR":
        raise ValueError("WorkDir must point to an INCAR file.")

    try:
        # Ensure Path object and read file
        IncarPath = Path(WorkDir)
        Params = ReadKeyValueFile(IncarPath, RequiredKeys=["ML_LMLFF"])
        MLFFValue = Params.get("ML_LMLFF", "").strip().upper()
        return MLFFValue == ".TRUE."
    
    except Exception:
        return False


def ReadKeyValueFile(FilePath: Path, RequiredKeys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse a simple key=value (or key value) configuration file.

    Supports:
      - Inline comments beginning with '#' or '!'
      - Empty lines and full-line comments
      - Flexible spacing around '='
      - Optional enforcement of required keys

    Works with files like INCAR, job.in, OxParams, or CovalentRadii.

    Args:
        FilePath (Path): Path to the input configuration file.
        RequiredKeys (Optional[List[str]]): List of required parameter names.
            If provided, the function raises ValueError if any are missing.

    Returns:
        Dict[str, Any]: Dictionary of parsed key-value pairs as strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required key is missing.
    """

    if not FilePath.exists():
        raise FileNotFoundError("Configuration file not found at: {}".format(FilePath))

    RawParams: Dict[str, str] = {}

    with FilePath.open("r", encoding="utf-8") as File:
        for Line in File:
            Line = Line.strip()
            if not Line:
                continue

            # Remove full-line comments
            if Line.startswith("#") or Line.startswith("!"):
                continue

            # Remove inline comments (after # or !)
            Line = re.split(r"[#!]", Line, 1)[0].strip()
            if not Line:
                continue

            # Accept either "key = value" or "key value"
            if "=" in Line:
                Key, Value = [x.strip() for x in Line.split("=", 1)]
            else:
                Parts = Line.split(None, 1)
                if len(Parts) != 2:
                    continue
                Key, Value = Parts[0].strip(), Parts[1].strip()

            if Key:
                RawParams[Key] = Value

    # --- Verify required keys ---
    if RequiredKeys:
        Missing = [Key for Key in RequiredKeys if Key not in RawParams or RawParams[Key] == ""]
        if Missing:
            raise ValueError("Missing required parameters: {}".format(", ".join(Missing)))

    return RawParams


def ReadPoscar(
    WorkDir: Optional[Union[str, Path]] = None,
    FileName: Optional[Union[str, Path]] = None,
    GiveVelocities: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Read a VASP POSCAR or CONTCAR file.

    Automatically locates POSCAR/CONTCAR in the given directory, parses lattice vectors,
    atomic positions, and optionally velocities. First tries for POSCAR unless specified in FileName.

    Args:
        WorkDir: Directory containing POSCAR or CONTCAR (defaults to current directory).
        FileName: Specific file to read (overrides automatic POSCAR/CONTCAR search).
        GiveVelocities: If True, return velocity data when available.

    Returns:
        (Position, CellDim) or (Position, CellDim, Velocities)
        where each is a pandas DataFrame.
    """
    def NextNonEmpty(Index: int, Lines: list) -> int:
        while Index < len(Lines) and not Lines[Index].strip():
            Index += 1
        return Index

    def ParseThreeFloats(Line: str) -> list:
        Values = []
        for Token in Line.split():
            try:
                Values.append(float(Token))
                if len(Values) == 3:
                    break
            except ValueError:
                continue
        if len(Values) != 3:
            raise ValueError("Could not parse 3 numeric coords from line: %r" % Line)
        return Values

    def ExpandElements(ElemLine: list, NumLine: list) -> list:
        Output = []
        for El, N in zip(ElemLine, NumLine):
            Output.extend([El] * int(N))
        return Output

    def CartToFrac(Cart: np.ndarray, Cell: np.ndarray) -> np.ndarray:
        InvCell = np.linalg.inv(Cell)
        return Cart @ InvCell

    Base = Path(WorkDir) if WorkDir is not None else Path.cwd()
    if FileName is None:
        PathFile = Base / "POSCAR"
        if not PathFile.exists():
            Alt = Base / "CONTCAR"
            if Alt.exists():
                PathFile = Alt
            else:
                raise FileNotFoundError("No POSCAR or CONTCAR found in %s" % Base)
    else:
        P = Path(FileName)
        PathFile = P if P.is_absolute() else (Base / P)
    if not PathFile.exists():
        raise FileNotFoundError("File not found: %s" % PathFile)

    with PathFile.open("r", encoding="utf-8") as F:
        Lines = [Ln.rstrip() for Ln in F]
    Start = NextNonEmpty(0, Lines)
    Lines = Lines[Start:]

    Idx = NextNonEmpty(1, Lines)
    Scale = float(Lines[Idx].split()[0])
    Idx = NextNonEmpty(Idx + 1, Lines)

    CellRows = []
    for _ in range(3):
        CellRows.append([float(X) * Scale for X in Lines[Idx].split()[:3]])
        Idx += 1
    Cell = np.array(CellRows, dtype=float)
    CellDim = pd.DataFrame(CellRows, columns=["x", "y", "z"])

    Idx = NextNonEmpty(Idx, Lines)
    ElemLine = Lines[Idx].split()
    Idx = NextNonEmpty(Idx + 1, Lines)
    NumLine = Lines[Idx].split()
    Counts = [int(X) for X in NumLine]
    NAtoms = sum(Counts)
    ElementsExpanded = ExpandElements(ElemLine, NumLine)

    Idx = NextNonEmpty(Idx + 1, Lines)
    Header = Lines[Idx].strip().lower()
    if Header.startswith("s"):
        Idx = NextNonEmpty(Idx + 1, Lines)
        Header = Lines[Idx].strip().lower()

    IsCart = Header.startswith("c")

    Idx = NextNonEmpty(Idx + 1, Lines)
    Coords = []
    for _ in range(NAtoms):
        if Idx >= len(Lines):
            raise ValueError("Unexpected end of file while reading positions.")
        Coords.append(ParseThreeFloats(Lines[Idx]))
        Idx += 1
    Coords = np.array(Coords, dtype=float)
    Frac = CartToFrac(Coords, Cell) if IsCart else Coords

    Position = pd.DataFrame(Frac, columns=["x", "y", "z"])
    Position.insert(0, "Element", ElementsExpanded)

    if not GiveVelocities:
        return Position, CellDim

    Idx = NextNonEmpty(Idx, Lines)
    Vels = []
    for _ in range(NAtoms):
        if Idx >= len(Lines) or not Lines[Idx].strip():
            break
        try:
            Vels.append(ParseThreeFloats(Lines[Idx]))
        except ValueError:
            break
        Idx += 1

    Velocities = None
    if len(Vels) == NAtoms:
        Velocities = pd.DataFrame(Vels, columns=["vx", "vy", "vz"])
        Velocities.insert(0, "Element", ElementsExpanded)

    if Velocities is not None:
        return Position, CellDim, Velocities
    else:
        return Position, CellDim


def WritePoscar(
    WorkDir: Union[str, Path],
    Position: pd.DataFrame,
    CellDim: pd.DataFrame,
    Velocities: Optional[pd.DataFrame] = None,
    FileName: Optional[str] = None,
    Title: str = "Structure generated by SLUSCHI + SGUSCHI"
) -> None:
    """
    Writes a VASP-format POSCAR file from internal CellDim and Position pd.

    Args:
        WorkDir: Directory in which to write the POSCAR file.
        Position: DataFrame with columns ['Element', 'x', 'y', 'z'] containing atomic positions (fractional).
        CellDim: 3×3 DataFrame with columns ['x', 'y', 'z'] defining lattice vectors.
        Velocities: Optional DataFrame with columns ['Element', 'vx', 'vy', 'vz'] for atomic velocities.
        FileName: Optional filename (defaults to 'POSCAR').
        Title: Title line to include in the POSCAR header.

    Returns:
        None
    """
    if not os.path.exists(WorkDir):
        os.makedirs(WorkDir)

    if FileName is None:
        FileName = "POSCAR"

    FilePath = os.path.join(WorkDir, FileName)

    RequiredCols = {"Element", "x", "y", "z"}
    if not RequiredCols.issubset(Position.columns):
        raise ValueError("Position DataFrame must contain columns: %s" % RequiredCols)

    if not {"x", "y", "z"}.issubset(CellDim.columns) or len(CellDim) != 3:
        raise ValueError("CellDim must be a 3×3 DataFrame with columns ['x','y','z'].")

    AtomCounts = Position["Element"].value_counts(sort=False)
    ElementOrder = list(AtomCounts.index)

    with open(FilePath, "w", encoding="utf-8") as F:
        F.write("%s\n" % Title)
        F.write("1.0\n")
        for i in range(3):
            F.write(
                "%22.16f %22.16f %22.16f\n"
                % (CellDim.iloc[i]["x"], CellDim.iloc[i]["y"], CellDim.iloc[i]["z"])
            )
        F.write("   " + "   ".join(ElementOrder) + "\n")
        F.write("   " + "   ".join(str(AtomCounts[E]) for E in ElementOrder) + "\n")
        F.write("Direct\n")
        for Elem in ElementOrder:
            Subset = Position[Position["Element"] == Elem]
            for _, Atom in Subset.iterrows():
                F.write("%22.16f %22.16f %22.16f\n" % (Atom["x"], Atom["y"], Atom["z"]))

        if Velocities is not None and not Velocities.empty:
            F.write("\n")
            for Elem in ElementOrder:
                Subset = Velocities[Velocities["Element"] == Elem]
                for _, V in Subset.iterrows():
                    F.write("%22.16f %22.16f %22.16f\n" % (V["vx"], V["vy"], V["vz"]))


def OutcarParser(WorkDir: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a VASP OUTCAR into a structured dictionary for saving and Postproc.

    Args:
        WorkDir (str or Path):
            Path to either a directory with OUTCAR file, or renamed OUTCAR file.
            If a directory is provided, function searches for OUTCAR
    
    Returns:
        Dict[str, Any]: A dictionary containing parsed simulation data with physical units:

        {
            "Temperature": float or None,
                Average target temperature from INCAR header (K).

            "TimesFs": List[float],
                Cumulative simulation time per ionic step (fs).
                Computed as (step_index + 1) × POTIM.

            "Energies": pd.DataFrame,
                Tabular data containing one row per ionic step, with columns:

                - Step (int): Step index (1-based).
                - EFree (float): Free energy of the ion–electron system (eV).
                - ETotal (float): Total energy (electronic + ionic) (eV).
                - EFermi (float): Fermi energy level (eV).
                - Temperature (float): Instantaneous ionic temperature (K), if available.
                - Pressure (float): Instantaneous pressure (bar if available, else kB).
                - Volume (float): Cell volume (Å³).
                - MaxForce (float): Maximum atomic force magnitude (eV/Å).
                - StressXX, StressYY, StressZZ (float): Normal stress components (kB).
                - StressXY, StressYZ, StressZX (float): Shear stress components (kB).

            "Positions": List[pd.DataFrame],
                One DataFrame per ionic step containing fractional coordinates:

                Columns:
                    - Element (str): Element symbol (e.g., "Zr", "O", "C").
                    - x, y, z (float): Fractional atomic positions (unitless).

            "CellVectors": pd.DataFrame,
                3×3 lattice vectors (Å) defining the simulation cell.
                Rows correspond to a, b, and c vectors; columns are ["x", "y", "z"].
                For ISIF=2 runs, the cell is fixed across all steps.
        }

    Notes:
        - Uses only the OUTCAR file (no dependency on INCAR, POSCAR, etc.).
        - Automatically detects energies, forces, and stresses across all steps.
        - Converts Cartesian coordinates to fractional using the fixed simulation cell.
        - Handles both standard and MLFF-type OUTCAR formats transparently.
    """
    
    # --- Resolve OUTCAR path ---
    WorkDir = Path(WorkDir)
    FilePath = WorkDir if (WorkDir.is_file() and WorkDir.name == "OUTCAR") else (WorkDir / "OUTCAR")
    if not FilePath.exists():
        raise FileNotFoundError("OUTCAR not found at: {}".format(FilePath))

    # Read OUTCAR
    OutcarText = FilePath.read_text(encoding="utf-8", errors="ignore")
    Lines = OutcarText.splitlines()
    Header = OutcarText[:150000]

       # --- Header tags: POTIM, TEBEG, TEEND ---
    def GetTagFloat(Tag: str) -> Optional[float]:
        Match = re.search(r"\b%s\s*=\s*([-\d\.Ee+]+)" % Tag, Header)
        return float(Match.group(1)) if Match else None

    Potim = GetTagFloat("POTIM")
    TEBEG = GetTagFloat("TEBEG")
    TEEND = GetTagFloat("TEEND")
    TargetTemperature = (
        0.5 * (TEBEG + TEEND) if (TEBEG is not None and TEEND is not None)
        else (TEBEG if TEBEG is not None else (TEEND if TEEND is not None else None))
    )

    # --- NIONS ---
    NionsMatch = re.search(r"\bNIONS\s*=\s*(\d+)", OutcarText)
    Nions = int(NionsMatch.group(1)) if NionsMatch else 0

    # --- Species (first contiguous POTCAR block) + counts → per-atom element labels ---
    PotcarLineIndices = [I for I, L in enumerate(Lines) if "POTCAR:" in L]
    Species: List[str] = []
    if PotcarLineIndices:
        StartIdx = PotcarLineIndices[0]
        BlockLines: List[str] = []
        I = StartIdx
        while I < len(Lines) and "POTCAR:" in Lines[I]:
            BlockLines.append(Lines[I])
            I += 1
        Periodic = set("""
            H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
            Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
            Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
            Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
            Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
            Rf Db Sg Bh Hs Mt Ds Rg Cn Fl Lv Ts Og
        """.split())
        
        for L in BlockLines:
            Payload = L.split("POTCAR:")[1]
            Tokens = Payload.replace("PAW_PBE", " ").replace("PAW-LDA", " ").split()
            Element = None
            for Tok in Tokens:
                Base = Tok.split("_")[0]
                if Base in Periodic:
                    Element = Base
                    break
            if Element:
                Species.append(Element)

    CountsMatch = re.search(r"ions\s+per\s+type\s*=\s*([0-9\.\s]+)", OutcarText, flags=re.I)
    Counts = [int(X) for X in CountsMatch.group(1).split()] if CountsMatch else []
    ElementsExpanded: List[str] = []
    if Species and Counts and len(Species) == len(Counts):
        for El, Cnt in zip(Species, Counts):
            ElementsExpanded.extend([El] * Cnt)
    elif Nions:
        ElementsExpanded = ["X"] * Nions

    # --- Cell (ISIF=2 → fixed cell). Parse the first 'direct lattice vectors' block ---
    def ParseFirstCell(OutcarLines: List[str]) -> Optional[pd.DataFrame]:
        for I, L in enumerate(OutcarLines):
            if re.search(r"direct\s+lattice\s+vectors", L, flags=re.I):
                Rows: List[List[float]] = []
                J = I + 1
                while J < len(OutcarLines) and len(Rows) < 3:
                    Floats = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?", OutcarLines[J])
                    if len(Floats) >= 3:
                        Rows.append([float(Floats[0]), float(Floats[1]), float(Floats[2])])
                    J += 1
                if len(Rows) == 3:
                    return pd.DataFrame(Rows, columns=["x", "y", "z"])
        return None

    CellVectors = ParseFirstCell(Lines)
    if CellVectors is not None:
        CellMat = CellVectors.values
        try:
            CellInv = np.linalg.inv(CellMat)
        except np.linalg.LinAlgError:
            CellInv = None
        Volume = float(abs(np.linalg.det(CellMat)))
    else:
        CellVectors = pd.DataFrame([[np.nan] * 3] * 3, columns=["x", "y", "z"])
        CellMat = None
        CellInv = None
        Volume = np.nan

    # --- Global events for robust last-step values (EFermi, Temp, Pressure, Stress) ---
    GlobalEvents: List[tuple] = []
    for M in re.finditer(r"\bE-fermi\s*:\s*([-\d\.Ee+]+)", OutcarText):
        GlobalEvents.append((M.start(), "Fermi", float(M.group(1))))
    for M in re.finditer(r"temperature\s+([-\d\.Ee+]+)\s*K", OutcarText, flags=re.I):
        GlobalEvents.append((M.start(), "Temp", float(M.group(1))))
    for M in re.finditer(r"total\s+pressure\s*=\s*([-\d\.Ee+]+)\s*kB", OutcarText, flags=re.I):
        GlobalEvents.append((M.start(), "PressKB", float(M.group(1))))
    for M in re.finditer(r"\bin kB\s+([-+\d\.\sEe]+)", OutcarText):
        Nums = [float(X) for X in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?", M.group(1))]
        if len(Nums) >= 6:
            GlobalEvents.append((M.start(), "Stress", Nums[:6]))
    GlobalEvents.sort(key=lambda T: T[0])

    EventIdx = 0
    EFermiLast = np.nan
    TempLast = np.nan
    PressLastKB = np.nan
    StressLast = [np.nan] * 6

    def UpdateGlobals(UptoOffset: int) -> None:
        nonlocal EventIdx, EFermiLast, TempLast, PressLastKB, StressLast
        while EventIdx < len(GlobalEvents) and GlobalEvents[EventIdx][0] <= UptoOffset:
            _, Kind, Payload = GlobalEvents[EventIdx]
            if Kind == "Fermi":
                EFermiLast = float(Payload)
            elif Kind == "Temp":
                TempLast = float(Payload)
            elif Kind == "PressKB":
                PressLastKB = float(Payload)
            elif Kind == "Stress":
                StressLast = list(Payload)
            EventIdx += 1

    # --- Step windows (robust for standard & MLFF prints) ---
    PosHeads = list(re.finditer(r"\n\s*POSITION\s+TOTAL-FORCE.*?\n", OutcarText))
    NumSteps = len(PosHeads)

    Positions: List[pd.DataFrame] = []
    Rows: List[Dict[str, Any]] = []

    for I, Head in enumerate(PosHeads):
        Start = Head.end()
        End = PosHeads[I + 1].start() if (I + 1) < NumSteps else len(OutcarText)

        # Capture globals seen up to this step
        UpdateGlobals(Start)

        Window = OutcarText[Start:End]
        WLines = Window.splitlines()

        # Skip dashed separator if present
        LineIdx = 0
        if LineIdx < len(WLines) and set(WLines[LineIdx].strip()) == set("-"):
            LineIdx += 1

        # Parse Cartesian positions and forces for Nions lines
        Coords: List[List[float]] = []
        Forces: List[List[float]] = []
        for _ in range(Nions):
            if LineIdx >= len(WLines):
                break
            Parts = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?", WLines[LineIdx])
            if len(Parts) >= 3:
                Coords.append([float(Parts[0]), float(Parts[1]), float(Parts[2])])
                if len(Parts) >= 6:
                    Forces.append([float(Parts[3]), float(Parts[4]), float(Parts[5])])
            LineIdx += 1

        # Convert to FRACTIONAL coordinates
        PosArr = np.array(Coords, dtype=float) if Coords else np.zeros((0, 3), dtype=float)
        if PosArr.size and (CellInv is not None):
            FracArr = PosArr.dot(CellInv)
        else:
            FracArr = PosArr.copy()
        Position = pd.DataFrame(FracArr, columns=["x", "y", "z"])
        if ElementsExpanded and len(ElementsExpanded) == len(Position):
            Position.insert(0, "Element", ElementsExpanded)
        else:
            Position.insert(0, "Element", ["X"] * len(Position))
        Positions.append(Position)

        # Max force magnitude
        MaxForce = np.nan
        if Forces:
            FArr = np.array(Forces, dtype=float)
            MaxForce = float(np.sqrt((FArr ** 2).sum(axis=1)).max())

        # Energies in this window (use last occurrence)
        EFree = np.nan
        Hits = list(re.finditer(r"free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)", Window, flags=re.I))
        if Hits:
            EFree = float(Hits[-1].group(1))

        ETotal = np.nan
        Hits = list(re.finditer(r"(?:energy\s+without\s+entropy|energy\(sigma->0\))\s*=\s*([-\d\.Ee+]+)", Window, flags=re.I))
        if Hits:
            ETotal = float(Hits[-1].group(1))
        if np.isnan(ETotal):
            ETotal = EFree

        # Per-window temperature, pressure, EFermi, stress with global fallbacks
        MTemp = list(re.finditer(r"temperature\s+([-\d\.Ee+]+)\s*K", Window, flags=re.I))
        StepTemp = float(MTemp[-1].group(1)) if MTemp else TempLast

        MPress = list(re.finditer(r"total\s+pressure\s*=\s*([-\d\.Ee+]+)\s*kB", Window, flags=re.I))
        StepPress = float(MPress[-1].group(1)) if MPress else PressLastKB

        MFermi = list(re.finditer(r"\bE-fermi\s*:\s*([-\d\.Ee+]+)", Window))
        StepFermi = float(MFermi[-1].group(1)) if MFermi else EFermiLast

        MStress = list(re.finditer(r"\bin kB\s+([-+\d\.\sEe]+)", Window))
        if MStress:
            Nums = [float(X) for X in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?", MStress[-1].group(1))]
            StepStress = Nums[:6] if len(Nums) >= 6 else StressLast
        else:
            StepStress = StressLast

        # Advance globals to end-of-window (helps the *next* step's fallback)
        UpdateGlobals(End)

        StressXX, StressYY, StressZZ, StressXY, StressYZ, StressZX = (StepStress + [np.nan] * 6)[:6]

        Rows.append({
            "Step": I + 1,
            "EFree": EFree,
            "ETotal": ETotal,
            "EFermi": StepFermi,
            "Temperature": StepTemp,   # K
            "Pressure": StepPress,     # kB
            "Volume": Volume,          # Å^3
            "MaxForce": MaxForce,      # eV/Å
            "StressXX": StressXX, "StressYY": StressYY, "StressZZ": StressZZ,
            "StressXY": StressXY, "StressYZ": StressYZ, "StressZX": StressZX,
        })

    # --- Times (fs) from POTIM ---
    TimesFs = [float((I + 1) * (Potim if Potim is not None else 1.0)) for I in range(NumSteps)]

    Energies = pd.DataFrame(Rows)

    return {
        "Temperature": float(TargetTemperature) if TargetTemperature is not None else None,
        "TimesFs": TimesFs,
        "Energies": Energies,
        "Positions": Positions,
        "CellVectors": CellVectors,
    }

    
'''
#Testing OUTCAR Parser
if __name__ == "__main__":
    OUTCARData = OutcarParser(WorkDir="../../Test/OUTCAR")
    print(OUTCARData['Temperature'], 'K\n')
    print('Times (fs):', OUTCARData['TimesFs'], '\n')
    print('Positions:', OUTCARData['Positions'], '\n')
    print('Cell Vectors:', OUTCARData['CellVectors'], '\n')
    
    for key in OUTCARData['Energies']:
        print(f'{key} =', OUTCARData['Energies'][key], '\n') 
'''
# %%
