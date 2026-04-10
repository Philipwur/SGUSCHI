import re
import sys
import os
import queue
import concurrent.futures
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from multiprocessing import Manager
from tqdm import tqdm
import scipy.sparse as Sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(r"c:\Users\pdwurzner\OneDrive - Delft University of Technology\Research\Shared_workspace\SGUSCHI\SGUSCHI\src\workflow")))
from OxidationAnalysis import FindGases


# ==========================================
# Physical Constants
# ==========================================

_O2_MASS_KG = 32.0 * 1.66054e-27     # kg
_KB          = 1.380649e-23           # J K⁻¹


def _SafeQueuePut(ProgressQueue, Message: tuple) -> None:
    """Best-effort queue write for cross-process progress events."""
    if ProgressQueue is None:
        return
    try:
        ProgressQueue.put(Message)
    except Exception:
        # Progress reporting must never break the main calculation.
        pass


def _CountStrideFrames(XyzFilePath: Path, Stride: int = 1) -> int:
    """
    Count frames that will be processed for a given stride.
    Enables a meaningful tqdm total in parallel mode.
    """
    FrameIndex = 0
    Selected   = 0
    with open(XyzFilePath, 'r', encoding='utf-8') as File:
        while True:
            NumAtomsLine = File.readline()
            if not NumAtomsLine:
                break
            try:
                NumAtoms = int(NumAtomsLine.strip())
            except ValueError:
                break
            File.readline()  # comment line
            for _ in range(NumAtoms):
                File.readline()
            if FrameIndex % Stride == 0:
                Selected += 1
            FrameIndex += 1
    return Selected


def _DrainProgressQueue(
    ProgressQueue,
    FrameBar,
    ActiveTasks: set,
    CompletedFiles: int,
    TotalFiles: int,
) -> None:
    """Drain queued worker progress events and refresh tqdm postfix."""
    Updated = False
    while True:
        try:
            Event = ProgressQueue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break

        if not Event:
            continue
        Kind = Event[0]

        if Kind == 'frames':
            FrameBar.update(int(Event[2]))
            Updated = True
        elif Kind == 'start':
            ActiveTasks.add(str(Event[1]))
            Updated = True
        elif Kind == 'done':
            ActiveTasks.discard(str(Event[1]))
            Updated = True

    if Updated:
        FrameBar.set_postfix_str(f"files {CompletedFiles}/{TotalFiles} | active {len(ActiveTasks)}")


def ComputeImpingementFlux(PressureBar, TempK):
    """
    Hertz-Knudsen impingement flux of O2:
        Z = P / sqrt(2 pi m k T)   [molecules m⁻² s⁻¹]

    PressureBar : float or array, O2 partial pressure in bar
    TempK       : float, temperature in K
    """
    P_Pa = np.asarray(PressureBar, dtype=float) * 1.0e5
    return P_Pa / np.sqrt(2.0 * np.pi * _O2_MASS_KG * _KB * TempK)


# ==========================================
# Volume Estimation
# ==========================================

def _FrameVolumeZrDistance(
    CellMat:   np.ndarray,
    Elements:  list,
    CartArray: np.ndarray,
) -> tuple:
    """
    ZrDistance volume for a single already-parsed frame.

    Returns (EffectiveVolumeM3, CellVolumeM3).
    CellMat  : (3,3) row-vector lattice in Angstrom.
    CartArray: (N,3) Cartesian coordinates in Angstrom.
    """
    CellVolA3 = abs(np.dot(CellMat[0], np.cross(CellMat[1], CellMat[2])))
    try:
        InvCell = np.linalg.inv(CellMat)
    except np.linalg.LinAlgError:
        InvCell = np.eye(3)

    ZrMask = np.array([e == 'Zr' for e in Elements])
    if np.any(ZrMask):
        FracX       = np.sort((CartArray[ZrMask] @ InvCell)[:, 0] % 1.0)
        Diffs       = np.diff(FracX)
        WrapGap     = 1.0 - FracX[-1] + FracX[0]
        GasFraction = max(WrapGap, np.max(Diffs)) if Diffs.size > 0 else WrapGap
    else:
        GasFraction = 1.0

    return CellVolA3 * GasFraction * 1e-30, CellVolA3 * 1e-30


def _FrameVolumeRayCast(
    CellMat:   np.ndarray,
    Elements:  list,
    CartArray: np.ndarray,
    GridResA:  float = 1.0,
) -> tuple:
    """
    RayCast volume for a single already-parsed frame.

    Returns (EffectiveVolumeM3, CellVolumeM3).
    CellMat  : (3,3) row-vector lattice in Angstrom.
    CartArray: (N,3) Cartesian coordinates in Angstrom.
    """
    ProbeRadiusA = 1.73
    SlabCutoffA  = 3.5
    RadiiDict    = {
        'Zr': 2.36 + ProbeRadiusA,
        'O':  1.52 + ProbeRadiusA,
        'C':  1.70 + ProbeRadiusA,
        'N':  1.55 + ProbeRadiusA,
    }

    CellX     = np.linalg.norm(CellMat[0])
    CellY     = np.linalg.norm(CellMat[1])
    CellZ     = np.linalg.norm(CellMat[2])
    YZArea    = np.linalg.norm(np.cross(CellMat[1], CellMat[2]))
    XProbe    = 0.66 * CellX
    CellVolA3 = abs(np.dot(CellMat[0], np.cross(CellMat[1], CellMat[2])))

    Radii  = np.array([RadiiDict.get(e, 2.0 + ProbeRadiusA) for e in Elements])
    Coords = CartArray % np.array([CellX, CellY, CellZ])

    Tree  = KDTree(Coords, boxsize=[CellX, CellY, CellZ])
    Pairs = Tree.query_pairs(r=SlabCutoffA)
    if Pairs:
        Rows, Cols = zip(*Pairs)
        N          = len(Elements)
        AdjGraph   = Sp.coo_matrix(
            (np.ones(len(Rows)), (Rows, Cols)),
            shape=(N, N),
        )
        _, Labels                 = connected_components(csgraph=AdjGraph, directed=False)
        UniqueLabels, LabelCounts = np.unique(Labels, return_counts=True)
        SlabMask = Labels == UniqueLabels[np.argmax(LabelCounts)]
        Coords   = Coords[SlabMask]
        Radii    = Radii[SlabMask]

    YGrid        = np.arange(0, CellY, GridResA)
    ZGrid        = np.arange(0, CellZ, GridResA)
    YMesh, ZMesh = np.meshgrid(YGrid, ZGrid)
    GridPoints   = np.vstack([YMesh.ravel(), ZMesh.ravel()]).T
    GasHeights   = np.zeros(len(GridPoints))

    for J, (Yg, Zg) in enumerate(GridPoints):
        Dy      = np.minimum(np.abs(Coords[:, 1] - Yg), CellY - np.abs(Coords[:, 1] - Yg))
        Dz      = np.minimum(np.abs(Coords[:, 2] - Zg), CellZ - np.abs(Coords[:, 2] - Zg))
        D2      = Dy**2 + Dz**2
        HitMask = D2 < Radii**2

        if not np.any(HitMask):
            GasHeights[J] = CellX
            continue

        Dx       = np.sqrt(Radii[HitMask]**2 - D2[HitMask])
        XCenters = Coords[HitMask, 0]

        Intervals = []
        for S, E in zip(XCenters - Dx, XCenters + Dx):
            if S < 0:
                Intervals.extend([[S + CellX, CellX], [0.0, E]])
            elif E > CellX:
                Intervals.extend([[S, CellX], [0.0, E - CellX]])
            else:
                Intervals.append([S, E])

        Intervals.sort(key=lambda I: I[0])
        Merged = []
        for Interval in Intervals:
            if not Merged or Interval[0] > Merged[-1][1]:
                Merged.append(list(Interval))
            else:
                Merged[-1][1] = max(Merged[-1][1], Interval[1])

        GapIntervals = []
        CurrX = 0.0
        for S, E in Merged:
            if S > CurrX:
                GapIntervals.append((CurrX, S))
            CurrX = max(CurrX, E)
        if CurrX < CellX:
            GapIntervals.append((CurrX, CellX))

        GasHeight = 0.0
        for S, E in GapIntervals:
            if S <= XProbe <= E:
                GasHeight = E - S
                break
        GasHeights[J] = GasHeight

    return np.mean(GasHeights) * YZArea * 1e-30, CellVolA3 * 1e-30


def EstimateVolumeZrDistance(
    XyzFilePath:   Path,
    Stride:        int = 1,
    ProgressQueue       = None,
    ProgressTask:  str  = None,
    ProgressEvery: int  = 25,
) -> pd.DataFrame:
    """
    Estimates gas volume per frame by finding the largest gap in Zr fractional
    coordinates along the first lattice vector (a-axis).

    The gap fraction is multiplied by the full cell volume to give an effective
    gas volume. Assumes the gas phase is a contiguous slab perpendicular to a.

    NOTE: Only valid when the surface normal is aligned with the a-axis. If VASP
    uses c as the surface normal, use EstimateVolumeRayCast or rotate the cell.

    Returns a DataFrame with columns:
        'Time (fs)', 'Effective Volume (m^3)', 'Cell Volume (m^3)'
    """
    ParsedData = []
    FrameIndex = 0
    PendingProgress = 0

    LatticeRegex = re.compile(r'Lattice="([^"]+)"')
    TimeRegex    = re.compile(r'Time=([\d\.]+)')

    with open(XyzFilePath, 'r', encoding='utf-8') as File:
        while True:
            NumAtomsLine = File.readline()
            if not NumAtomsLine:
                break
            try:
                NumAtoms = int(NumAtomsLine.strip())
            except ValueError:
                break

            CommentLine = File.readline()

            if FrameIndex % Stride != 0:
                for _ in range(NumAtoms):
                    File.readline()
                FrameIndex += 1
                continue

            PendingProgress += 1
            if PendingProgress >= ProgressEvery:
                _SafeQueuePut(ProgressQueue, ('frames', ProgressTask, PendingProgress))
                PendingProgress = 0

            LatticeMatch = LatticeRegex.search(CommentLine)
            TimeMatch    = TimeRegex.search(CommentLine)

            if LatticeMatch and TimeMatch:
                Values    = [float(V) for V in LatticeMatch.group(1).split()]
                CellMat   = np.array([Values[0:3], Values[3:6], Values[6:9]])
                CellVolA3 = np.abs(np.dot(CellMat[0], np.cross(CellMat[1], CellMat[2])))

                try:
                    InvCell = np.linalg.inv(CellMat)
                except np.linalg.LinAlgError:
                    InvCell = np.eye(3)

                ZrCartesian = []
                for _ in range(NumAtoms):
                    Parts = File.readline().split()
                    if Parts[0] == 'Zr':
                        ZrCartesian.append([float(Parts[1]), float(Parts[2]), float(Parts[3])])

                if ZrCartesian:
                    # Project Zr onto a-axis in fractional coordinates
                    FracX       = np.sort((np.array(ZrCartesian) @ InvCell)[:, 0] % 1.0)
                    Diffs       = np.diff(FracX)
                    WrapGap     = 1.0 - FracX[-1] + FracX[0]
                    GasFraction = max(WrapGap, np.max(Diffs)) if Diffs.size > 0 else WrapGap
                else:
                    GasFraction = 1.0

                ParsedData.append({
                    'Time (fs)':              float(TimeMatch.group(1)),
                    'Effective Volume (m^3)': CellVolA3 * GasFraction * 1e-30,
                    'Cell Volume (m^3)':      CellVolA3 * 1e-30,
                })
            else:
                for _ in range(NumAtoms):
                    File.readline()

            FrameIndex += 1

    if PendingProgress > 0:
        _SafeQueuePut(ProgressQueue, ('frames', ProgressTask, PendingProgress))

    return pd.DataFrame(ParsedData)


def EstimateVolumeRayCast(
    XyzFilePath:   Path,
    Stride:        int   = 1,
    GridResA:      float = 1.0,
    ProgressQueue        = None,
    ProgressTask:  str   = None,
    ProgressEvery: int   = 25,
) -> pd.DataFrame:
    """
    Estimates gas volume per frame by ray-casting along the x-axis.

    For each (y, z) grid point, atom spheres (covalent radius + O2 probe radius)
    are projected onto the x-axis. Overlapping intervals are merged and the empty
    space at XProbe = 0.66 * CellX is recorded as the local gas height.
    The mean gas height * YZ cross-sectional area gives the effective gas volume.

    The slab is identified as the largest bonded connected component, so isolated
    gas molecules do not contribute to the excluded volume.

    NOTE: Assumes the surface normal is along x and the cell is orthorhombic
    (Cartesian PBC wrapping uses axis lengths only).

    Returns a DataFrame with columns:
        'Time (fs)', 'Effective Volume (m^3)', 'Cell Volume (m^3)'
    """
    ParsedData      = []
    FrameIndex      = 0
    ProbeRadiusA    = 1.73   # O2 molecular radius for excluded-volume probe
    SlabCutoffA     = 3.5    # Bond cutoff for identifying the slab component
    PendingProgress = 0

    LatticeRegex = re.compile(r'Lattice="([^"]+)"')
    TimeRegex    = re.compile(r'Time=([\d\.]+)')
    RadiiDict    = {
        'Zr': 2.36 + ProbeRadiusA,
        'O':  1.52 + ProbeRadiusA,
        'C':  1.70 + ProbeRadiusA,
        'N':  1.55 + ProbeRadiusA,
    }

    with open(XyzFilePath, 'r', encoding='utf-8') as File:
        while True:
            NumAtomsLine = File.readline()
            if not NumAtomsLine:
                break
            try:
                NumAtoms = int(NumAtomsLine.strip())
            except ValueError:
                break

            CommentLine = File.readline()

            if FrameIndex % Stride != 0:
                for _ in range(NumAtoms):
                    File.readline()
                FrameIndex += 1
                continue

            PendingProgress += 1
            if PendingProgress >= ProgressEvery:
                _SafeQueuePut(ProgressQueue, ('frames', ProgressTask, PendingProgress))
                PendingProgress = 0

            LatticeMatch = LatticeRegex.search(CommentLine)
            TimeMatch    = TimeRegex.search(CommentLine)

            if LatticeMatch and TimeMatch:
                TimeFs  = float(TimeMatch.group(1))
                Values  = [float(V) for V in LatticeMatch.group(1).split()]
                CellMat = np.array([Values[0:3], Values[3:6], Values[6:9]])

                CellX  = np.linalg.norm(CellMat[0])
                CellY  = np.linalg.norm(CellMat[1])
                CellZ  = np.linalg.norm(CellMat[2])
                YZArea = np.linalg.norm(np.cross(CellMat[1], CellMat[2]))
                XProbe = 0.66 * CellX

                Coords, Radii = [], []
                for _ in range(NumAtoms):
                    Parts = File.readline().split()
                    Coords.append([float(Parts[1]), float(Parts[2]), float(Parts[3])])
                    Radii.append(RadiiDict.get(Parts[0], 2.0 + ProbeRadiusA))

                Coords = np.array(Coords) % np.array([CellX, CellY, CellZ])
                Radii  = np.array(Radii)

                # Identify slab as the largest bonded component
                Tree  = KDTree(Coords, boxsize=[CellX, CellY, CellZ])
                Pairs = Tree.query_pairs(r=SlabCutoffA)
                if Pairs:
                    Rows, Cols = zip(*Pairs)
                    AdjGraph   = Sp.coo_matrix(
                        (np.ones(len(Rows)), (Rows, Cols)),
                        shape=(NumAtoms, NumAtoms),
                    )
                    _, Labels                 = connected_components(csgraph=AdjGraph, directed=False)
                    UniqueLabels, LabelCounts = np.unique(Labels, return_counts=True)
                    SlabMask = Labels == UniqueLabels[np.argmax(LabelCounts)]
                    Coords   = Coords[SlabMask]
                    Radii    = Radii[SlabMask]

                # Ray cast: for each (y,z) find the gas height at XProbe
                YGrid        = np.arange(0, CellY, GridResA)
                ZGrid        = np.arange(0, CellZ, GridResA)
                YMesh, ZMesh = np.meshgrid(YGrid, ZGrid)
                GridPoints   = np.vstack([YMesh.ravel(), ZMesh.ravel()]).T
                GasHeights   = np.zeros(len(GridPoints))

                for J, (Yg, Zg) in enumerate(GridPoints):
                    Dy      = np.minimum(np.abs(Coords[:, 1] - Yg), CellY - np.abs(Coords[:, 1] - Yg))
                    Dz      = np.minimum(np.abs(Coords[:, 2] - Zg), CellZ - np.abs(Coords[:, 2] - Zg))
                    D2      = Dy**2 + Dz**2
                    HitMask = D2 < Radii**2

                    if not np.any(HitMask):
                        GasHeights[J] = CellX
                        continue

                    Dx       = np.sqrt(Radii[HitMask]**2 - D2[HitMask])
                    XCenters = Coords[HitMask, 0]

                    # Build PBC-aware solid intervals along x, then merge overlaps
                    Intervals = []
                    for S, E in zip(XCenters - Dx, XCenters + Dx):
                        if S < 0:
                            Intervals.extend([[S + CellX, CellX], [0.0, E]])
                        elif E > CellX:
                            Intervals.extend([[S, CellX], [0.0, E - CellX]])
                        else:
                            Intervals.append([S, E])

                    Intervals.sort(key=lambda I: I[0])
                    Merged = []
                    for Interval in Intervals:
                        if not Merged or Interval[0] > Merged[-1][1]:
                            Merged.append(list(Interval))
                        else:
                            Merged[-1][1] = max(Merged[-1][1], Interval[1])

                    # Invert merged solid intervals to find empty (gas) gaps
                    GapIntervals = []
                    CurrX = 0.0
                    for S, E in Merged:
                        if S > CurrX:
                            GapIntervals.append((CurrX, S))
                        CurrX = max(CurrX, E)
                    if CurrX < CellX:
                        GapIntervals.append((CurrX, CellX))

                    GasHeight = 0.0
                    for S, E in GapIntervals:
                        if S <= XProbe <= E:
                            GasHeight = E - S
                            break
                    GasHeights[J] = GasHeight

                ParsedData.append({
                    'Time (fs)':              TimeFs,
                    'Effective Volume (m^3)': np.mean(GasHeights) * YZArea * 1e-30,
                    'Cell Volume (m^3)':      CellX * YZArea * 1e-30,
                })
            else:
                for _ in range(NumAtoms):
                    File.readline()

            FrameIndex += 1

    if PendingProgress > 0:
        _SafeQueuePut(ProgressQueue, ('frames', ProgressTask, PendingProgress))

    return pd.DataFrame(ParsedData)


# ==========================================
# Molecule Counting
# ==========================================

def CountMoleculesRateAnalysis(CsvFilePath: Path) -> pd.DataFrame:
    """
    Read O2 counts from a pre-computed RateAnalysis CSV.

    Returns a DataFrame with columns ['Time (fs)', 'O2 Count'].
    """
    if not CsvFilePath.exists():
        return pd.DataFrame(columns=['Time (fs)', 'O2 Count'])
    RateData = pd.read_csv(CsvFilePath)
    if 'Time (fs)' in RateData.columns and 'O2 Count' in RateData.columns:
        return RateData[['Time (fs)', 'O2 Count']].sort_values('Time (fs)')
    return pd.DataFrame(columns=['Time (fs)', 'O2 Count'])


def CountMoleculesXYZFindGases(
    XyzFilePath:   Path,
    Stride:        int = 1,
    ProgressQueue       = None,
    ProgressTask:  str  = None,
    ProgressEvery: int  = 25,
) -> pd.DataFrame:
    """
    Count O2 molecules per XYZ frame using FindGases from OxidationAnalysis.

    Bonding rules enforced by FindGases:
      - Each O atom participates in at most one O-O bond (EnforceUniqueOOBonds),
        preventing fictitious O3/O4 clusters in the high-density gas phase.
      - Any O bonded to Zr joins the large slab connected component and is
        excluded from gas-phase counting via the MaximumComplexity=2 filter.

    Fractional coordinates are passed to FindGases using the extended-XYZ
    row-vector lattice convention: Cartesian = fractional @ Lattice.

    Returns a DataFrame with columns ['Time (fs)', 'O2 Count'].
    """
    CovalentRadii   = {'C': 0.77, 'Zr': 1.45, 'O': 0.66, 'N': 0.71}
    AtomicRadiusTol = 1.5

    LatticeRegex = re.compile(r'Lattice="([^"]+)"')
    TimeRegex    = re.compile(r'(?:Time|t)\s*[:=]\s*([\d\.]+)', re.IGNORECASE)

    Records    = []
    FrameIndex = 0
    PendingProgress = 0

    with open(XyzFilePath, 'r', encoding='utf-8') as F:
        while True:
            NumAtomsLine = F.readline()
            if not NumAtomsLine:
                break
            try:
                NumAtoms = int(NumAtomsLine.strip())
            except ValueError:
                break

            CommentLine = F.readline()

            if FrameIndex % Stride != 0:
                for _ in range(NumAtoms):
                    F.readline()
                FrameIndex += 1
                continue

            PendingProgress += 1
            if PendingProgress >= ProgressEvery:
                _SafeQueuePut(ProgressQueue, ('frames', ProgressTask, PendingProgress))
                PendingProgress = 0

            AtomLines = [F.readline() for _ in range(NumAtoms)]

            # Parse time
            TimeMatch = TimeRegex.search(CommentLine)
            if not TimeMatch:
                AnyMatch = re.search(r'([\d\.]+)', CommentLine)
                TimeFs   = float(AnyMatch.group(1)) if AnyMatch else 0.0
            else:
                TimeFs = float(TimeMatch.group(1))

            # Parse lattice (extended XYZ: row vectors, Cartesian = fractional @ Lattice)
            LatticeMatch = LatticeRegex.search(CommentLine)
            if not LatticeMatch:
                FrameIndex += 1
                continue

            Nums = [float(X) for X in LatticeMatch.group(1).split()]
            if len(Nums) != 9:
                FrameIndex += 1
                continue

            Lattice    = np.array(Nums).reshape(3, 3)
            InvLattice = np.linalg.inv(Lattice)

            # Parse atoms and convert Cartesian -> fractional
            Elements, CartCoords = [], []
            for Line in AtomLines:
                Parts = Line.split()
                if len(Parts) < 4:
                    continue
                Elements.append(Parts[0])
                CartCoords.append([float(Parts[1]), float(Parts[2]), float(Parts[3])])

            if not Elements:
                FrameIndex += 1
                continue

            UnknownElements = set(Elements) - set(CovalentRadii.keys())
            if UnknownElements:
                tqdm.write(f"  [WARNING] Unknown elements {UnknownElements} at {TimeFs} fs — skipping frame.")
                FrameIndex += 1
                continue

            FracCoords = np.mod(np.array(CartCoords) @ InvLattice, 1.0)
            Position   = pd.DataFrame({
                'Element': Elements,
                'x': FracCoords[:, 0],
                'y': FracCoords[:, 1],
                'z': FracCoords[:, 2],
            })

            Gases = FindGases(
                Position,
                Lattice,
                CovalentRadii=CovalentRadii,
                AtomicRadiusTol=AtomicRadiusTol,
                MinimumComplexity=2,
                MaximumComplexity=2,
            )

            O2Count = int(sum(M == ('O', 'O') for M in Gases['Molecule']))
            Records.append({'Time (fs)': TimeFs, 'O2 Count': O2Count})
            FrameIndex += 1

    if PendingProgress > 0:
        _SafeQueuePut(ProgressQueue, ('frames', ProgressTask, PendingProgress))

    return pd.DataFrame(Records)


def ParseXYZSinglePass(
    XyzFilePath:  Path,
    VolumeMethod: str   = 'ZrDistance',
    Stride:       int   = 1,
    GridResA:     float = 1.0,
    ProgressQueue       = None,
    ProgressTask: str   = None,
    ProgressEvery: int  = 25,
) -> tuple:
    """
    Single-pass XYZ reader combining volume estimation and O2 counting.

    Parses each frame once and calls the appropriate frame-level helpers,
    halving I/O cost compared to calling EstimateVolume* and
    CountMoleculesXYZFindGases separately.  Only useful when CountingMethod
    is 'XYZFindGases' (RateAnalysis reads a CSV, not the XYZ file).

    Returns (VolumeData, CountData) with the same column schemas as
    EstimateVolume* and CountMoleculesXYZFindGases respectively.
    """
    CovalentRadii   = {'C': 0.77, 'Zr': 1.45, 'O': 0.66, 'N': 0.71}
    AtomicRadiusTol = 1.5

    LatticeRegex = re.compile(r'Lattice="([^"]+)"')
    TimeRegex    = re.compile(r'(?:Time|t)\s*[:=]\s*([\d\.]+)', re.IGNORECASE)

    VolumeRecords    = []
    CountRecords     = []
    FrameIndex       = 0
    PendingProgress  = 0

    with open(XyzFilePath, 'r', encoding='utf-8') as F:
        while True:
            NumAtomsLine = F.readline()
            if not NumAtomsLine:
                break
            try:
                NumAtoms = int(NumAtomsLine.strip())
            except ValueError:
                break

            CommentLine = F.readline()

            if FrameIndex % Stride != 0:
                for _ in range(NumAtoms):
                    F.readline()
                FrameIndex += 1
                continue

            PendingProgress += 1
            if PendingProgress >= ProgressEvery:
                _SafeQueuePut(ProgressQueue, ('frames', ProgressTask, PendingProgress))
                PendingProgress = 0

            AtomLines = [F.readline() for _ in range(NumAtoms)]

            TimeMatch = TimeRegex.search(CommentLine)
            if not TimeMatch:
                AnyMatch = re.search(r'([\d\.]+)', CommentLine)
                TimeFs   = float(AnyMatch.group(1)) if AnyMatch else 0.0
            else:
                TimeFs = float(TimeMatch.group(1))

            LatticeMatch = LatticeRegex.search(CommentLine)
            if not LatticeMatch:
                FrameIndex += 1
                continue

            Nums = [float(X) for X in LatticeMatch.group(1).split()]
            if len(Nums) != 9:
                FrameIndex += 1
                continue

            Lattice    = np.array(Nums).reshape(3, 3)
            InvLattice = np.linalg.inv(Lattice)

            Elements, CartCoords = [], []
            for Line in AtomLines:
                Parts = Line.split()
                if len(Parts) < 4:
                    continue
                Elements.append(Parts[0])
                CartCoords.append([float(Parts[1]), float(Parts[2]), float(Parts[3])])

            if not Elements:
                FrameIndex += 1
                continue

            CartArray = np.array(CartCoords)

            # ── Volume (single-pass frame helper) ──────────────────────────────
            if VolumeMethod == 'RayCast':
                EffVolM3, CellVolM3 = _FrameVolumeRayCast(Lattice, Elements, CartArray, GridResA)
            else:
                EffVolM3, CellVolM3 = _FrameVolumeZrDistance(Lattice, Elements, CartArray)

            VolumeRecords.append({
                'Time (fs)':              TimeFs,
                'Effective Volume (m^3)': EffVolM3,
                'Cell Volume (m^3)':      CellVolM3,
            })

            # ── O2 count ───────────────────────────────────────────────────────
            UnknownElements = set(Elements) - set(CovalentRadii.keys())
            if UnknownElements:
                tqdm.write(f"  [WARNING] Unknown elements {UnknownElements} at {TimeFs} fs — skipping O2 count.")
                CountRecords.append({'Time (fs)': TimeFs, 'O2 Count': 0})
                FrameIndex += 1
                continue

            FracCoords = np.mod(CartArray @ InvLattice, 1.0)
            Position   = pd.DataFrame({
                'Element': Elements,
                'x': FracCoords[:, 0],
                'y': FracCoords[:, 1],
                'z': FracCoords[:, 2],
            })
            Gases   = FindGases(
                Position, Lattice,
                CovalentRadii=CovalentRadii,
                AtomicRadiusTol=AtomicRadiusTol,
                MinimumComplexity=2,
                MaximumComplexity=2,
            )
            CountRecords.append({
                'Time (fs)': TimeFs,
                'O2 Count':  int(sum(M == ('O', 'O') for M in Gases['Molecule'])),
            })
            FrameIndex += 1

    if PendingProgress > 0:
        _SafeQueuePut(ProgressQueue, ('frames', ProgressTask, PendingProgress))

    return pd.DataFrame(VolumeRecords), pd.DataFrame(CountRecords)


# ==========================================
# Figures
# ==========================================

def PlotConvergenceFigure(
    AlignedData:    pd.DataFrame,
    Temperature:    int,
    RunNumber:      int,
    VolumeMethod:   str,
    CountingMethod: str,
    OutputDir:      Path,
) -> None:
    """
    Per-run two-panel convergence figure.

    Top    — O2 partial pressure: instantaneous (grey), cumulative average (red dashed).
    Bottom — Hertz-Knudsen impingement flux: same two traces.
    """
    TimePsArray = AlignedData['Time (fs)'].to_numpy() / 1000.0
    RunLabel    = f"{Temperature} K  Run {RunNumber}"

    Figure = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            "Partial Pressure Convergence",
            "Impingement Flux Convergence",
        ),
    )

    # ── Pressure panel (row 1) ────────────────────────────────────────────────
    Figure.add_trace(go.Scatter(
        x=TimePsArray, y=AlignedData['Instantaneous Pressure (bar)'],
        mode='lines', name='Instantaneous P',
        line=dict(color='lightgrey', width=1),
    ), row=1, col=1)

    Figure.add_trace(go.Scatter(
        x=TimePsArray, y=AlignedData['Cumulative Average (bar)'],
        mode='lines', name='Cumulative Average P',
        line=dict(color='#d62728', width=2, dash='dash'),
    ), row=1, col=1)

    # ── Flux panel (row 2) ────────────────────────────────────────────────────
    Figure.add_trace(go.Scatter(
        x=TimePsArray, y=AlignedData['Instantaneous Flux (m^-2 s^-1)'],
        mode='lines', name='Instantaneous Flux',
        line=dict(color='lightgrey', width=1),
    ), row=2, col=1)

    Figure.add_trace(go.Scatter(
        x=TimePsArray, y=AlignedData['Cumulative Flux (m^-2 s^-1)'],
        mode='lines', name='Cumulative Average Flux',
        line=dict(color='#9467bd', width=2, dash='dash'),
    ), row=2, col=1)

    Figure.update_layout(
        title=f"Convergence: {RunLabel}<br><sup>{VolumeMethod} + {CountingMethod}</sup>",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=1.02, y=0.5),
        margin=dict(r=200),
    )
    Figure.update_yaxes(title_text='Pressure (bar)',  row=1, col=1)
    Figure.update_yaxes(title_text='Flux (m-2 s-1)',  row=2, col=1)
    Figure.update_xaxes(title_text='Time (ps)',        row=2, col=1)

    OutPath = OutputDir / f"Plot_Convergence_{VolumeMethod}_{CountingMethod}_{Temperature}_{RunNumber}.html"
    Figure.write_html(str(OutPath))


def PlotSummaryFigures(
    SummaryTraces:  list,
    VolumeMethod:   str,
    CountingMethod: str,
    SystemName:     str,
    OutputDir:      Path,
) -> None:
    """
    Summary figure overlaying cumulative-average partial pressure for all runs/temperatures.

    SummaryTraces: list of dicts with keys:
        'TimePsArray', 'CumulativePressure', 'Label', 'Color'
    """
    SubTitle   = f"({SystemName}) | Vol: {VolumeMethod} | Count: {CountingMethod}"
    LayoutOpts = dict(
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=1.02, y=0.5),
        margin=dict(r=150),
        xaxis_title='Time (ps)',
    )

    CumulativeFig = go.Figure()

    for Trace in SummaryTraces:
        CumulativeFig.add_trace(go.Scatter(
            x=Trace['TimePsArray'],
            y=Trace['CumulativePressure'],
            mode='lines',
            name=Trace['Label'],
            line=dict(color=Trace['Color'], width=2),
        ))

    CumulativeFig.update_layout(
        title=f"Partial Pressure: Cumulative Average<br><sup>{SubTitle}</sup>",
        yaxis_title='Cumulative Average Pressure (bar)',
        **LayoutOpts,
    )
    CumulativeFig.write_html(OutputDir / f"Plot_CumulativeAvg_{VolumeMethod}_{CountingMethod}.html")


def PlotMethodComparisonPlotly(
    XyzFilePath:             str,
    PostProcessingDirectory: str   = "../postprocessing",
    Stride:                  int   = 400,
    GridResA:                float = 2.0,
) -> None:
    """
    Compare ZrDistance and RayCast volume estimates for a single XYZ file.
    Saves an interactive two-panel HTML figure to the system postprocessing directory.
    """
    TargetFile = Path(XyzFilePath)
    if not TargetFile.exists():
        print(f"File not found: {TargetFile}")
        return

    SystemName = TargetFile.parent.name
    OutputDir  = Path(PostProcessingDirectory) / SystemName
    OutputDir.mkdir(parents=True, exist_ok=True)

    print("ZrDistance method ...")
    DfZr  = EstimateVolumeZrDistance(TargetFile, Stride=Stride)
    print("RayCast method ...")
    DfRay = EstimateVolumeRayCast(TargetFile, Stride=Stride, GridResA=GridResA)

    if DfZr.empty or DfRay.empty:
        print("Error: one or both methods returned an empty DataFrame.")
        return

    DfMerged = pd.merge(DfZr, DfRay, on='Time (fs)', suffixes=('_Zr', '_Ray'))
    DfMerged['Time (ps)']        = DfMerged['Time (fs)'] / 1000.0
    DfMerged['Vol_Zr_nm3']       = DfMerged['Effective Volume (m^3)_Zr']  * 1e27
    DfMerged['Vol_Ray_nm3']      = DfMerged['Effective Volume (m^3)_Ray'] * 1e27
    DfMerged['VolumeDifference'] = DfMerged['Vol_Zr_nm3'] - DfMerged['Vol_Ray_nm3']

    Figure = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"Volume Estimation Comparison: {TargetFile.stem}",
            "Delta Volume (ZrDist - RayCast)",
        ),
    )

    Figure.add_trace(go.Scatter(
        x=DfMerged['Time (ps)'], y=DfMerged['Vol_Zr_nm3'],
        mode='lines', name='Zr-Distance (1D gap)',
        line=dict(color='#1f77b4', width=2),
    ), row=1, col=1)

    Figure.add_trace(go.Scatter(
        x=DfMerged['Time (ps)'], y=DfMerged['Vol_Ray_nm3'],
        mode='lines', name='RayCast (excluded vol.)',
        line=dict(color='#ff7f0e', width=2),
    ), row=1, col=1)

    Figure.add_trace(go.Scatter(
        x=DfMerged['Time (ps)'], y=DfMerged['VolumeDifference'],
        mode='lines', name='Delta Volume',
        line=dict(color='#d62728', width=2),
    ), row=2, col=1)

    MeanDiff = DfMerged['VolumeDifference'].mean()
    Figure.add_hline(
        y=MeanDiff, line_dash='dot', line_color='black',
        annotation_text=f"Mean Delta: {MeanDiff:.2f} nm3",
        annotation_position='bottom right',
        row=2, col=1,
    )

    Figure.update_layout(
        template='plotly_white',
        height=800,
        hovermode='x unified',
        legend=dict(x=1.02, y=0.9),
        margin=dict(r=150),
    )
    Figure.update_yaxes(title_text='Effective Gas Volume (nm3)', row=1, col=1)
    Figure.update_yaxes(title_text='Delta Volume (nm3)',          row=2, col=1)
    Figure.update_xaxes(title_text='Time (ps)',                   row=2, col=1)

    OutputPath = OutputDir / f"Plot_VolumeComparison_{TargetFile.stem}.html"
    Figure.write_html(str(OutputPath))
    print(f"Saved to: {OutputPath}")


# ==========================================
# Main Processing Pipeline
# ==========================================

def _ProcessSingleFile(Args: tuple):
    """
    Top-level worker for ProcessPoolExecutor — must be a module-level function
    to be picklable.

    Processes one XYZ file: reads volume + O2 count (single-pass when
    CountingMethod='XYZFindGases'), aligns timelines, computes pressure and
    flux, saves per-run CSVs and the convergence figure.

    Returns (SummaryTrace dict, SummaryData dict) on success, or None if the
    file could not be processed.
    """
    (XyzPath, DataDirPath, SystemOutDir,
     VolumeMethod, CountingMethod,
     Stride, GridResA,
     Temperature, RunNumber, ColorMap, TaskLabel, ProgressQueue) = Args

    _SafeQueuePut(ProgressQueue, ('start', TaskLabel))

    # ── Volume + count ───────────────────────────────────────────────────────
    if CountingMethod == 'XYZFindGases':
        VolumeData, CountData = ParseXYZSinglePass(
            XyzPath, VolumeMethod, Stride, GridResA,
            ProgressQueue=ProgressQueue, ProgressTask=TaskLabel,
        )
    else:
        if VolumeMethod == 'RayCast':
            VolumeData = EstimateVolumeRayCast(
                XyzPath, Stride=Stride, GridResA=GridResA,
                ProgressQueue=ProgressQueue, ProgressTask=TaskLabel,
            )
        else:
            VolumeData = EstimateVolumeZrDistance(
                XyzPath, Stride=Stride,
                ProgressQueue=ProgressQueue, ProgressTask=TaskLabel,
            )
        CountData = CountMoleculesRateAnalysis(DataDirPath / f"RateAnalysis_{XyzPath.stem}.csv")

    if VolumeData.empty or CountData.empty:
        _SafeQueuePut(ProgressQueue, ('done', TaskLabel))
        return None

    # ── Align timelines ──────────────────────────────────────────────────────
    VolumeData = (VolumeData
                  .drop_duplicates(subset=['Time (fs)'], keep='last')
                  .sort_values('Time (fs)')
                  .reset_index(drop=True))
    CountData  = (CountData
                  .drop_duplicates(subset=['Time (fs)'], keep='last')
                  .sort_values('Time (fs)')
                  .reset_index(drop=True))

    AlignedData = (pd.merge_asof(VolumeData, CountData, on='Time (fs)', direction='nearest')
                   .dropna(subset=['O2 Count'])
                   .reset_index(drop=True))

    VolM3Array = AlignedData['Effective Volume (m^3)'].to_numpy()
    O2Array    = AlignedData['O2 Count'].to_numpy()
    TimeArray  = AlignedData['Time (fs)'].to_numpy()
    N          = len(AlignedData)

    # ── Pressure ─────────────────────────────────────────────────────────────
    InstantPressureBar    = (O2Array * _KB * Temperature / VolM3Array) * 1e-5
    CumulativePressureBar = np.cumsum(InstantPressureBar) / np.arange(1, N + 1)

    # ── Flux ─────────────────────────────────────────────────────────────────
    InstantFlux    = ComputeImpingementFlux(InstantPressureBar,    Temperature)
    CumulativeFlux = ComputeImpingementFlux(CumulativePressureBar, Temperature)

    # ── Gas fraction diagnostic ───────────────────────────────────────────────
    GasFractionArray = AlignedData['Effective Volume (m^3)'] / AlignedData['Cell Volume (m^3)']
    MeanGasFraction  = float(GasFractionArray.mean())
    tqdm.write(
        f"  {Temperature}K Run {RunNumber}: mean gas fraction = {MeanGasFraction:.3f}  "
        f"(eff. vol = {AlignedData['Effective Volume (m^3)'].mean():.3e} m3  "
        f"cell = {AlignedData['Cell Volume (m^3)'].mean():.3e} m3)"
    )

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    AlignedData = AlignedData.copy()
    AlignedData['Gas Fraction']                    = GasFractionArray.values
    AlignedData['Instantaneous Pressure (bar)']    = InstantPressureBar
    AlignedData['Cumulative Average (bar)']        = CumulativePressureBar
    AlignedData['Instantaneous Flux (m^-2 s^-1)'] = InstantFlux
    AlignedData['Cumulative Flux (m^-2 s^-1)']    = CumulativeFlux

    # ── Save CSVs ────────────────────────────────────────────────────────────
    Stem = f"{VolumeMethod}_{CountingMethod}_{Temperature}_{RunNumber}"
    AlignedData.to_csv(SystemOutDir / f"PP_{Stem}.csv", index=False)
    AlignedData[['Time (fs)',
                 'Instantaneous Flux (m^-2 s^-1)',
                 'Cumulative Flux (m^-2 s^-1)']].to_csv(
        SystemOutDir / f"Flux_{Stem}.csv", index=False,
    )

    # ── Per-run convergence figure ────────────────────────────────────────────
    PlotConvergenceFigure(
        AlignedData, Temperature, RunNumber,
        VolumeMethod, CountingMethod, SystemOutDir,
    )

    _SafeQueuePut(ProgressQueue, ('done', TaskLabel))
    return (
        {
            'TimePsArray':        TimeArray / 1000.0,
            'CumulativePressure': CumulativePressureBar,
            'Label': f"{Temperature} K (Run {RunNumber})",
            'Color': ColorMap.get(Temperature, 'black'),
        },
        {
            'Temperature (K)':                   Temperature,
            'Run Number':                        RunNumber,
            'Final Cumulative PP (bar)':         round(float(CumulativePressureBar[-1]),     4),
            'Mean Gas Fraction':                 round(MeanGasFraction,                      4),
            'Final Cumulative Flux (m^-2 s^-1)': round(float(CumulativeFlux[-1]),           4),
        },
    )


def ProcessSystemFolder(
    DataDirectory:           str,
    PostProcessingDirectory: str,
    VolumeMethod:            str   = 'RayCast',
    CountingMethod:          str   = 'RateAnalysis',
    Stride:                  int   = 1,
    GridResA:                float = 1.0,
    MaxWorkers:              int   = None,
) -> None:
    """
    Process all XYZ files in DataDirectory to compute O2 partial pressure
    and Hertz-Knudsen impingement flux time series.

    For each (temperature, run) file the following outputs are written:
      - PP_<vol>_<count>_<T>_<run>.csv          full aligned time series
      - Flux_<vol>_<count>_<T>_<run>.csv        flux columns only
      - Plot_Convergence_...html                 per-run pressure + flux convergence

    Across all files:
      - Plot_CumulativeAvg_...html               summary cumulative-average pressure
      - Summary_...csv                           one row per run

    VolumeMethod  : 'ZrDistance' or 'RayCast'
    CountingMethod: 'RateAnalysis' or 'XYZFindGases'
    Stride        : process every Nth frame
    GridResA      : RayCast grid resolution in Angstrom (ignored for ZrDistance)
    MaxWorkers    : parallel processes (None = half of available logical CPUs)
    """
    DataDirPath  = Path(DataDirectory)
    SystemName   = DataDirPath.name
    SystemOutDir = Path(PostProcessingDirectory) / SystemName
    SystemOutDir.mkdir(parents=True, exist_ok=True)

    if VolumeMethod not in ('ZrDistance', 'RayCast'):
        raise ValueError(f"Unknown VolumeMethod '{VolumeMethod}'. Choose from ['ZrDistance', 'RayCast']")
    if CountingMethod not in ('RateAnalysis', 'XYZFindGases'):
        raise ValueError(f"Unknown CountingMethod '{CountingMethod}'. Choose from ['RateAnalysis', 'XYZFindGases']")

    XyzFiles = sorted(DataDirPath.glob('*.xyz'))
    if not XyzFiles:
        print(f"No XYZ files found in {DataDirPath}.")
        return

    CpuCount = os.cpu_count() or 1
    if MaxWorkers is None:
        # Conservative default to keep the machine responsive on heavy workloads.
        MaxWorkers = max(1, CpuCount // 2)
    else:
        MaxWorkers = max(1, min(int(MaxWorkers), CpuCount))
    print(f"Using {MaxWorkers} worker process(es) out of {CpuCount} logical CPU(s).")

    ColorMap = {873: '#1f77b4', 973: '#2ca02c', 1073: '#ff7f0e', 1273: '#d62728'}

    # Build one argument tuple per XYZ file
    WorkItems = []
    for XyzPath in XyzFiles:
        Match = re.search(r'(\d+)_(\d+)', XyzPath.stem)
        if not Match:
            continue
        Temperature = int(Match.group(1))
        RunNumber   = int(Match.group(2))
        TaskLabel   = f"{Temperature}K Run {RunNumber}"
        WorkItems.append((
            XyzPath, DataDirPath, SystemOutDir,
            VolumeMethod, CountingMethod,
            Stride, GridResA,
            Temperature, RunNumber, ColorMap, TaskLabel,
        ))

    if not WorkItems:
        print(f"No XYZ files in {DataDirPath} matched '<temperature>_<run>.xyz'.")
        return

    TotalFrames = 0
    for XyzPath, *_ in WorkItems:
        TotalFrames += _CountStrideFrames(XyzPath, Stride=Stride)
    print(f"Tracking progress across {TotalFrames} frame(s) in {len(WorkItems)} file(s).")

    SummaryTraces = []
    SummaryData   = []

    with Manager() as MpManager:
        ProgressQueue = MpManager.Queue()
        WorkItemsWithProgress = [Item + (ProgressQueue,) for Item in WorkItems]

        with concurrent.futures.ProcessPoolExecutor(max_workers=MaxWorkers) as Pool:
            Futures = {
                Pool.submit(_ProcessSingleFile, Item): Item
                for Item in WorkItemsWithProgress
            }
            Pending       = set(Futures.keys())
            ActiveTasks   = set()
            CompletedRuns = 0

            with tqdm(
                total=TotalFrames if TotalFrames > 0 else None,
                desc=f"Processing {SystemName} ({VolumeMethod} + {CountingMethod})",
                unit='frame',
            ) as FrameBar:
                while Pending:
                    Done, Pending = concurrent.futures.wait(
                        Pending,
                        timeout=0.5,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    _DrainProgressQueue(
                        ProgressQueue, FrameBar, ActiveTasks,
                        CompletedRuns, len(Futures),
                    )

                    for Future in Done:
                        Item = Futures[Future]
                        TaskLabel = Item[-2]
                        ActiveTasks.discard(TaskLabel)

                        Result = Future.result()
                        CompletedRuns += 1
                        FrameBar.set_postfix_str(
                            f"files {CompletedRuns}/{len(Futures)} | active {len(ActiveTasks)}"
                        )

                        if Result is None:
                            continue
                        SummaryTrace, SummaryRow = Result
                        SummaryTraces.append(SummaryTrace)
                        SummaryData.append(SummaryRow)

                _DrainProgressQueue(
                    ProgressQueue, FrameBar, ActiveTasks,
                    CompletedRuns, len(Futures),
                )

    # ── Summary figures and table ────────────────────────────────────────────
    if SummaryTraces:
        PlotSummaryFigures(SummaryTraces, VolumeMethod, CountingMethod, SystemName, SystemOutDir)

    if SummaryData:
        (pd.DataFrame(SummaryData)
           .sort_values(['Temperature (K)', 'Run Number'])
           .to_csv(SystemOutDir / f"Summary_{VolumeMethod}_{CountingMethod}.csv", index=False))
        print(f"\nDone. Outputs in: {SystemOutDir}")


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    ProcessSystemFolder(
        DataDirectory           = "../../DevArea/Data/ZrC",
        PostProcessingDirectory = "../postprocessing",
        VolumeMethod            = 'RayCast',
        CountingMethod          = 'RateAnalysis',
        Stride                  = 1,
        GridResA                = 1.0,
    )

    # Uncomment to compare volume methods side-by-side for a single file:
    # PlotMethodComparisonPlotly("../../DevArea/Data/ZrC/1073_1.xyz", Stride=400, GridResA=2.0)
