import re
import sys
import os
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

# Resolve the workflow package relative to this file so the script runs from any
# working directory (it is launched from inside the data folder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
import OxidationAnalysis as an
from pymatgen.optimization.neighbors import find_points_in_spheres

# Gas volume is estimated with the Zr-distance gap method; the tag labels plots.
VOLUME_TAG = "ZrDistance"

# pymatgen's find_points_in_spheres wants `pbc` as C-long; np.int_ matches that
# on every platform (int32 on Windows, int64 on Linux).
_PBC = np.array([1, 1, 1], dtype=np.int_)


@dataclass
class RunTask:
    """One (temperature, run) postprocessing job. Picklable for ProcessPoolExecutor."""
    XyzPath:     Path
    OutputDir:   Path
    Temperature: int
    RunNumber:   int
    Stride:      int
    Color:       str         # resolved from ColorMap up front
    Label:       str


# ==========================================
# Physical Constants
# ==========================================

_O2_MASS_KG = 32.0 * 1.66054e-27     # kg
_KB          = 1.380649e-23           # J K⁻¹


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
    InvCell:   np.ndarray,
    Elements:  list,
    CartArray: np.ndarray,
) -> tuple:
    """
    ZrDistance volume for a single already-parsed frame.

    Returns (EffectiveVolumeM3, CellVolumeM3).
    CellMat  : (3,3) row-vector lattice in Angstrom.
    InvCell  : precomputed inverse of CellMat (shared with the O2 counter).
    CartArray: (N,3) Cartesian coordinates in Angstrom.
    """
    CellVolA3 = abs(np.dot(CellMat[0], np.cross(CellMat[1], CellMat[2])))

    ZrMask = np.array([e == 'Zr' for e in Elements])
    if np.any(ZrMask):
        FracX       = np.sort((CartArray[ZrMask] @ InvCell)[:, 0] % 1.0)
        Diffs       = np.diff(FracX)
        WrapGap     = 1.0 - FracX[-1] + FracX[0]
        GasFraction = max(WrapGap, np.max(Diffs)) if Diffs.size > 0 else WrapGap
    else:
        GasFraction = 1.0

    return CellVolA3 * GasFraction * 1e-30, CellVolA3 * 1e-30


def _IterXyzFrames(XyzFilePath: Path, Stride: int = 1):
    """
    Shared streaming reader for extended-XYZ trajectories. Yields one frame
    every `Stride` as a tuple (TimeFs, Lattice, Elements, CartArray):

        TimeFs    : float, cumulative time from the `Time=`/`t=` comment tag
                    (falls back to the first number in the comment, else 0.0).
        Lattice   : (3,3) row-vector lattice in Angstrom (Cartesian = fractional @ Lattice).
        Elements  : list[str] of element symbols.
        CartArray : (N,3) Cartesian coordinates in Angstrom.

    Frames whose comment line lacks a valid 9-element `Lattice="..."`, or that
    contain no parseable atoms, are skipped. All callers (volume estimation,
    O2 counting, single-pass) share this one parser.
    """
    LatticeRegex = re.compile(r'Lattice="([^"]+)"')
    TimeRegex    = re.compile(r'(?:Time|t)\s*[:=]\s*([\d\.]+)', re.IGNORECASE)

    FrameIndex = 0
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

            AtomLines = [File.readline() for _ in range(NumAtoms)]
            FrameIndex += 1

            LatticeMatch = LatticeRegex.search(CommentLine)
            if not LatticeMatch:
                continue
            Nums = [float(X) for X in LatticeMatch.group(1).split()]
            if len(Nums) != 9:
                continue
            Lattice = np.array(Nums).reshape(3, 3)

            TimeMatch = TimeRegex.search(CommentLine)
            if TimeMatch:
                TimeFs = float(TimeMatch.group(1))
            else:
                AnyMatch = re.search(r'([\d\.]+)', CommentLine)
                TimeFs   = float(AnyMatch.group(1)) if AnyMatch else 0.0

            Elements, CartCoords = [], []
            for Line in AtomLines:
                Parts = Line.split()
                if len(Parts) < 4:
                    continue
                Elements.append(Parts[0])
                CartCoords.append([float(Parts[1]), float(Parts[2]), float(Parts[3])])

            if not Elements:
                continue

            yield TimeFs, Lattice, Elements, np.array(CartCoords)


# ==========================================
# Molecule Counting
# ==========================================

def _FrameO2Count(
    Lattice:         np.ndarray,
    Elements:        list,
    CartArray:       np.ndarray,
    CovalentRadii:   dict,
    AtomicRadiusTol: float,
):
    """
    O2 molecule count for a single already-parsed frame.

    Uses pymatgen's `find_points_in_spheres` — a fast, general-cell (triclinic-
    correct) PBC neighbour search — to get candidate close pairs, filters them by
    the per-pair covalent cutoff, then reuses the exact bonding rules from
    OxidationAnalysis:
      - Each O atom participates in at most one O-O bond (EnforceUniqueOOBonds),
        preventing fictitious O3/O4 clusters in the high-density gas phase.
      - Any O bonded to Zr joins the large slab connected component and is
        excluded by counting only size-2 ('O','O') components (== FindGases with
        MinimumComplexity = MaximumComplexity = 2).

    Returns the O2 count (int), or None if the frame contains elements absent
    from CovalentRadii.
    """
    if set(Elements) - set(CovalentRadii):
        return None

    Elements = np.asarray(Elements)
    N        = len(Elements)
    Radii    = np.array([CovalentRadii[e] for e in Elements], dtype=float)
    MaxCut   = 2.0 * float(Radii.max()) * AtomicRadiusTol  # upper bound over all pairs

    Cart = np.ascontiguousarray(CartArray, dtype=float)
    Lat  = np.ascontiguousarray(Lattice, dtype=float)
    I1, I2, _Off, Dist = find_points_in_spheres(Cart, Cart, MaxCut, _PBC, Lat)

    # Distinct atom pairs within their specific covalent cutoff.
    Keep = I1 != I2
    Ii, Jj, Dd = I1[Keep], I2[Keep], Dist[Keep]
    Bonded = Dd < (Radii[Ii] + Radii[Jj]) * AtomicRadiusTol
    Bi, Bj, Bd = Ii[Bonded], Jj[Bonded], Dd[Bonded]

    BondMatrix = np.zeros((N, N), dtype=bool)
    BondMatrix[Bi, Bj] = True                  # both directions present in the pair list
    DistMatrix = np.full((N, N), np.inf, dtype=float)
    np.minimum.at(DistMatrix, (Bi, Bj), Bd)    # min over multiple periodic images

    BondMatrix = an.EnforceUniqueOOBonds(BondMatrix, Elements, DistMatrix)
    Components = an.FindConnectedSubcomponents(BondMatrix)
    return sum(
        1 for Comp in Components
        if len(Comp) == 2 and tuple(sorted(Elements[k] for k in Comp)) == ('O', 'O')
    )


def ParseTrajectory(XyzFilePath: Path, Stride: int = 1) -> tuple:
    """
    Parse an XYZ trajectory once, computing per-frame gas volume and O2 count.

    For each frame (every `Stride`) this records the gap-based effective gas
    volume (via _FrameVolumeZrDistance), the full cell volume, the
    cross-sectional area perpendicular to the a-axis (|b x c| — the surface the
    gas impinges on), and the O2 molecule count (via pymatgen neighbour search).
    Volume and count therefore share an identical time base, so no timeline
    alignment is needed downstream.

    Returns (FrameData, Warnings):
        FrameData : DataFrame with columns
            ['Time (fs)', 'O2 Count', 'Effective Volume (m^3)',
             'Cell Volume (m^3)', 'Cross-sectional Area (m^2)'].
        Warnings  : list of warning strings for the parent process to emit
                    (workers have no progress bar of their own).
    """
    CovalentRadii   = {'C': 0.77, 'Zr': 1.45, 'O': 0.66, 'N': 0.71}
    AtomicRadiusTol = 1.5

    Records  = []
    Warnings = []
    for TimeFs, Lattice, Elements, CartArray in _IterXyzFrames(XyzFilePath, Stride):
        try:
            InvLattice = np.linalg.inv(Lattice)
        except np.linalg.LinAlgError:
            InvLattice = np.eye(3)

        EffVolM3, CellVolM3 = _FrameVolumeZrDistance(Lattice, InvLattice, Elements, CartArray)
        AreaM2 = np.linalg.norm(np.cross(Lattice[1], Lattice[2])) * 1e-20

        O2Count = _FrameO2Count(Lattice, Elements, CartArray, CovalentRadii, AtomicRadiusTol)
        if O2Count is None:
            Warnings.append(f"  [WARNING] Unknown elements {set(Elements) - set(CovalentRadii)} "
                            f"at {TimeFs} fs — skipping O2 count.")
            O2Count = 0

        Records.append({
            'Time (fs)':                  TimeFs,
            'O2 Count':                   O2Count,
            'Effective Volume (m^3)':     EffVolM3,
            'Cell Volume (m^3)':          CellVolM3,
            'Cross-sectional Area (m^2)': AreaM2,
        })

    return pd.DataFrame(Records), Warnings


# ==========================================
# Figures
# ==========================================

def PlotConvergenceFigure(
    AlignedData: pd.DataFrame,
    Temperature: int,
    RunNumber:   int,
    OutputDir:   Path,
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

    # Row 1 = pressure panel, row 2 = flux panel.
    # (row, column, legend name, color, width, dash)
    TraceSpecs = [
        (1, 'Instantaneous Pressure (bar)',   'Instantaneous P',         'lightgrey', 1, None),
        (1, 'Cumulative Average (bar)',       'Cumulative Average P',    '#d62728',   2, 'dash'),
        (2, 'Instantaneous Flux (m^-2 s^-1)', 'Instantaneous Flux',      'lightgrey', 1, None),
        (2, 'Cumulative Flux (m^-2 s^-1)',    'Cumulative Average Flux', '#9467bd',   2, 'dash'),
    ]
    for Row, Column, Name, LineColor, Width, Dash in TraceSpecs:
        Figure.add_trace(go.Scatter(
            x=TimePsArray, y=AlignedData[Column],
            mode='lines', name=Name,
            line=dict(color=LineColor, width=Width, dash=Dash),
        ), row=Row, col=1)

    Figure.update_layout(
        title=f"Convergence: {RunLabel}<br><sup>{VOLUME_TAG}</sup>",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=1.02, y=0.5),
        margin=dict(r=200),
    )
    Figure.update_yaxes(title_text='Pressure (bar)',  row=1, col=1)
    Figure.update_yaxes(title_text='Flux (m-2 s-1)',  row=2, col=1)
    Figure.update_xaxes(title_text='Time (ps)',        row=2, col=1)

    OutPath = OutputDir / f"Plot_Convergence_{Temperature}_{RunNumber}.html"
    Figure.write_html(str(OutPath))


def PlotSummaryFigures(
    SummaryTraces: list,
    SystemName:    str,
    OutputDir:     Path,
) -> None:
    """
    Summary figure overlaying cumulative-average partial pressure for all runs/temperatures.

    SummaryTraces: list of dicts with keys:
        'TimePsArray', 'CumulativePressure', 'Label', 'Color'
    """
    SubTitle   = f"({SystemName}) | Vol: {VOLUME_TAG}"
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
    CumulativeFig.write_html(OutputDir / "Plot_CumulativeAvg.html")


# ==========================================
# Main Processing Pipeline
# ==========================================

def _ProcessSingleFile(Task: RunTask):
    """
    Top-level worker for ProcessPoolExecutor — must be a module-level function
    to be picklable.

    Parses one XYZ trajectory (volume + O2 count in a single pass), computes
    pressure and flux, writes the per-run PP_<T>_<run>.csv with all quantities,
    and renders the convergence figure.

    Returns (Result, Messages), where Result is (SummaryTrace dict, SummaryRow
    dict) on success or None if the file could not be processed, and Messages is
    a list of strings for the parent to print via the progress bar (workers run
    in child processes with no bar of their own).
    """
    Temperature, RunNumber = Task.Temperature, Task.RunNumber

    Data, Messages = ParseTrajectory(Task.XyzPath, Task.Stride)
    if Data.empty:
        return None, Messages

    # Frames stream out in time order; drop_duplicates(keep='last') guards against
    # overlapping timestamps at cycle boundaries, so no explicit sort is needed.
    Data = (Data
            .drop_duplicates(subset=['Time (fs)'], keep='last')
            .reset_index(drop=True))

    # ── Gas fraction (+ diagnostic) ───────────────────────────────────────────
    Data['Gas Fraction'] = Data['Effective Volume (m^3)'] / Data['Cell Volume (m^3)']
    MeanGasFraction = float(Data['Gas Fraction'].mean())
    Messages.append(
        f"  {Temperature}K Run {RunNumber}: mean gas fraction = {MeanGasFraction:.3f}  "
        f"(eff. vol = {Data['Effective Volume (m^3)'].mean():.3e} m3  "
        f"cell = {Data['Cell Volume (m^3)'].mean():.3e} m3)"
    )

    # ── Pressure + flux (instantaneous and cumulative average) ────────────────
    Data['Instantaneous Pressure (bar)'] = (
        Data['O2 Count'] * _KB * Temperature / Data['Effective Volume (m^3)'] * 1e-5
    )
    Data['Cumulative Average (bar)']       = Data['Instantaneous Pressure (bar)'].expanding().mean()
    Data['Instantaneous Flux (m^-2 s^-1)'] = ComputeImpingementFlux(Data['Instantaneous Pressure (bar)'], Temperature)
    Data['Cumulative Flux (m^-2 s^-1)']    = ComputeImpingementFlux(Data['Cumulative Average (bar)'], Temperature)

    Data.to_csv(Task.OutputDir / f"PP_{Temperature}_{RunNumber}.csv", index=False)

    # ── Per-run convergence figure ────────────────────────────────────────────
    PlotConvergenceFigure(Data, Temperature, RunNumber, Task.OutputDir)

    return (
        {
            'TimePsArray':        Data['Time (fs)'].to_numpy() / 1000.0,
            'CumulativePressure': Data['Cumulative Average (bar)'].to_numpy(),
            'Label': f"{Temperature} K (Run {RunNumber})",
            'Color': Task.Color,
        },
        {
            'Temperature (K)':                   Temperature,
            'Run Number':                        RunNumber,
            'Final Cumulative PP (bar)':         round(float(Data['Cumulative Average (bar)'].iloc[-1]),     4),
            'Mean Gas Fraction':                 round(MeanGasFraction,                                      4),
            'Final Cumulative Flux (m^-2 s^-1)': round(float(Data['Cumulative Flux (m^-2 s^-1)'].iloc[-1]),  4),
        },
    ), Messages


def ProcessSystemFolder(
    DataDirectory: str,
    Stride:        int = 1,
    MaxWorkers:    int = None,
    OutputDirName: str = 'partial_pressure',
) -> None:
    """
    Process all XYZ files in DataDirectory to compute O2 partial pressure
    and Hertz-Knudsen impingement flux time series.

    DataDirectory is the flat folder (e.g. an `xyz_files` directory) holding
    `{T}_{run}.xyz` trajectories. All outputs are written to a `OutputDirName`
    subfolder created inside it. Gas volume is estimated with the Zr-distance gap
    method and O2 counts come from the trajectory directly (FindGases).

    For each (temperature, run) a single CSV with all per-frame quantities is
    written:
      - PP_<T>_<run>.csv            time, O2 count, volumes, area, gas fraction,
                                    pressure (instant + cumulative), flux
      - Plot_Convergence_<T>_<run>.html   per-run pressure + flux convergence

    Across all runs:
      - Plot_CumulativeAvg.html     summary cumulative-average pressure
      - Summary.csv                 one row per run

    Stride        : process every Nth frame
    MaxWorkers    : parallel processes (None = half of available logical CPUs)
    OutputDirName : name of the output subfolder created inside DataDirectory
    """
    DataDirPath = Path(DataDirectory).resolve()
    SystemName  = DataDirPath.name
    if SystemName == 'xyz_files':
        SystemName = DataDirPath.parent.name   # titles read e.g. "ZrC", not "xyz_files"
    SystemOutDir = DataDirPath / OutputDirName
    SystemOutDir.mkdir(parents=True, exist_ok=True)

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

    # Build one RunTask per XYZ file, resolving the color up front.
    Tasks = []
    for XyzPath in XyzFiles:
        Match = re.search(r'(\d+)_(\d+)', XyzPath.stem)
        if not Match:
            continue
        Temperature = int(Match.group(1))
        RunNumber   = int(Match.group(2))
        Tasks.append(RunTask(
            XyzPath=XyzPath,
            OutputDir=SystemOutDir,
            Temperature=Temperature,
            RunNumber=RunNumber,
            Stride=Stride,
            Color=ColorMap.get(Temperature, 'black'),
            Label=f"{Temperature}K Run {RunNumber}",
        ))

    if not Tasks:
        print(f"No XYZ files in {DataDirPath} matched '<temperature>_<run>.xyz'.")
        return
    print(f"Processing {len(Tasks)} file(s).")

    SummaryTraces = []
    SummaryData   = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MaxWorkers) as Pool:
        Futures = {Pool.submit(_ProcessSingleFile, Task): Task for Task in Tasks}
        with tqdm(
            total=len(Tasks),
            desc=f"Processing {SystemName} ({VOLUME_TAG})",
            unit='file',
        ) as Bar:
            for Future in concurrent.futures.as_completed(Futures):
                Result, Messages = Future.result()
                Bar.update(1)
                for Msg in Messages:
                    Bar.write(Msg)
                if Result is None:
                    continue
                SummaryTrace, SummaryRow = Result
                SummaryTraces.append(SummaryTrace)
                SummaryData.append(SummaryRow)

    # ── Summary figures and table ────────────────────────────────────────────
    if SummaryTraces:
        PlotSummaryFigures(SummaryTraces, SystemName, SystemOutDir)

    if SummaryData:
        (pd.DataFrame(SummaryData)
           .sort_values(['Temperature (K)', 'Run Number'])
           .to_csv(SystemOutDir / "Summary.csv", index=False))
        print(f"\nDone. Outputs in: {SystemOutDir}")


# ==========================================
# Entry Point
# ==========================================

def main(argv=None):
    Parser = argparse.ArgumentParser(
        description="Compute O2 partial pressure + impingement flux from XYZ trajectories.")
    Parser.add_argument('folder', nargs='?', default='.',
        help="Folder with {T}_{run}.xyz trajectories (default: current dir).")
    Parser.add_argument('--stride', type=int, default=1, help="Process every Nth frame (default: 1).")
    Parser.add_argument('--workers', type=int, default=None,
        help="Parallel processes (default: half of logical CPUs).")
    Parser.add_argument('--output-name', dest='output_name', default='partial_pressure',
        help="Name of the output subfolder created inside FOLDER (default: partial_pressure).")
    Args = Parser.parse_args(argv)
    ProcessSystemFolder(
        DataDirectory=Args.folder,
        Stride=Args.stride,
        MaxWorkers=Args.workers,
        OutputDirName=Args.output_name,
    )


if __name__ == "__main__":
    main()
