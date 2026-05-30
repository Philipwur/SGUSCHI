# File for preparing POSCAR files for simulations
#%% Imports 

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional  # Added for Python 3.9 compatibility

sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio
from workflow import OxidationAnalysis as an






def AddVacuum(Position, CellDim, GasRatio, Axis='x'):
    '''
    Function which adds vacuum to the CellDim along Axis by GasRatio.
    Scales fractional positions accordingly (keeps Cartesian positions unchanged).

    Args:
        Position : pd.DataFrame with columns ['Element','x','y','z'] (fractional)
        CellDim  : pd.DataFrame 3x3 with columns ['x','y','z'] (cartesian lattice vectors)
        GasRatio : float, new_length = old_length * (1 + GasRatio)
                   e.g. GasRatio=2 -> triple the length along Axis
        Axis     : one of 'x','y','z' (maps to a, b, c lattice vectors respectively)

    Returns:
        Position_new, CellDim_new
    '''
    if Axis not in ('x', 'y', 'z'):
        raise ValueError("Axis must be one of 'x', 'y', 'z'.")

    try:
        scale = 1.0 + float(GasRatio)
    except Exception as e:
        raise ValueError(f"GasRatio must be numeric: {e}") from e

    if scale <= 0.0:
        raise ValueError("GasRatio must be > -1.0 so the scaled lattice length stays positive.")

    # Map fractional axis -> lattice vector row index
    axis_to_row = {'x': 0, 'y': 1, 'z': 2}
    row = axis_to_row[Axis]

    # --- scale lattice vector along Axis ---
    CellDim_new = CellDim.copy()
    CellDim_new.iloc[row, :] = CellDim_new.iloc[row, :].values * scale

    # --- shift origin so PBC boundary (x=0) falls in the largest inter-layer gap ---
    # This prevents a layer straddling x=0 from being split across both slab surfaces.
    # NOTE: This heuristic assumes the input is a bulk cubic structure (all inter-atom
    # gaps are roughly equal). If non-cubic or slab structures are ever supported, the
    # "largest gap" will be the pre-existing vacuum, not an inter-layer gap, and this
    # logic will need to be revised (e.g. use a user-supplied shift, or detect the
    # vacuum vs. inter-layer gap distinction explicitly).
    Position_new = Position.copy()
    Coords = Position_new[Axis].to_numpy() % 1.0
    Sorted = np.sort(Coords)
    Gaps = np.diff(Sorted)
    WrapGap = 1.0 - Sorted[-1] + Sorted[0]
    AllGaps = np.append(Gaps, WrapGap)
    MaxIdx = int(np.argmax(AllGaps))

    if MaxIdx < len(Gaps):
        GapCenter = (Sorted[MaxIdx] + Sorted[MaxIdx + 1]) / 2.0
    else:
        GapCenter = ((Sorted[-1] + Sorted[0] + 1.0) / 2.0) % 1.0

    Position_new[Axis] = (Position_new[Axis] - GapCenter) % 1.0

    # --- rescale fractional coordinates along Axis to preserve Cartesian positions ---
    Position_new[Axis] = Position_new[Axis] / scale

    # Keep fractional coordinates in [0,1)
    Position_new[Axis] = Position_new[Axis] % 1.0

    return Position_new, CellDim_new


def PreparePOSCAR(Position, CellDim, GasRatio = 2, InitO2 = 10):
    '''
    Function which prepares the POSCAR for oxidation simulations.
    Expands the x-dimension by GasRatio and scales fractional positions.
    Also places InitO2 O2 molecules in optimal positions.
    Does not do supercell expansion, seperate helper functin can be used for that.
    
    '''
    
    #First expand the x-dimension by GasRatio and scale fractional positions
     
    Position, CellDim = AddVacuum(Position, CellDim, GasRatio = GasRatio)
    NewSites = an.FindOptimalCoords(Position, CellDim, N = InitO2)
    Position = an.PlaceO2Molecules(Positions = Position,
                                  CellDim = CellDim,
                                  NewSites = NewSites)
    
    return Position, CellDim

# %%
