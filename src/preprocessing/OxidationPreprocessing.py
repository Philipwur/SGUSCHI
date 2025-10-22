#%% Imports 

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from workflow import VaspIO as vio
from workflow import OxidationAnalysis as an


def SwapAxes(Positions: pd.DataFrame, CellDim: pd.DataFrame, Axes=('x', 'z')):
    '''
    Helper function for swapping axes in Positions (fractional coordinates)
    and CellDim (Cartesian lattice vectors).

    Args:
        Positions : pd.DataFrame
            Must include columns ['Element', 'x', 'y', 'z'] (fractional coords).
        CellDim : pd.DataFrame
            Must include columns ['x', 'y', 'z'] (Cartesian lattice vectors, rows = a1,a2,a3).
        Axes : tuple[str, str]
            Pair of axes to swap, e.g. ('x','z') or ('y','z').

    Returns:
        (Positions_swapped, CellDim_swapped)
    '''
    Axes = tuple(Axes)
    if len(Axes) != 2:
        raise ValueError("Axes must be a tuple of two elements, e.g. ('x','z').")

    valid_axes = ['x', 'y', 'z']
    if any(ax not in valid_axes for ax in Axes):
        raise ValueError("Axes must be from ['x','y','z'].")

    # Create a mapping for axis reordering
    order = valid_axes.copy()
    i, j = order.index(Axes[0]), order.index(Axes[1])
    order[i], order[j] = order[j], order[i]

    # --- Swap CellDim rows ---
    CellDim_swapped = CellDim.loc[:, order].copy()
    CellDim_swapped = CellDim_swapped.reindex([valid_axes.index(ax) for ax in order]).reset_index(drop=True)
    CellDim_swapped.columns = ['x','y','z']

    # --- Swap Positions fractional coordinates ---
    frac = Positions[['x','y','z']].values
    frac_swapped = frac[:, [valid_axes.index(ax) for ax in order]]

    Positions_swapped = Positions.copy()
    Positions_swapped[['x','y','z']] = frac_swapped

    return Positions_swapped, CellDim_swapped


def CreateSupercell(Position, CellDim, SupercellMatrix):
    '''
    Function which creates a supercell based on the SupercellMatrix.
    Not implmeneted right now.
    '''
    
    return Position, CellDim


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

    # --- rescale fractional coordinates along Axis to preserve Cartesian positions ---
    Position_new = Position.copy()
    Position_new[Axis] = Position_new[Axis] / scale

    # Keep fractional coordinates in [0,1) (harmless for positive scale; useful if shrinking)
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
    NewSites = an.FindOptimalCoords(Position, CellDim, n = InitO2)
    Position = an.PlaceO2Molecules(Position = Position,
                                  CellDim = CellDim,
                                  NewSites = NewSites)
    
    return Position, CellDim

# %% Demos and useful fileprep

if __name__ == "__main__":
    
    Position, CellDim = vio.ReadPOSCAR(workdir = '../../Test/',
                                       filename ='POSCAR_ZrCN_NoGas')
    
    Position, CellDim = SwapAxes(Position, CellDim, Axes = ('x','z'))
    
    vio.WritePOSCAR(WorkDir = '../../Test/',
                    Position = Position,
                    CellDim = CellDim,
                    FileName = 'POSCAR_ZrCN_Swapped')
    
    
    Position, CellDim = AddVacuum(Position, CellDim, GasRatio = 2, 
                                  Axis = 'x')
    
    vio.WritePOSCAR(WorkDir = '../../Test/',
                    Position = Position,
                    CellDim = CellDim,
                    FileName = 'POSCAR_ZrCN_Vacuum')
    
    NewSites = an.FindOptimalCoords(Position, CellDim, n = 10)
    
    Position = an.PlaceO2Molecules(Position = Position,
                                  CellDim = CellDim,
                                  NewSites = NewSites)

    vio.WritePOSCAR(WorkDir = '../../Test/',
                    Position = Position,
                    CellDim = CellDim,
                    FileName = 'POSCAR_ZrCN_O2Placed')
# %%
