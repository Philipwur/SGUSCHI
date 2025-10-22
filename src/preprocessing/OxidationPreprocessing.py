#%% Imports 

import pandas as pd
import numpy as np
import sys
import os

sys.path.append('..')

from workflow import VaspIO as vio


#from ..workflow import OxidationAnalysis as an

#from scipy.optimize import differential_evolution
#from pymatgen.core import Structure, Lattice
#from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

#Give this script a POSCAR file, and it will turn it into an oxidation ready
#struct. Messy for now.


#This bit is for PoscarPrep and future oxygen replacement and optional function
#Needs a bit of reformatting and better documentation
#Move these to PreProcessing. no longer needed here
#------------------------------------------------------------------------------

'''



def SwapAxes(Positions, CellDim, Axes):

    Swap two axes in the Positions and CellDim DataFrames.

    
    Positions_swapped = Positions.copy()
    Positions_swapped[Axes[0], Axes[1]] = Positions[Axes[1]], Positions[Axes[0]]
    
    CellDim_swapped = CellDim.copy()
    CellDim_swapped[Axes[0], Axes[1]] = CellDim[Axes[1]], CellDim[Axes[0]]
    
    return Positions_swapped, CellDim_swapped

def FindOptimalPositions(Position, CellDim, n, maxiter=1000, popsize=15, seed=None):
    cell_matrix = CellDim.to_numpy()
    existing_frac = Position[['x','y','z']].to_numpy()

    def wrap_frac(diff):
        return diff - np.round(diff)

    def objective(x):
        new_frac = x.reshape((n, 3))
        
        # Compute all distances (Vectorized)
        wrapped_diffs = wrap_frac(new_frac[:, None, :] - existing_frac[None, :, :])  
        cart_diffs = np.einsum('ijk,kl->ijl', wrapped_diffs, cell_matrix)  
        dists_new_existing = np.linalg.norm(cart_diffs, axis=2)  

        if n > 1:
            wrapped_diffs_self = wrap_frac(new_frac[:, None, :] - new_frac[None, :, :])  
            cart_diffs_self = np.einsum('ijk,kl->ijl', wrapped_diffs_self, cell_matrix)  
            dists_new_new = np.linalg.norm(cart_diffs_self, axis=2)  
            dists_new_new = dists_new_new[np.triu_indices(n, k=1)]  
        else:
            dists_new_new = np.array([])

        min_dist = np.min(np.concatenate([dists_new_existing.ravel(), dists_new_new])) if n > 1 else np.min(dists_new_existing)
        return -min_dist  

    bounds = [(0,1)] * (3 * n)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=max(5, popsize),  
        tol=1e-5,  
        seed=seed
    )

    best_x = result.x.reshape((n, 3))
    return pd.DataFrame(best_x, columns = ['x','y','z'])


def PlaceOAtoms(new_sites_df, m, CellDim_df, bond_length = 1.2):
    """
    Place O atoms on proposed sites within a periodic cell.

    Parameters
    ----------
    new_sites_df : pd.DataFrame
        DataFrame with columns ['x', 'y', 'z'] giving the fractional coordinates of proposed sites.
    m : int
        Total number of O atoms to place. Must satisfy m <= 2 * number of sites.
    CellDim_df : pd.DataFrame
        3x3 DataFrame whose rows are the cell vectors in Cartesian space.
        Columns should be ['x', 'y', 'z'].
    bond_length : float, optional
        Desired bond length between paired O atoms in Cartesian coordinates. Default is 1.2 Å.

    Returns
    -------
    new_O_df : pd.DataFrame
        DataFrame with columns ['Element', 'x', 'y', 'z'] containing fractional coordinates of placed O atoms.
    """
    
    # Number of proposed sites
    n = len(new_sites_df)
    
    # Validation
    if m > 2 * n:
        raise ValueError(f"Number of atoms to place (m={m}) exceeds twice the number of sites (2*{n}= {2*n}).")
    if m < 0:
        raise ValueError("Number of atoms to place (m) cannot be negative.")
    
    # Determine number of O2 and O
    num_O2 = m // 2  # Number of sites to host O2
    num_O = m % 2    # Number of sites to host O
    
    # Assign O2 to the first num_O2 sites and O to the next num_O sites
    O2_sites = new_sites_df.iloc[:num_O2].reset_index(drop=True)
    O_sites = new_sites_df.iloc[num_O2:num_O2 + num_O].reset_index(drop=True)
    
    # Initialize list to collect new O atoms
    new_O_atoms = []
    
    # Convert cell matrix to numpy array
    cell_matrix = CellDim_df.to_numpy()  # Shape: (3,3)
    
    # Compute inverse of cell matrix for Cartesian to fractional conversion
    try:
        M_inv = np.linalg.inv(cell_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("CellDim_df matrix is singular and cannot be inverted.")
    
    # Helper functions
    def fractional_to_cartesian(fractional, M):
        """Convert fractional to Cartesian coordinates."""
        return np.dot(fractional, M)
    
    def cartesian_to_fractional(cartesian, M_inv):
        """Convert Cartesian to fractional coordinates."""
        return np.dot(cartesian, M_inv)
    
    def wrap_fractional(fractional):
        """Wrap fractional coordinates into [0, 1)."""
        return fractional % 1.0
    
    def find_primary_direction(M):
        """
        Determine the primary direction based on the cell vectors.
        Chooses the cell vector with the largest magnitude.
        Returns a unit vector in that direction.
        """
        a = M[0]
        b = M[1]
        c = M[2]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        norm_c = np.linalg.norm(c)
        
        if norm_a >= norm_b and norm_a >= norm_c:
            direction = np.array([1, 0, 0])
        elif norm_b >= norm_a and norm_b >= norm_c:
            direction = np.array([0, 1, 0])
        else:
            direction = np.array([0, 0, 1])
        
        return direction / np.linalg.norm(direction)  # Unit vector
    
    # Determine primary direction for bond placement
    primary_direction = find_primary_direction(cell_matrix)
    
    # Place O2 atoms
    for idx, site in O2_sites.iterrows():
        site_frac = site[['x', 'y', 'z']].values
        site_cart = fractional_to_cartesian(site_frac, cell_matrix)
        
        # First O atom at the site
        new_O_atoms.append({'Element': 'O', 'x': site_frac[0], 'y': site_frac[1], 'z': site_frac[2]})
        
        # Second O atom at bond_length along primary direction
        second_cart = site_cart + primary_direction * bond_length
        
        # Convert back to fractional coordinates
        second_frac = cartesian_to_fractional(second_cart, M_inv)
        
        # Wrap into the unit cell
        second_frac_wrapped = wrap_fractional(second_frac)
        
        # Append second O atom
        new_O_atoms.append({'Element': 'O', 'x': second_frac_wrapped[0], 'y': second_frac_wrapped[1], 'z': second_frac_wrapped[2]})
    
    # Place O atoms
    for idx, site in O_sites.iterrows():
        site_frac = site[['x', 'y', 'z']].values
        new_O_atoms.append({'Element': 'O', 'x': site_frac[0], 'y': site_frac[1], 'z': site_frac[2]})
    
    # Convert list to DataFrame
    new_O_df = pd.DataFrame(new_O_atoms)
    
    return new_O_df


def AddO2(AtomInfo, O2AtomsRemoved, Position, CellDim, Velocities = None):
    #purpose of this function, checks if number of oxygens is even or odd
    #if number is even and one (or more) O2 atoms have been removed, find best 
    #spot to add O2 atom
    
    #if first step has 3 moved this function doesnt work atm, need better solution

    #first decide the amount of positions needed to find.
    NumberOfSites = O2AtomsRemoved / 2
    if not NumberOfSites.is_integer():
        NumberOfSites += 0.5
    NumberOfSites = round(NumberOfSites)
    
    #Global Optimisier to find best O sites
    ProposedSites = FindOptimalPositions(Position,
                                         CellDim, 
                                         NumberOfSites)
    
    #Coordinates of new O and O2 atoms
    NewOs = PlaceOAtoms(ProposedSites, O2AtomsRemoved, CellDim, bond_length = 1.2)
    
    
    #Prepare positions, Velocities and AtomInfo to combine with NewOs
    IndexLastO = len(Position.loc[Position['Element'] == 'O'].index)
    #Add in new Os to position file
    Position = pd.concat([Position.iloc[:IndexLastO], 
                          NewOs, 
                          Position.iloc[IndexLastO:]]).reset_index(drop = True)

    if 'O' in AtomInfo["Element"].unique():
        AtomInfo.loc[AtomInfo["Element"] == 'O', "Number"] += O2AtomsRemoved
    else:
        new_row = pd.DataFrame([{"Element": "O", "Number": O2AtomsRemoved}])
        AtomInfo = pd.concat([new_row, AtomInfo], ignore_index=True)


    if Velocities:
        #Add in velocities of new particles (for now, set to 0)
        NewOs[['x', 'y', 'z']] = 0
        NewOs.rename(columns = {'x' : 'vx',
                                'y' : 'vy',
                                'z' : 'vz'}, 
                    inplace = True)
        
        #Future function figuring out what we want velocities to be goes here
        
        Velocities = pd.concat([Velocities.iloc[:IndexLastO], 
                            NewOs, 
                            Velocities.iloc[IndexLastO:]]).reset_index(drop = True)
    
        return Position, Velocities, AtomInfo
    else:
        return Position, AtomInfo


def ProcessStructure(poscar_file, supercell_matrix):
    """
    Reads a POSCAR file, converts it to its **conventional** form, and creates a supercell.
    Avoids using the primitive cell to prevent loss of atoms.

    :param poscar_file: Path to the POSCAR file.
    :param supercell_matrix: 3x3 list defining supercell expansion.
    :return: Supercell structure.
    """
    # Load structure from POSCAR
    structure = Structure.from_file(poscar_file)
    
    #Use the **conventional** cell instead of the primitive cell
    analyzer = SpacegroupAnalyzer(structure)
    ConventionalStructure = analyzer.get_conventional_standard_structure()  # ✅ FIXED HERE

    #Create the supercell based on the conventional structure
    SupercellStructure = ConventionalStructure * supercell_matrix

    return SupercellStructure


def DoubleXDimension(PoscarFile, Factor, OutputFile="POSCAR_DoubledX"):
    """
    Reads a POSCAR file and expands the lattice along the x-direction by the given factor,
    creating an empty region without moving existing atoms.

    Parameters:
    - PoscarFile (str): Path to the original POSCAR file.
    - Factor (float): Expansion factor for the x-dimension.
    - OutputFile (str): Name of the output file (default: POSCAR_DoubledX).
    """
    # Load structure from POSCAR
    StructureObject = Structure.from_file(PoscarFile)

    # Get original lattice matrix
    OriginalLattice = StructureObject.lattice
    OriginalMatrix = OriginalLattice.matrix

    # Clean the original x-axis lattice vector to remove tiny non-x components
    CleanXVector = np.array([OriginalMatrix[0][0], 0.0, 0.0])

    # Create the new lattice matrix with scaled x-axis
    NewLatticeMatrix = np.array([
        CleanXVector * Factor,     # Scaled, clean x-axis vector
        OriginalMatrix[1],         # Original y-axis vector
        OriginalMatrix[2]          # Original z-axis vector
    ])

    # Create the new lattice object
    NewLattice = Lattice(NewLatticeMatrix)

    # Scale fractional x-coordinates to remain in the original position
    NewFracCoords = StructureObject.frac_coords.copy()
    NewFracCoords[:, 0] /= Factor

    # Create the new structure with the expanded lattice and original atoms in place
    NewStructure = Structure(
        lattice=NewLattice,
        species=StructureObject.species,
        coords=NewFracCoords,
        coords_are_cartesian=False
    )

    # Save the new structure to file
    NewStructure.to(fmt="poscar", filename=OutputFile)

    # Calculate lengths
    TotalLength = np.linalg.norm(NewLatticeMatrix[0])
    SolidLength = TotalLength / Factor
    VoidLength = TotalLength - SolidLength

    # Print lengths
    print(f"New total x-axis length: {TotalLength:.6f} Å")
    print(f"Solid region length: {SolidLength:.6f} Å")
    print(f"Void region length: {VoidLength:.6f} Å")
    print(f"Modified structure saved as '{OutputFile}'.")


def PivotAxis(Positions, CellDim, AtomInfo):
    """
    Pivot the positions of atoms along a specified axis (default is 'x').
    
    Parameters:
    - Positions (pd.DataFrame): DataFrame containing atom positions.
    - CellDim (pd.DataFrame): DataFrame containing cell dimensions.
    - Axis (str): Axis to pivot ('x', 'y', or 'z').
    
    Returns:
    - pd.DataFrame: Updated positions after pivoting.
    """

    Positions_swapped = Positions.copy()
    # Swap axes: Z becomes X, X becomes Y, Y becomes Z
    Positions_swapped['x', 'y', 'z'] = Positions['z'], Positions['x'], Positions['y']
    
    CellDim_swapped = CellDim.copy()
    # Swap cell dimensions: Z becomes X, X becomes Y, Y becomes Z
    CellDim_swapped['x', 'y', 'z'] = CellDim['z'], CellDim['x'], CellDim['y']
    
    vp.WritePOSCAR(WorkDir = 'Structures/ZrC75N25',
                   Position = Positions_swapped,
                   CellDim = CellDim_swapped,  # Assuming CellDim is not needed here
                   AtomInfo = AtomInfo)  # Assuming AtomInfo is not needed here
    
    # Pivot logic here
    # Placeholder for actual pivoting logic
    return Positions_swapped  # Return modified positions



#if __name__ == "__main__":
    #print(vp.__file__)
    #WorkDir = "Structures/ZrC75N25" 
    #PoscarPath = "Structures/ZrC75N25/ZrCN.poscar"  # Modify with your POSCAR file path
    #supercell_matrix = [3, 2, 2]  # Expand to 4x3x3 supercell
    #new_structure = ProcessStructure(PoscarPath, supercell_matrix)

    # Save to new POSCAR file
    #PoscarPath = "Structures/POSCAR"
    #new_structure.to(fmt="poscar", filename = PoscarPath)
    
    
    
    #factor = 3 #Factor of inital gas to solid ratio
    
    #DoubleXDimension(PoscarPath, factor, OutputFile=f"{WorkDir}/POSCAR")
    
    #Position, AtomInfo, CellDim = vp.ContcarParser(WorkDir = WorkDir,
    #                                               ReadPOSCAR = True)
    
    #Position, AtomInfo = AddO2(AtomInfo = AtomInfo,
    #                            O2AtomsRemoved = 20, 
    #                            Position = Position, 
    #                            CellDim = CellDim, 
    #                            Velocities = None)
    
    #vp.WritePOSCAR(WorkDir = WorkDir,
    #               Position = Position,
    #               CellDim = CellDim,
    #               AtomInfo = AtomInfo)
    
    #print('Done')
    

    

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #print(Position)
    
 
     

    
    vp.WritePOSCAR(WorkDir = 'Structures/ZrC75N25',
                   Position = Position,
                   CellDim = CellDim,
                   AtomInfo = AtomInfo)
    
    print('Done')
    #Next, add O2 atoms

    
'''

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
    '''
    
    return Position, CellDim

def FindOptimalPositions(Position, CellDim, n):
    '''
    Function which finds optimal positions for n new gas molecules.
    Uses a global optimiser to find positions maximising distance to existing atoms.
    '''
    
    return Position
    
    
def PreparePOSCAR(Position, CellDim, GasRatio = 2, InitO2 = 10):
    '''
    Function which prepares the POSCAR for oxidation simulations.
    Expands the x-dimension by GasRatio and scales fractional positions.
    Also places InitO2 O2 molecules in optimal positions.
    Does not do supercell expansion, seperate helper functin can be used for that.
    
    '''
    
    #First expand the x-dimension by GasRatio and scale fractional positions
     
     
    
    return Position, CellDim

# %% Demos and useful fileprep

if __name__ == "__main__":
    
    Position, CellDim = vio.ReadPOSCAR(workdir = '../../Test/',
                                       filename ='POSCAR_ZrCN_NoGas')
    
    print(Position)
    print(CellDim)
    
    Position, CellDim = SwapAxes(Position, CellDim, Axes = ('x','z'))
    
    print(Position)
    print(CellDim)
    
    vio.WritePOSCAR(WorkDir = '../../Test/',
    
# %%
