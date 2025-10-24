#%%
#Suite of functions to analyse VASP data parsed by VaspIO.py
#Most upstream import, make sure this doesnt have SGUSCHI imports

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
from typing import Optional, Tuple



# Graveyard for now

def ConvertCartesianToDirect(Position, CellDim):
    
    '''
    Converts all x, y, z coordinates in a position dataframe from their cartesian
    coordinates back to direct coordinates.
    
    Args:
        Position (pd.DataFrame): Atom positions with 'Element' and cartesian 
            coordinates ('x', 'y', 'z').
        CellDim (pd.DataFrame): A 3x3 DataFrame defining cell dimensions in angstroms.
    
    Returns:
        Position (pd.DataFrame): Atom positions with 'Element' and direct 
            coordinates ('x', 'y', 'z') in angstrom.
    '''
    
    CellDim = CellDim.to_numpy()
    InvCellDim = np.linalg.inv(CellDim)
    FracCoords = Position[['x','y','z']].to_numpy()
    CartCoords = FracCoords @ InvCellDim
    Position[['x','y','z']] = CartCoords
    
    return Position


def CalculateGasVolume(Position, CellDim):
    
    '''
    Calculates the volume of the oxidising gas by taking the distance between two 
    outward Zr atoms.
    '''

    PositionZrX = Position['x'].loc[Position['Element'] == 'Zr']        
    PositionZrX = PositionZrX.sort_values()
    
    XDistances = PositionZrX - PositionZrX.shift(1)
    XDistances.iloc[0] =  1 - (PositionZrX.iloc[-1] + PositionZrX.iloc[0])
    
    GasWidth = XDistances.max() * np.linalg.norm(CellDim['x'])
    SurfaceArea = np.linalg.norm(np.cross(CellDim['y'], CellDim['z']))
    
    GasVolume = GasWidth * SurfaceArea * 10 ** -30

    return GasVolume


def CalculateScaleThickness(Position, CellDim, AtomicRadiusTol):
    BondMatrix = FindGases(Position, 
                           CellDim, 
                           AtomicRadiusTol = AtomicRadiusTol,
                           ReturnBondMatrix = True)
    return BondMatrix


def CalculatePartialPressure(O2Molecules, Temp, GasVolume):
    
    R = 8.314462
    Na = 6.022 * 10**23
    atm = 101325

    PartialPressure = (O2Molecules * R * Temp) / (GasVolume * atm * Na)

    return PartialPressure


def CalculateOxidationRate(N, t, CellDim, PartialPressure,
                           PPConversion = 0.02, Alpha = 0.05):
    
    t = t * 1e-15
    PureRate = N / t
    
    CellSurfaceArea = (2 * np.linalg.norm(np.cross(CellDim['y'], CellDim['z'])) 
                       * 10 ** -20)
    
    OxidationRateConversion = PPConversion / CellSurfaceArea / PartialPressure
    
    OxRate = PureRate * OxidationRateConversion

    if N == 0:
        UpperBound = -np.log(Alpha) / t 
        UpperBound *= OxidationRateConversion
        return (OxRate, 0, UpperBound)
    else:
        ChiSquaredLowerBound = stats.chi2.ppf((Alpha / 2), (2 * N))
        ChiSquaredUpperBound = stats.chi2.ppf((1 - (Alpha / 2)), (2 * (N + 1)))
        LowerBound = ChiSquaredLowerBound / (2 * t)
        UpperBound = ChiSquaredUpperBound / (2 * t)
        LowerBound *= OxidationRateConversion
        UpperBound *= OxidationRateConversion
        return (OxRate, LowerBound, UpperBound)




'''
Gasses = an.FindGases(Position, 
                      CellDim, 
                      CovalentRadii = CovalentRadii,
                      AtomicRadiusTol = AtomicRadiusTol, 
                      MinimumComplexity = 2,
                      MaximumComplexity = 3,
                      ReturnBondMatrix = False)

New Workflow:

0. Checks to see if all elements in Position are covered in CovalentRadii
-> if not raise error and specify element which needs to be added

1. Build a Cartesian Distance Matrix Accounting for PBC

2. Run Bond Conditions over cartesian distance:
Takes bond existence as radius of overlap of CovalentRadii * User defined tolerance
-> the higher the tolerance the safer the algo is before removing a gas molecule

3. Prune O Bonds
Oxygen can only have 1 Oxygen bond at a time
(Is this enough?, requires some more testing)

4. Build Networks of Bonded Atoms

5. Filter out Any network MinimumComplexity >= x >= Maximum Complexity

6. Sort All Elements in Molecule

6. Return pd.DataFrame({'Molecule' : Molecules, 'Indices' : Indices})
'''

def MinimumDistancePBCVectorised(Position, CellDim):

    '''
    Returns the shortest distance between all points in a periodic boundary condition
    simulation according to the minimum image convention. CellDim must be converted to numpy format too.
    
    Args:
        Position (pd.DataFrame): Atom positions with 'Element' and fractional 
            coordinates ('x', 'y', 'z').
        pd.DataFrame: CellDim (pd.DataFrame): a 3x3 DataFrame defining cell dimensions in angstroms.
        
    Returns:
        CartDistanceMatrix (np.array): NxN matrix of all distances between atoms in angstrom. 
    '''
    
    FracCoords = Position[['x', 'y', 'z']].to_numpy()
    Displacement = FracCoords[:, np.newaxis, :] - FracCoords[np.newaxis, :, :]
    Displacement -= np.round(Displacement)
    CartDisplacement = Displacement @ CellDim
    CartDistanceMatrix = np.linalg.norm(CartDisplacement, axis = 2)
    
    return CartDistanceMatrix


def FindConnectedSubcomponents(AdjacencyMatrix):
    
    """
    Identifies connected subcomponents in a molecular system given an adjacency matrix.

    This function takes an (N, N) adjacency matrix and finds groups of atoms that are 
    connected to each other via bonds. Each subcomponent is a tuple containing the 
    indices of connected atoms. 
    Uses Depth-First Search (explores each branch before backtracking).

    Args:
        AdjacencyMatrix (np.ndarray): A (N, N) binary matrix (0s and 1s) where:
            - AdjacencyMatrix[i, j] = 1 indicates that atom i is bonded to atom j.

    Returns:
        list of tuple: A list where each tuple represents a connected molecular 
        subcomponent. Each tuple contains atom indices in ascending order.
    """

    NumAtoms = AdjacencyMatrix.shape[0]
    VisitedAtoms = set()
    Subcomponents = []

    for StartIndex in range(NumAtoms):
        if StartIndex not in VisitedAtoms:
            Queue = [StartIndex]
            CurrentMolecule = []
            
            while Queue:
                CurrentAtom = Queue.pop()
                if CurrentAtom not in VisitedAtoms:
                    VisitedAtoms.add(CurrentAtom)
                    CurrentMolecule.append(CurrentAtom)
                    Neighbors = np.where(AdjacencyMatrix[CurrentAtom])[0]
                    for Neighbor in Neighbors:
                        if Neighbor not in VisitedAtoms:
                            Queue.append(Neighbor)

            CurrentMolecule.sort()
            Subcomponents.append(tuple(CurrentMolecule))
            
    return Subcomponents



def FindGases(Position, 
              CellDim, 
              Targets = [['C','O','O'], ['C','O'], ['O','O']],
              Method = 'Adjacency',
              AtomicRadiusTol = 1.3,
              ReturnBondMatrix = False):

    """
    Identifies gas-phase molecules using a fixed-distance nearest neighbor algorithm.
    """

    Targets = [sorted(i) for i in Targets]
    
    GasIndices = pd.DataFrame(columns = ['Molecule', 'Indices'])
    Count = pd.DataFrame({'Molecule' : [tuple(i) for i in Targets], 
                          'Count' : np.zeros(len(Targets))})    

    Elements = Position['Element'].to_numpy()
    CellDim = CellDim.to_numpy()
    
    CartDistanceMatrix = MinimumDistancePBCVectorised(Position, CellDim)

    if Method == 'Adjacency':
        
        BondCutoffs = {
            ('O', 'O'): 1.5,
            ('C', 'O'): 1.7,
            ('O', 'C'): 1.7,
            ('C', 'C'): 1.7,
            ('Zr', 'O'): 1.8,
            ('O', 'Zr'): 1.8,
            ('Zr', 'Zr'): 1.9,
            ('Zr', 'C'): 1.9,
            }
        
        BondCutoffs.update((x, y * AtomicRadiusTol) for x, y in BondCutoffs.items())
        
        N = len(Position.index)
        BondMatrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):
                pair = (Elements[i], Elements[j])
                if pair in BondCutoffs:
                    cutoff = BondCutoffs[pair]
                    if CartDistanceMatrix[i, j] < cutoff:
                        BondMatrix[i, j] = 1
                        BondMatrix[j, i] = 1
        
        OIndices = [idx for idx, elem in enumerate(Elements) if elem == 'O']
        
        for i in OIndices:
            ONeighbors = [j for j in OIndices
                          if j != i and BondMatrix[i, j] == 1]
            if len(ONeighbors) > 1:
                OClosest = min(ONeighbors, key=lambda j: CartDistanceMatrix[i, j])
                for j in ONeighbors:
                    if j != OClosest:
                        BondMatrix[i, j] = 0
                        BondMatrix[j, i] = 0 
        
        if ReturnBondMatrix:
            return BondMatrix
        
        MoleculeIndices = FindConnectedSubcomponents(BondMatrix)
        
        for i in MoleculeIndices:
            Molecule = sorted([Elements[j] for j in i])
            if Molecule in Targets:
                GasIndices.loc[len(GasIndices.index)] = [tuple(Molecule), i]
                Count.loc[Count['Molecule'] == tuple(Molecule), 'Count'] += 1
            
        BondPairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if BondMatrix[i, j] == 1:
                    BondPairs.append(tuple(sorted([Elements[i], Elements[j]])))

        BondCounts = pd.DataFrame(pd.Series(BondPairs).value_counts()).reset_index()
        BondCounts.columns = ['Element Pair', 'Bond Count']
        
        return Count, GasIndices, BondCounts


# --- New gen Code ---

def ConvertDirectToCartesian(Position, CellDim): 
    '''
    Converts all x, y, z coordinates in a position dataframe from their fractional
    coordinates to cartesian coordinates. Useful for distance measurements.
    
    Args:
        Position (pd.DataFrame): Atom positions with 'Element' and fractional 
            coordinates ('x', 'y', 'z').
        CellDim (pd.DataFrame): A 3x3 DataFrame defining cell dimensions in angstroms.
    
    Returns:
        Position (pd.DataFrame): Atom positions with 'Element' and cartesian 
            coordinates ('x', 'y', 'z') in angstrom.
    '''
    
    CellDim = CellDim.to_numpy()
    FracCoords = Position[['x','y','z']].to_numpy()
    CartCoords = FracCoords @ CellDim
    Position[['x','y','z']] = CartCoords
    
    return Position


def CalculateGasFraction(Position, GasRatio):
    """
    Compute the gas fraction relative to beginning value from fractional x 
    positions of Zr atoms and Initial GasRatio.

    Parameters
    ----------
    Position : pandas.DataFrame
        Must contain columns 'x' and 'Element'.
    GasRatio : float
        Ratio used to compute starting distance.

    Returns
    -------
    float
        The gas fraction value. This is the fraction of gas in the simulation
        relative to the starting value.
    """

    # Reference distance (based on GasRatio)
    StartingDistance = 1.0 / (GasRatio + 1.0)

    # Extract and sort x-coordinates for Zr atoms only
    XCoordinates = (
        Position.loc[Position['Element'] == 'Zr', 'x']
        .sort_values()
        .to_numpy()
    )

    # Compute neighbor distances (left differences)
    Diffs = np.diff(XCoordinates)

    # Account for periodic boundary condition (wrap-around)
    WrapDistance = 1.0 - XCoordinates[-1] + XCoordinates[0]

    # Combine all distances, including wrap-around
    AllDistances = np.append(Diffs, WrapDistance)

    # Find largest distance between any two neighboring points
    MaxDistance = np.max(AllDistances)

    # Compute gas fraction
    GasFraction = MaxDistance / StartingDistance

    return GasFraction


def MaxwellBoltzmannVelocities(Elements, Temperature):
    """
    Generate random Maxwell–Boltzmann distributed velocities for a list of atomic elements.

    This function assigns each element in the input list a 3D velocity vector 
    sampled from a normal distribution centered at zero, with a standard deviation 
    determined by the Maxwell–Boltzmann distribution for a given temperature. 
    The velocities are converted to Ångström per femtosecond (Å/fs).

    Parameters
    ----------
    Elements : list of str
        A list of element symbols (e.g., ['O', 'C', 'Zr', 'N']).
    Temperature : float
        The temperature in Kelvin at which to sample velocities.

    Returns
    -------
    list of numpy.ndarray
        A list of 3D velocity vectors (each as a NumPy array) corresponding to 
        each element in the input list, in units of Å/fs.

    Raises
    ------
    ValueError
        If any element in `Elements` is not found in the predefined mass dictionary.

    Notes
    -----
    The following constants are used:
        - Boltzmann constant (k_B) = 1.380649 × 10⁻²³ J/K
        - Atomic mass unit (AMU_to_kg) = 1.66054 × 10⁻²⁷ kg
        - Conversion factor from m/s to Å/fs (ms_to_Afs) = 1 × 10⁻⁵
    """
    ElementMass = {'O': 15.99, 'C': 12.01, 'Zr': 91.22, 'N': 14.01}

    k_B = 1.380649e-23
    AMU_to_kg = 1.66054e-27
    ms_to_Afs = 1e-5

    Velocities = []

    for Element in Elements:
        if Element not in ElementMass:
            raise ValueError("Element {} not found in mass dictionary.".format(Element))

        MassKg = ElementMass[Element] * AMU_to_kg
        Sigma = np.sqrt(k_B * Temperature / MassKg)
        Velocity = np.random.normal(0, Sigma, 3)
        Velocity *= ms_to_Afs
        Velocities.append(Velocity)

    return Velocities


def FindOptimalCoords(Position: pd.DataFrame,
                      CellDim: pd.DataFrame,
                      N: int = 1,
                      ReturnRadius: bool = False,
                      Seed: Optional[int] = None,
                      MaxIterDE: int = 300,
                      PopSizeDE: int = 15) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Find optimal fractional coordinates for placing new atoms or molecules
    within a periodic simulation cell using global optimization.

    Uses differential evolution (global) for all N to avoid local minima. The objective
    maximizes the minimal Cartesian distance to existing atoms (and between new sites
    when N > 1), under periodic boundary conditions. Returned coordinates are fractional.

    Parameters
    ----------
    Position : pandas.DataFrame
        Existing atomic positions with columns ['x', 'y', 'z'] (fractional).
    CellDim : pandas.DataFrame
        Lattice vectors with columns ['x', 'y', 'z'] (Cartesian, rows are a,b,c).
    N : int, optional
        Number of new sites to place. Default is 1.
    ReturnRadius : bool, optional
        If True, also return the achieved minimum distance (radius). Default False.
    Seed : int, optional
        Random seed for reproducibility.
    MaxIterDE : int, optional
        Max iterations for differential evolution. Default 300.
    PopSizeDE : int, optional
        Population size multiplier for differential evolution. Default 15.

    Returns
    -------
    pandas.DataFrame or Tuple[pandas.DataFrame, float]
        DataFrame with columns ['x','y','z'] of optimal sites; optionally the radius.

    Raises
    ------
    ValueError
        If the required columns are missing.

    Notes
    -----
    - Fractional coordinates are wrapped to [0, 1).
    - DE provides global exploration to avoid local maxima traps.
    """

    # --- Helper functions ---
    def Wrap01(Array: np.ndarray) -> np.ndarray:
        return Array - np.floor(Array)

    def PBCDelta(FracA: np.ndarray, FracB: np.ndarray) -> np.ndarray:
        Delta = FracA - FracB
        return Delta - np.round(Delta)

    def CartDistFromFracDelta(DeltaFrac: np.ndarray, CellDimArray: np.ndarray) -> np.ndarray:
        RCart = DeltaFrac @ CellDimArray
        return np.linalg.norm(RCart, axis=-1)

    def MinDistanceToExisting(PointsFrac: np.ndarray,
                              FracExisting: np.ndarray,
                              CellDimArray: np.ndarray) -> np.ndarray:
        if FracExisting.size == 0:
            return np.full(PointsFrac.shape[0], np.inf)
        DFrac = PBCDelta(PointsFrac[:, None, :], FracExisting[None, :, :])
        Dists = CartDistFromFracDelta(DFrac, CellDimArray)
        return Dists.min(axis=1)

    def MinPairwiseAmongNew(PointsFrac: np.ndarray,
                            CellDimArray: np.ndarray) -> float:
        M = PointsFrac.shape[0]
        if M < 2:
            return np.inf
        DFrac = PBCDelta(PointsFrac[:, None, :], PointsFrac[None, :, :])
        Dists = CartDistFromFracDelta(DFrac, CellDimArray)
        np.fill_diagonal(Dists, np.inf)
        return Dists.min()

    def Radius(PointsFrac: np.ndarray,
               FracExisting: np.ndarray,
               CellDimArray: np.ndarray) -> float:
        PointsFrac = Wrap01(PointsFrac)
        DistToExisting = MinDistanceToExisting(PointsFrac, FracExisting, CellDimArray)
        R1 = float(DistToExisting.min()) if DistToExisting.size else np.inf
        R2 = float(MinPairwiseAmongNew(PointsFrac, CellDimArray))
        return min(R1, R2)

    # --- Input validation ---
    if not all(C in Position.columns for C in ['x', 'y', 'z']):
        raise ValueError("Position must include columns ['x','y','z'].")
    if not all(C in CellDim.columns for C in ['x', 'y', 'z']):
        raise ValueError("CellDim must include columns ['x','y','z'].")

    FracExisting = Position[['x', 'y', 'z']].to_numpy(float)
    CellDimArray = CellDim[['x', 'y', 'z']].to_numpy(float)

    # --- Objective (works for any N) ---
    def ObjectiveMultiple(FlatPoints: np.ndarray) -> float:
        PointsFrac = Wrap01(np.asarray(FlatPoints, dtype=float).reshape(N, 3))
        return -Radius(PointsFrac, FracExisting, CellDimArray)

    # --- Global optimization with DE for all N (including N == 1) ---
    Bounds = [(0.0, 1.0)] * (3 * N)
    
    Result = opt.differential_evolution(
        ObjectiveMultiple,
        bounds=Bounds,
        seed=Seed,
        strategy='best1bin',
        popsize=PopSizeDE,
        maxiter=MaxIterDE,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,             
        updating='deferred',
        workers=1,
        init='latinhypercube' #Best for identifying gas phase
    )

    OptimalPoints = Wrap01(Result.x.reshape(N, 3))
    OptimalSites = pd.DataFrame(OptimalPoints, columns=['x', 'y', 'z'])
    RadiusValue = -float(Result.fun)

    return (OptimalSites, RadiusValue) if ReturnRadius else OptimalSites


def PlaceO2Molecules(Positions: pd.DataFrame,
                     CellDim: pd.DataFrame,
                     NewSites: pd.DataFrame,
                     BondLength: float = 1.2):
    '''
    Places O2 molecules aligned along the x-axis at specified fractional coordinates.

    Each molecule is centered at the given fractional coordinate from NewSites
    and consists of two O atoms separated by BondLength (Å) along the x-axis.

    Args:
        Positions (pd.DataFrame): Existing atom positions with columns ['Element','x','y','z']
            in fractional coordinates.
        CellDim (pd.DataFrame): 3x3 DataFrame defining lattice vectors in Ångström.
        NewSites (pd.DataFrame): Fractional coordinates ('x','y','z') where O2 molecules
            should be centered.
        BondLength (float): O–O bond length in Ångström.

    Returns:
        pd.DataFrame: Updated Positions DataFrame including added O atoms.
    '''

    # Validate inputs
    for df, name in [(Positions, "Positions"), (CellDim, "CellDim"), (NewSites, "NewSites")]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
    if not all(c in Positions.columns for c in ['x','y','z']):
        raise ValueError("Positions must contain ['x','y','z'] columns.")
    if not all(c in CellDim.columns for c in ['x','y','z']):
        raise ValueError("CellDim must contain ['x','y','z'] columns.")
    if not all(c in NewSites.columns for c in ['x','y','z']):
        raise ValueError("NewSites must contain ['x','y','z'] columns.")

    # Convert to numpy arrays
    CellArray = CellDim[['x','y','z']].to_numpy(float)
    NewSitesArray = NewSites[['x','y','z']].to_numpy(float)

    # Compute x-axis lattice vector and its norm (Å)
    AVector = CellArray[0, :]   # first lattice vector (x-axis)
    ALength = np.linalg.norm(AVector)
    if ALength == 0:
        raise ValueError("Invalid CellDim: x lattice vector has zero length.")

    # Bond displacement in fractional coordinates
    HalfFracDisp = (BondLength / (2.0 * ALength))  # half bond in fractional units along x

    # Prepare list for new O atoms
    NewAtoms = []

    for Site in NewSitesArray:
        # Two atoms along ±x direction in fractional coordinates
        O1 = Site.copy()
        O2 = Site.copy()
        O1[0] = (O1[0] - HalfFracDisp) % 1.0
        O2[0] = (O2[0] + HalfFracDisp) % 1.0

        NewAtoms.append({'Element': 'O', 'x': O1[0], 'y': O1[1], 'z': O1[2]})
        NewAtoms.append({'Element': 'O', 'x': O2[0], 'y': O2[1], 'z': O2[2]})

    # Convert to DataFrame
    NewAtomsDF = pd.DataFrame(NewAtoms, columns=['Element','x','y','z'])

    # Combine with existing positions
    UpdatedPositions = pd.concat([Positions, NewAtomsDF], ignore_index=True)

    return UpdatedPositions

