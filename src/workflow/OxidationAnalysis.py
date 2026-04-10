#%%
#Suite of functions to analyse VASP data parsed by VaspIO.py
#Most upstream import, make sure this doesnt have SGUSCHI imports

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
from typing import Dict, List, Tuple, Optional, Union, Set



# Graveyard for now

def ConvertCartesianToDirect(Position: pd.DataFrame, CellDim: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Cartesian coordinates (Å) to direct (fractional, unitless).
    Overwrites Position[['x','y','z']] in-place and returns Position.
    """
    CellMatrix = CellDim.to_numpy()
    InvCellMatrix = np.linalg.inv(CellMatrix)
    Cartesian = Position[['x', 'y', 'z']].to_numpy()
    Direct = Cartesian @ InvCellMatrix
    Position[['x', 'y', 'z']] = Direct
    return Position


# Check where this is used, this doesn't seem done lol
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

New BondFinder Workflow:

0. Checks to see if all elements in Position are in in CovalentRadii dict
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

6. Sort All Elements in Molecule for identification steps downsteam

6. Return pd.DataFrame({'Molecule' : Molecules, 'Indices' : Indices})
if Bondmatrix gets requested give that too. Maybe do some PP on it to make it 
more useful. 
'''


# --- New gen Code ---



def CheckElementsInRadii(Position: pd.DataFrame, 
                         CovalentRadii: Dict[str, float]
                         ) -> None:
    """
    Validate that every element present in Position has a radius in CovalentRadii.
    Raises a ValueError listing any missing symbols.
    """
    ElementsInFrame = set(Position["Element"].unique().tolist())
    ElementsInRadii = set(CovalentRadii.keys())
    Missing = sorted(ElementsInFrame - ElementsInRadii)
    if Missing:
        raise ValueError(
            "CovalentRadii is missing entries for: {}".format(", ".join(Missing))
        )


def MinimumDistancePBCVectorised(Position: pd.DataFrame, 
                                 CellDim: Union[pd.DataFrame, np.ndarray]
                                 ) -> np.ndarray:
    """
    Build NxN Cartesian distance matrix with minimum-image PBC.
    Expects fractional coordinates in Position[['x','y','z']].
    CellDim: 3x3 (Å) with lattice vectors as columns (standard VASP-style).
    """
    if isinstance(CellDim, pd.DataFrame):
        CellMat = CellDim.to_numpy()
    else:
        CellMat = np.asarray(CellDim)

    FracCoords = Position[["x", "y", "z"]].to_numpy()

    # Pairwise fractional displacements
    Displacement = FracCoords[:, np.newaxis, :] - FracCoords[np.newaxis, :, :]
    # Tie-safe minimum image (avoids bankers' rounding issues)
    Displacement -= np.floor(Displacement + 0.5)

    # Convert to Cartesian
    CartDisplacement = Displacement @ CellMat
    CartDistanceMatrix = np.linalg.norm(CartDisplacement, axis=2)

    return CartDistanceMatrix


def BuildBondMatrixFromRadii(
    Elements: np.ndarray,
    CartDistanceMatrix: np.ndarray,
    CovalentRadii: Dict[str, float],
    AtomicRadiusTol: float
    ) -> np.ndarray:
    """
    Create a symmetric boolean adjacency by comparing distances
    to (r_i + r_j) * AtomicRadiusTol.
    """
    #N = Elements.shape[0]
    RadiiPerAtom = np.array([CovalentRadii[el] for el in Elements], dtype=float)
    PairwiseCutoff = (RadiiPerAtom[:, None] + RadiiPerAtom[None, :]) * AtomicRadiusTol

    BondMatrix = (CartDistanceMatrix < PairwiseCutoff)
    # Never bond an atom to itself
    np.fill_diagonal(BondMatrix, False)

    return BondMatrix


def EnforceUniqueOOBonds(
    BondMatrix: np.ndarray,
    Elements: np.ndarray,
    CartDistanceMatrix: np.ndarray
    ) -> np.ndarray:
    """
    Step 3: Prune O–O bonds so each O participates in at most one O–O bond.
    Greedy global matching on shortest O–O distances.
    """
    OIndices = np.where(Elements == "O")[0]
    if OIndices.size < 2:
        return BondMatrix

    # Get all current O–O bonds in the upper triangle
    TriI, TriJ = np.triu_indices(BondMatrix.shape[0], k=1)
    MaskOO = (
        (Elements[TriI] == "O") &
        (Elements[TriJ] == "O") &
        (BondMatrix[TriI, TriJ])
    )

    CandI = TriI[MaskOO]
    CandJ = TriJ[MaskOO]
    if CandI.size == 0:
        return BondMatrix

    Dists = CartDistanceMatrix[CandI, CandJ]
    Order = np.argsort(Dists)

    Used = np.zeros(BondMatrix.shape[0], dtype=bool)

    # Remove all O–O bonds; add back only matched ones
    BondMatrix[np.ix_(OIndices, OIndices)] = False
    for k in Order:
        I = CandI[k]
        J = CandJ[k]
        if (not Used[I]) and (not Used[J]):
            BondMatrix[I, J] = True
            BondMatrix[J, I] = True
            Used[I] = True
            Used[J] = True

    return BondMatrix


def FindConnectedSubcomponents(AdjacencyMatrix: np.ndarray
                               ) -> List[Tuple[int, ...]]:
    """
    Step 4: Identify connected components on a boolean adjacency (NxN).
    Returns a list of tuples of atom indices (ascending).
    """
    N = AdjacencyMatrix.shape[0]
    Visited = np.zeros(N, dtype=bool)
    Subcomponents: List[Tuple[int, ...]] = []

    for StartIndex in range(N):
        if not Visited[StartIndex]:
            Queue = [StartIndex]
            Current: List[int] = []
            while Queue:
                Node = Queue.pop()
                if not Visited[Node]:
                    Visited[Node] = True
                    Current.append(Node)
                    Neighbors = np.where(AdjacencyMatrix[Node])[0]
                    for Nei in Neighbors:
                        if not Visited[Nei]:
                            Queue.append(Nei)
            Current.sort()
            Subcomponents.append(tuple(Current))
    return Subcomponents


def FindGases(
    Position: pd.DataFrame,
    CellDim: Union[pd.DataFrame, np.ndarray],
    CovalentRadii: Dict[str, float],
    AtomicRadiusTol: float = 1.05,
    MinimumComplexity: int = 2,
    MaximumComplexity: int = 3,
    ReturnBondMatrix: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
   Identify small, gas-like molecular fragments in a single AIMD frame using
   PBC-aware distances and covalent-radii bonding.

   Workflow:
   0) Verifies that every element in `Position` has an entry in `CovalentRadii`.
   1) Builds a minimum-image Cartesian distance matrix from fractional coordinates and `CellDim`.
   2) Creates bonds where d_ij < (r_i + r_j) * `AtomicRadiusTol`.
   3) Prunes O–O bonds via a shortest-first global matching so each O has at most one O–O neighbor.
   4) Finds connected components (molecular subgraphs).
   5) Keeps only components whose size (atom count) is within [`MinimumComplexity`, `MaximumComplexity`].
   6) Sorts the element symbols within each component for stable identification and returns the results.

    Args:
    Position (pd.DataFrame):
        Per-atom data with columns:
          - 'Element' (str): chemical symbol for each atom.
          - 'x', 'y', 'z' (float): **fractional** coordinates (0–1) with respect to `CellDim`.
    CellDim (Union[pd.DataFrame, np.ndarray]):
        3×3 lattice matrix in ångström. Lattice vectors are expected as **columns**
        (i.e., Cartesian = fractional @ CellDim). A pandas 3×3 DataFrame or a NumPy array are accepted.
    CovalentRadii (Dict[str, float]):
        Mapping from element symbol to covalent radius (Å). Every element present in `Position`
        must be provided; otherwise a ValueError is raised.
    AtomicRadiusTol (float, optional):
        Multiplicative tolerance applied to the sum of covalent radii when deciding bonds.
        Larger values yield more permissive bonding. Default is 1.05.
    MinimumComplexity (int, optional):
        Minimum number of atoms allowed in a returned component (inclusive). Default is 2.
    MaximumComplexity (int, optional):
        Maximum number of atoms allowed in a returned component (inclusive). Default is 3.
    ReturnBondMatrix (bool, optional):
        If True, also return the boolean N×N adjacency (bond) matrix. Default is False.

    Returns:
    pd.DataFrame or Tuple[pd.DataFrame, np.ndarray]:
        If `ReturnBondMatrix` is False:
            A DataFrame with two columns:
              - 'Molecule': tuple of sorted element symbols in the component (e.g., ('C','O','O')).
              - 'Indices' : tuple of atom indices (ascending) belonging to that component.
        If `ReturnBondMatrix` is True:
            A tuple of (ResultDataFrame, BondMatrix), where `BondMatrix` is an (N, N) boolean
            NumPy array indicating bonds (symmetric with a False diagonal).
    """
    # Step 0
    CheckElementsInRadii(Position, CovalentRadii)

    # Step 1
    CartDistanceMatrix = MinimumDistancePBCVectorised(Position, CellDim)

    # Step 2
    Elements = Position["Element"].to_numpy()
    BondMatrix = BuildBondMatrixFromRadii(
        Elements=Elements,
        CartDistanceMatrix=CartDistanceMatrix,
        CovalentRadii=CovalentRadii,
        AtomicRadiusTol=AtomicRadiusTol
    )

    # Step 3
    BondMatrix = EnforceUniqueOOBonds(BondMatrix, Elements, CartDistanceMatrix)

    # Step 4
    Components = FindConnectedSubcomponents(BondMatrix)

    # Step 5: filter by complexity and Step 6: sort element labels
    Molecules: List[Tuple[str, ...]] = []
    Indices: List[Tuple[int, ...]] = []

    for Comp in Components:
        Size = len(Comp)
        if (Size >= MinimumComplexity) and (Size <= MaximumComplexity):
            Labels = tuple(sorted([Elements[i] for i in Comp]))
            Molecules.append(Labels)
            Indices.append(tuple(Comp))

    Result = pd.DataFrame({"Molecule": Molecules, "Indices": Indices})

    if ReturnBondMatrix:
        return Result, BondMatrix
    return Result



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


def CalculateGasFraction(Position: pd.DataFrame, GasRatio: float, 
                         ClipToOne: bool = True) -> float:
    """
    Compute the gas fraction (current void size relative to starting void size)
    along the fractional x-direction using Zr atom positions.

    GasRatio is defined as L_gas / L_solid along x, so the starting void length
    is L_gas0 = GasRatio / (1 + GasRatio). For GasRatio=2, L_gas0 = 2/3 ≈ 0.6667.

    The current void length is estimated as the largest gap between neighboring
    Zr fractional x-coordinates under PBC. The returned fraction is:
        GasFraction = CurrentVoidLength / StartingVoidLength
    which begins at ~1.0 and decreases (e.g., 0.8) as the gas region shrinks.

    Args:
        Position (pd.DataFrame): Must contain columns 'Element' and fractional 'x'.
        GasRatio (float): Positive ratio L_gas / L_solid at the start.
        ClipToOne (bool): If True, cap the fraction at 1.0 (never exceed starting).

    Returns:
        float: Gas fraction relative to the starting amount of gas.

    Raises:
        ValueError: If GasRatio <= 0.
    """

    # Starting void length (fraction of the unit cell along x)
    StartingVoidLength = float(GasRatio / (GasRatio + 1.0))

    # Extract Zr x-fractional coordinates
    XCoordinates = Position.loc[Position['Element'] == 'Zr', 'x'].to_numpy()

    # Sort and compute nearest-neighbor gaps with PBC wrap-around
    XCoordinates.sort()
    Diffs = np.diff(XCoordinates)
    WrapDistance = 1.0 - XCoordinates[-1] + XCoordinates[0]

    if Diffs.size:
        MaxDistance = float(max(WrapDistance, np.max(Diffs)))
    else:
        MaxDistance = float(WrapDistance)

    GasFraction = MaxDistance / StartingVoidLength

    if ClipToOne:
        GasFraction = min(GasFraction, 1.0)

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


def RemoveNonO2Gasses(
    Position: pd.DataFrame,
    Velocity: Optional[pd.DataFrame],
    Gasses: Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Remove all gas-like fragments identified by `FindGases` that are NOT O2.
    Keeps any ('O','O') molecules and drops all atoms belonging to other
    2–3 atom components from Position (and Velocity, if provided).

    Returns:
        If Velocity is None:
            PositionFiltered (pd.DataFrame)
        Else:
            (PositionFiltered, VelocityFiltered)
    """
    if isinstance(Gasses, tuple):
        Gasses = Gasses[0]

    RequiredColumns = {"Molecule", "Indices"}
    if not RequiredColumns.issubset(set(Gasses.columns)):
        raise ValueError("Gasses must have columns {'Molecule','Indices'}.")

    KeepTuple = ("O", "O")
    IndicesToRemove: Set[int] = set()

    for Molecule, Indices in zip(Gasses["Molecule"], Gasses["Indices"]):
        MoleculeTuple = tuple(Molecule)
        if MoleculeTuple != KeepTuple:
            IndicesToRemove.update(int(i) for i in Indices)

    if not IndicesToRemove:
        if Velocity is None:
            return Position.copy()
        return Position.copy(), Velocity.copy()

    MissingInPosition = [i for i in IndicesToRemove if i not in Position.index]
    if MissingInPosition:
        raise IndexError(f"Some removal indices not found in Position: {MissingInPosition[:10]}...")

    PositionFiltered = Position.drop(labels=list(IndicesToRemove)).reset_index(drop=True)

    if Velocity is None:
        return PositionFiltered

    MissingInVelocity = [i for i in IndicesToRemove if i not in Velocity.index]
    if MissingInVelocity:
        raise IndexError(f"Some removal indices not found in Velocity: {MissingInVelocity[:10]}...")

    VelocityFiltered = Velocity.drop(labels=list(IndicesToRemove)).reset_index(drop=True)
    return PositionFiltered, VelocityFiltered

# %%
