#%%
#Suite of functions to analyse VASP data parsed by VaspIO.py
#Cannot have dependencies

#Hard coded things that could be taken from database: 
#1. Bond Length (FindGasses), Look at ionic radii overlap
#2. Atomic Mass (AMU) (MaxwellBoltmannVelocities), just obtain from DataBase

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
from typing import Optional


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


def MinimumDistancePBCVectorised(FracCoords, CellDim):

    '''
    Returns the shortest distance between all points in a periodic boundary condition
    simulation according to the minimum image convention. CellDim must be converted to numpy format too.
    
    Args:
        FracCoords (np.array): Fractional coordinates of atoms in numpy array form. 
            Dimensions are 3xN. Elements column must have been removed.
        pd.DataFrame: CellDim (pd.DataFrame): a 3x3 DataFrame defining cell dimensions in angstroms.
        
    Returns:
        CartDistanceMatrix (np.array): NxN matrix of all distances between atoms in angstrom. 
    '''

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

    FracCoords = Position[['x', 'y', 'z']].to_numpy()
    Elements = Position['Element'].to_numpy()
    CellDim = CellDim.to_numpy()
    
    CartDistanceMatrix = MinimumDistancePBCVectorised(FracCoords, CellDim)

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
        
        N = len(FracCoords)
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


#New code

def MaxwellBoltzmannVelocities(Elements, Temperature):
    '''
    Generate random Maxwell-Boltzmann velocities for atoms.
    '''
    
    ElementMass = {'O': 15.99, 'C': 12.01, 'Zr': 91.22}
    
    k_B = 1.380649e-23
    AMU_to_kg = 1.66054e-27
    ms_to_Afs = 1e-5

    Velocities = []
    
    for Element in Elements:
        if Element not in ElementMass:
            raise ValueError("Element {} not found in mass dictionary.".format(Element))
        
        Mass_kg = ElementMass[Element] * AMU_to_kg
        Sigma = np.sqrt(k_B * Temperature / Mass_kg)
        Velocity = np.random.normal(0, Sigma, 3)
        Velocity *= ms_to_Afs
        Velocities.append(Velocity)
    
    return Velocities


def FindOptimalCoords(Position: pd.DataFrame,
                      CellDim: pd.DataFrame,
                      n: int = 1,
                      ReturnRadius: bool = False,
                      Seed: Optional[int] = None,
                      MaxIterDE: int = 300,
                      PopSizeDE: int = 15):
    '''
    Finds optimal fractional coordinates for the placement of n new atoms/molecules.
    '''

    def Wrap01(Array):
        return Array - np.floor(Array)

    def PBCDelta(FracA, FracB):
        Delta = FracA - FracB
        return Delta - np.round(Delta)

    def CartDistFromFracDelta(DeltaFrac, CellDimArray):
        RCart = DeltaFrac @ CellDimArray
        return np.linalg.norm(RCart, axis=-1)

    def MinDistanceToExisting(PointsFrac, FracExisting, CellDimArray):
        if FracExisting.size == 0:
            return np.full(PointsFrac.shape[0], np.inf)
        DFrac = PBCDelta(PointsFrac[:, None, :], FracExisting[None, :, :])
        Dists = CartDistFromFracDelta(DFrac, CellDimArray)
        return Dists.min(axis=1)

    def MinPairwiseAmongNew(PointsFrac, CellDimArray):
        M = PointsFrac.shape[0]
        if M < 2:
            return np.inf
        DFrac = PBCDelta(PointsFrac[:, None, :], PointsFrac[None, :, :])
        Dists = CartDistFromFracDelta(DFrac, CellDimArray)
        np.fill_diagonal(Dists, np.inf)
        return Dists.min()

    def Radius(PointsFrac, FracExisting, CellDimArray):
        PointsFrac = Wrap01(PointsFrac)
        DistToExisting = MinDistanceToExisting(PointsFrac, FracExisting, CellDimArray)
        R1 = float(DistToExisting.min()) if DistToExisting.size else np.inf
        R2 = float(MinPairwiseAmongNew(PointsFrac, CellDimArray))
        return min(R1, R2)

    if not all(c in Position.columns for c in ['x','y','z']):
        raise ValueError("Position must include columns ['x','y','z'].")
    if not all(c in CellDim.columns for c in ['x','y','z']):
        raise ValueError("CellDim must include columns ['x','y','z'].")

    FracExisting = Position[['x','y','z']].to_numpy(float)
    CellDimArray = CellDim[['x','y','z']].to_numpy(float)

    def ObjectiveSingle(Point):
        Point = Wrap01(np.asarray(Point, dtype=float))
        return -Radius(Point[None, :], FracExisting, CellDimArray)

    def ObjectiveMultiple(FlatPoints):
        PointsFrac = Wrap01(np.asarray(FlatPoints, dtype=float).reshape(n, 3))
        return -Radius(PointsFrac, FracExisting, CellDimArray)

    RNG = np.random.default_rng(Seed)

    if n == 1:
        Seeds = [np.array([0.5, 0.5, 0.5])]
        Seeds += list(RNG.random((7, 3)))

        BestPoint = None
        BestVal = np.inf
        for Start in Seeds:
            Result = opt.minimize(
                ObjectiveSingle,
                x0=Start,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-4, 'ftol': 1e-4},
            )
            if Result.fun < BestVal:
                BestVal = Result.fun
                BestPoint = Wrap01(Result.x)

        OptimalSites = pd.DataFrame([BestPoint], columns=['x','y','z'])
        RadiusValue = -float(BestVal)

    else:
        Bounds = [(0.0, 1.0)] * (3 * n)
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
        )

        OptimalPoints = Wrap01(Result.x.reshape(n, 3))
        OptimalSites = pd.DataFrame(OptimalPoints, columns=['x','y','z'])
        RadiusValue = -float(Result.fun)

    if ReturnRadius:
        return OptimalSites, RadiusValue
    else:
        return OptimalSites


def PlaceO2Molecules(Position: pd.DataFrame,
                     CellDim: pd.DataFrame,
                     NewSites: pd.DataFrame,
                     BondLength: float = 1.2):
    '''
    Places O2 molecules aligned along the x-axis at specified fractional coordinates.
    '''

    for df, name in [(Position, "Positions"), (CellDim, "CellDim"), (NewSites, "NewSites")]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("{} must be a pandas DataFrame.".format(name))
    if not all(c in Position.columns for c in ['x','y','z']):
        raise ValueError("Positions must contain ['x','y','z'] columns.")
    if not all(c in CellDim.columns for c in ['x','y','z']):
        raise ValueError("CellDim must contain ['x','y','z'] columns.")
    if not all(c in NewSites.columns for c in ['x','y','z']):
        raise ValueError("NewSites must contain ['x','y','z'] columns.")

    CellArray = CellDim[['x','y','z']].to_numpy(float)
    NewSitesArray = NewSites[['x','y','z']].to_numpy(float)

    AVector = CellArray[0, :]
    ALength = np.linalg.norm(AVector)
    if ALength == 0:
        raise ValueError("Invalid CellDim: x lattice vector has zero length.")

    HalfFracDisp = (BondLength / (2.0 * ALength))

    NewAtoms = []

    for Site in NewSitesArray:
        O1 = Site.copy()
        O2 = Site.copy()
        O1[0] = (O1[0] - HalfFracDisp) % 1.0
        O2[0] = (O2[0] + HalfFracDisp) % 1.0

        NewAtoms.append({'Element': 'O', 'x': O1[0], 'y': O1[1], 'z': O1[2]})
        NewAtoms.append({'Element': 'O', 'x': O2[0], 'y': O2[1], 'z': O2[2]})

    # Convert to DataFrame and append to existing positions
    NewAtomsDF = pd.DataFrame(NewAtoms, columns=['Element', 'x', 'y', 'z'])
    UpdatedPositions = pd.concat([Position, NewAtomsDF], ignore_index=True)

    return UpdatedPositions
