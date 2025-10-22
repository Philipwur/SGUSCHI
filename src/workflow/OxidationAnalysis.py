#%%
#Suite of functions to analyse VASP data parsed by VaspIO.py

#Hard coded things that could be taken from database: 
#1. Bond Length (FindGasses), Look at ionic radii overlap
#2. Atomic Mass (AMU) (MaxwellBoltmannVelocities), just obtain from DataBase

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt



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
    VisitedAtoms = set() #To ensure we don't repeat count atoms
    Subcomponents = [] #Array for storing each seperate molecule

    #Loop through atoms to find connected components
    for StartIndex in range(NumAtoms):
        
        #Check if already visited
        if StartIndex not in VisitedAtoms:
            Queue = [StartIndex] #Used to keep track of search
            CurrentMolecule = [] #Start new molecule stack (if not in other molecule must be new)
            
            while Queue:
                CurrentAtom = Queue.pop() #Remove last atom added to search stack
                if CurrentAtom not in VisitedAtoms:
                    VisitedAtoms.add(CurrentAtom) #Mark as visted
                    CurrentMolecule.append(CurrentAtom) #Add atom to molecule
                    
                    #Find all bonded neighbors (atoms directly connected to CurrentAtom)
                    Neighbors = np.where(AdjacencyMatrix[CurrentAtom])[0] # Get indices of connected atoms
                    for Neighbor in Neighbors:
                        if Neighbor not in VisitedAtoms: #Only visit unvisited atoms
                            Queue.append(Neighbor) #Add to queue for future exploration

            #Sort indices for consistency
            CurrentMolecule.sort()
            Subcomponents.append(tuple(CurrentMolecule)) #Add to seperate molecules.
            
    return Subcomponents


def FindGases(Position, 
              CellDim, 
              Targets = [['C','O','O'], ['C','O'], ['O','O']],
              Method = 'Adjacency',
              AtomicRadiusTol = 1.3,
              ReturnBondMatrix = False):

    """
    Identifies gas-phase molecules using a fixed-distance nearest neighbor algorithm.

    This function analyses atomic positions, detects bonds based on fixed cutoff distances, 
    and identifies molecular subcomponents. It returns:
    - A count of detected gas-phase molecules.
    - The indices of atoms belonging to each identified molecule.
    - A bond count DataFrame that records the number of bonds for each element pair.
    - When O2 chains are found (not bonded to surface) they are treated as seperate O2 molecules. 
        - BEWARE!! The indices will not be returned in this insance.

    Args:
        Position (pd.DataFrame): Atomic positions with 'Element', 'x', 'y', and 'z'.
        CellDim (pd.DataFrame): 3x3 matrix defining cell dimensions in angstroms.
        Targets (list of list, optional): List of target molecule compositions (sorted elements).
        Method (str, optional): Bond detection method ('Adjacency' for fixed cutoffs).
        AtomicRadiusTol (float, optional): Scaling factor for bond cutoff distances.

    Returns:
        tuple: 
            - Count (pd.DataFrame): A DataFrame containing the count of detected molecules.
              ['Molecule', 'Indices']
            - GasIndices (pd.DataFrame): A DataFrame listing each molecule and its atom indices.
              ['Molecule', 'Count']
            - BondCounts (pd.DataFrame): A DataFrame containing the number of bonds detected 
              for each element pair. ['Element Pair', 'Bond Count']
    """

    #Ensure Order of molecules is same on repeats
    Targets = [sorted(i) for i in Targets]
    
    GasIndices = pd.DataFrame(columns = ['Molecule', 'Indices'])
    Count = pd.DataFrame({'Molecule' : [tuple(i) for i in Targets], 
                          'Count' : np.zeros(len(Targets))})    

    FracCoords = Position[['x', 'y', 'z']].to_numpy()
    Elements = Position['Element'].to_numpy()
    CellDim = CellDim.to_numpy()
    
    CartDistanceMatrix = MinimumDistancePBCVectorised(FracCoords, CellDim)

    if Method == 'Adjacency':
        
        #This should be taken from csv instead
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
        
        #Add some tolerance to bond finding
        BondCutoffs.update((x, y * AtomicRadiusTol) for x, y in BondCutoffs.items())
        
        N = len(FracCoords)
        BondMatrix = np.zeros((N, N))
        #Build Ajadency Matrix (1 is a bond) by comparing each distance to the relevant cutoff
        
        #First Pass: Build Full Adjacency Matrix
        for i in range(N):
            for j in range(i + 1, N):
                
                pair = (Elements[i], Elements[j])
                
                if pair in BondCutoffs:
                    cutoff = BondCutoffs[pair]
                    
                    if CartDistanceMatrix[i, j] < cutoff:
                        BondMatrix[i, j] = 1
                        BondMatrix[j, i] = 1
        
        #Second Pass: Trim all O-O bonds except Nearest Neighbour O
        
        #Identify which atoms are oxygen
        OIndices = [idx for idx, elem in enumerate(Elements) if elem == 'O']
        
        for i in OIndices:
            
            #Find all oxygens j to which i is currently bonded
            ONeighbors = [j for j in OIndices
                          if j != i and BondMatrix[i, j] == 1]

            #If 0 or 1 neighbors, nothing to prune
            if len(ONeighbors) > 1:
                
                #Among these O–O neighbors, find the single closest one
                OClosest = min(ONeighbors, key=lambda j: CartDistanceMatrix[i, j])

                #Remove the bond to all other oxygen neighbors
                for j in ONeighbors:
                    
                    if j != OClosest:
                        BondMatrix[i, j] = 0
                        BondMatrix[j, i] = 0 
        
        if ReturnBondMatrix:
            return BondMatrix
        
        #Identify all seperated molecules
        MoleculeIndices = FindConnectedSubcomponents(BondMatrix)
        
        #Count total number of molecules found
        for i in MoleculeIndices:
            
            Molecule = sorted([Elements[j] for j in i])
            if Molecule in Targets:
                GasIndices.loc[len(GasIndices.index)] = [tuple(Molecule), i]
                Count.loc[Count['Molecule'] == tuple(Molecule), 'Count'] += 1
            
        #Count Number of bonds found between each element type from BondMatrix
        BondPairs = []
        for i in range(N):
            for j in range(i + 1, N):  #Only upper tri to avoid double count
                if BondMatrix[i, j] == 1:
                    BondPairs.append(tuple(sorted([Elements[i], Elements[j]])))

        #Convert to a DataFrame for bond counts
        BondCounts = pd.DataFrame(pd.Series(BondPairs).value_counts()).reset_index()
        BondCounts.columns = ['Element Pair', 'Bond Count']
        
        return Count, GasIndices, BondCounts


def CalculateGasVolume(Position, CellDim):
    
    '''
    Calculates the volume of the oxidising gas by taking the distance between two 
    outward Zr atoms.
    
    Args:
        Position (pd.DataFrame): Atom positions with 'Element' and fractional 
            coordinates ('x', 'y', 'z').
        CellDim (np.ndarray): A 3x3 matrix representing the cell dimensions in Angstroms.
    Returns:
        GasVolume (float): The value of the volume for gas in m^3
    '''

    #Take only 'x' row and filter for only Zr!    
    PositionZrX = Position['x'].loc[Position['Element'] == 'Zr']        
    PositionZrX = PositionZrX.sort_values()
    
    XDistances = PositionZrX - PositionZrX.shift(1) #Calculate distances between neighbouring Zr
    XDistances.iloc[0] =  1 - (PositionZrX.iloc[-1] + PositionZrX.iloc[0]) #Account for periodic boundary
    
    GasWidth = XDistances.max() * np.linalg.norm(CellDim['x'])
    SurfaceArea = np.linalg.norm(np.cross(CellDim['y'], CellDim['z']))
    
    GasVolume = GasWidth * SurfaceArea * 10 ** -30 #Convert Å³ to m³

    return GasVolume



def CalculateScaleThickness(Position, CellDim, AtomicRadiusTol):
    '''
    Returns the thicknesses of both Oxide scales
    '''
    BondMatrix = FindGases(Position, 
                           CellDim, 
                           AtomicRadiusTol = AtomicRadiusTol,
                           ReturnBondMatrix = True)

    return BondMatrix


def CalculatePartialPressure(O2Molecules: int, 
                             Temp: float,
                             GasVolume: float) -> float:
    
    """
    Computes the partial pressure of an oxidizing gas (O₂) in a simulation cell.

    This function calculates the **partial pressure of oxygen (O₂) gas** inside a 
    simulation cell based on the number of O₂ molecules, temperature, and volume.
    
    **Note:** The number of **O₂ molecules must be given, not individual oxygen atoms.**

    Args:
        O2Molecules (int): The number of O₂ molecules in the simulation cell.
        Temp (float): Temperature in Kelvin.
        GasVolume (float): The value of the volume for gas in m^3

    Returns:
        float: The partial pressure of O₂ gas in bars.

    Notes:
        - **Temperature** is in Kelvin.
        - **Cell volume** is computed from the determinant of `CellDim` and converted to cubic meters.
        - **Partial pressure is calculated using the ideal gas law**:  
        P = {nRT}/{V}
        
        - **Constants Used:**
            - R = 8.314462 J/(mol·K) (Universal gas constant)
            - N_A = 6.022 \times 10^{23} (Avogadros number)
            - 1  atmosphere = 101325  Pascals
        - The output **partial pressure is in bars**.    
    """

    # Physical constants
    R = 8.314462  # Universal gas constant (J/(mol·K))
    Na = 6.022 * 10**23  # Avogadro's number
    atm = 101325  # Atmospheric pressure in Pascals

    PartialPressure = (O2Molecules * R * Temp) / (GasVolume * atm * Na)

    return PartialPressure


def CalculateOxidationRate(N, t, CellDim, PartialPressure,
                           PPConversion = 0.02, Alpha = 0.05):
    
    """
    Computes the oxidation rate (λ) and exact Poisson confidence intervals (95%).

    This function estimates the oxidation rate by measuring **C-containing gases leaving carbides** 
    using a **Poisson distribution**. It also calculates exact confidence intervals for λ.
    
    Args:
        N (int): Observed count of oxidation events.
        t (float): Observation time in femtoseconds (fs).
        CellDim (pd.DataFrame): A (3x3) matrix containing the cell lattice vectors.
        PartialPressure (float): The partial pressure of the oxidizing gas.
        PPConversion (float, optional): Conversion factor for oxidation rate scaling.
        Alpha (float, optional): Significance level (default **0.05** for **95% CI**).

    Returns:
        tuple: (OxRate, LowerBound, UpperBound)
            - OxRate (float): Oxidation rate (scaled by partial pressure & surface area).
            - LowerBound (float): Lower bound of the 95% confidence interval.
            - UpperBound (float): Upper bound of the 95% confidence interval.
    """ 
    

    t = t * 1e-15 #Conversion from fs to s
    PureRate = N / t
    
    CellSurfaceArea = (2 * np.linalg.norm(np.cross(CellDim['y'], CellDim['z'])) 
                       * 10 ** -20)
    
    OxidationRateConversion = PPConversion / CellSurfaceArea / PartialPressure
    
    OxRate = PureRate * OxidationRateConversion

    if N == 0:
        
        UpperBound = -np.log(Alpha) / t 

        # Scale to oxidation rate values
        UpperBound *= OxidationRateConversion
        
        return (OxRate, 0, UpperBound)
    
    else:
        
        ChiSquaredLowerBound = stats.chi2.ppf((Alpha / 2), 
                                              (2 * N))
        
        ChiSquaredUpperBound = stats.chi2.ppf((1 - (Alpha / 2)),
                                              (2 * (N + 1)))
        
        LowerBound = ChiSquaredLowerBound / (2 * t)
        UpperBound = ChiSquaredUpperBound / (2 * t)
        
        LowerBound *= OxidationRateConversion
        UpperBound *= OxidationRateConversion
        
        return (OxRate, LowerBound, UpperBound)






def MaxwellBoltzmannVelocities(Elements, Temperature):
    '''
    Generate random Maxwell-Boltzmann velocities for atoms.

    Args:
        Elements (list): List of element symbols (e.g., ['O', 'C', 'Zr']).
        Temperature (float): Temperature in Kelvin.

    Returns:
        list: List of velocity vectors [vx, vy, vz] for each atom in Å/fs.
    '''
    
    # Atomic masses in amu
    ElementMass = {'O': 15.99, 'C': 12.01, 'Zr': 91.22}
    
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    AMU_to_kg = 1.66054e-27  # Conversion factor from amu to kg
    ms_to_Afs = 1e-5  # Convert m/s to Å/fs

    Velocities = []
    
    for Element in Elements:
        if Element not in ElementMass:
            raise ValueError(f"Element {Element} not found in mass dictionary.")
        
        Mass_kg = ElementMass[Element] * AMU_to_kg  # Convert amu to kg
        Sigma = np.sqrt(k_B * Temperature / Mass_kg)  # Maxwell-Boltzmann standard deviation (Convert to norm.)
        Velocity = np.random.normal(0, Sigma, 3)  # Sample [vx, vy, vz]
        Velocity *= ms_to_Afs  # Convert from m/s to Å/fs
        
        Velocities.append(Velocity)
    
    return Velocities



#--- New Gen functions Here

def FindOptimalCoords(Position: pd.DataFrame,
                      CellDim: pd.DataFrame,
                      n: int = 1,
                      ReturnRadius: bool = False,
                      Seed: int | None = None,
                      MaxIterDE: int = 300,
                      PopSizeDE: int = 15):
    '''
    Finds optimal fractional coordinates for the placement of n new atoms/molecules
    inside a periodic cell, maximising the minimum Cartesian distance to existing
    atoms (and to each other for n ≥ 2).

    Args:
        Position (pd.DataFrame): Atom positions with columns ['Element','x','y','z']
            in fractional coordinates.
        CellDim (pd.DataFrame): A 3x3 DataFrame defining cell dimensions in Ångström.
        n (int): Number of new points to place.
        ReturnRadius (bool): If True, also return the achieved minimum separation.
        Seed (int, optional): Random seed for reproducibility.
        MaxIterDE (int): Maximum iterations for differential evolution (n ≥ 2).
        PopSizeDE (int): Population size multiplier for differential evolution (n ≥ 2).

    Returns:
        OptimalSites (pd.DataFrame): DataFrame of fractional coordinates ('x','y','z').
        If ReturnRadius is True:
            (OptimalSites, Radius)
    '''

    # ---------------- Helper Functions ---------------- #
    def Wrap01(Array):
        '''Map fractional coordinates to [0,1).'''
        return Array - np.floor(Array)

    def PBCDelta(FracA, FracB):
        '''Minimal-image fractional delta under PBC.'''
        Delta = FracA - FracB
        return Delta - np.round(Delta)

    def CartDistFromFracDelta(DeltaFrac, CellDimArray):
        '''Convert fractional delta(s) to Cartesian and compute norms.'''
        RCart = DeltaFrac @ CellDimArray
        return np.linalg.norm(RCart, axis=-1)

    def MinDistanceToExisting(PointsFrac, FracExisting, CellDimArray):
        '''Minimum distance from each proposed point to existing atoms.'''
        if FracExisting.size == 0:
            return np.full(PointsFrac.shape[0], np.inf)
        DFrac = PBCDelta(PointsFrac[:, None, :], FracExisting[None, :, :])
        Dists = CartDistFromFracDelta(DFrac, CellDimArray)
        return Dists.min(axis=1)

    def MinPairwiseAmongNew(PointsFrac, CellDimArray):
        '''Minimum distance among proposed new points themselves.'''
        M = PointsFrac.shape[0]
        if M < 2:
            return np.inf
        DFrac = PBCDelta(PointsFrac[:, None, :], PointsFrac[None, :, :])
        Dists = CartDistFromFracDelta(DFrac, CellDimArray)
        np.fill_diagonal(Dists, np.inf)
        return Dists.min()

    def Radius(PointsFrac, FracExisting, CellDimArray):
        '''Minimum Cartesian distance to existing atoms and among new points.'''
        PointsFrac = Wrap01(PointsFrac)
        DistToExisting = MinDistanceToExisting(PointsFrac, FracExisting, CellDimArray)
        R1 = float(DistToExisting.min()) if DistToExisting.size else np.inf
        R2 = float(MinPairwiseAmongNew(PointsFrac, CellDimArray))
        return min(R1, R2)

    # ---------------- Input Validation ---------------- #
    if not all(c in Position.columns for c in ['x','y','z']):
        raise ValueError("Position must include columns ['x','y','z'].")
    if not all(c in CellDim.columns for c in ['x','y','z']):
        raise ValueError("CellDim must include columns ['x','y','z'].")

    FracExisting = Position[['x','y','z']].to_numpy(float)
    CellDimArray = CellDim[['x','y','z']].to_numpy(float)

    # ---------------- Objective Functions ---------------- #
    def ObjectiveSingle(Point):
        Point = Wrap01(np.asarray(Point, dtype=float))
        return -Radius(Point[None, :], FracExisting, CellDimArray)

    def ObjectiveMultiple(FlatPoints):
        PointsFrac = Wrap01(np.asarray(FlatPoints, dtype=float).reshape(n, 3))
        return -Radius(PointsFrac, FracExisting, CellDimArray)

    # ---------------- Optimisation ---------------- #
    RNG = np.random.default_rng(Seed)

    if n == 1:
        # ---- Single point: cheap multi-start Powell optimisation ---- #
        Seeds = [np.array([0.5, 0.5, 0.5])]
        Seeds += list(RNG.random((7, 3)))  # Random initial guesses

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
        # ---- Multiple points: global optimisation (Differential Evolution) ---- #
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

    # ---------------- Return ---------------- #
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

    Each molecule is centered at the given fractional coordinate from NewSites
    and consists of two O atoms separated by BondLength (Å) along the x-axis.

    Args:
        Position (pd.DataFrame): Existing atom positions with columns ['Element','x','y','z']
            in fractional coordinates.
        CellDim (pd.DataFrame): 3x3 DataFrame defining lattice vectors in Ångström.
        NewSites (pd.DataFrame): Fractional coordinates ('x','y','z') where O2 molecules
            should be centered.
        BondLength (float): O–O bond length in Ångström.

    Returns:
        pd.DataFrame: Updated Positions DataFrame including added O atoms.
    '''

    # Validate inputs
    for df, name in [(Position, "Positions"), (CellDim, "CellDim"), (NewSites, "NewSites")]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
    if not all(c in Position.columns for c in ['x','y','z']):
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
    UpdatedPositions = pd.concat([Position, NewAtomsDF], ignore_index=True)

    return UpdatedPositions

# %%
