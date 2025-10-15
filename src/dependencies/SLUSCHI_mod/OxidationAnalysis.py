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


def FindNewCoord(Positions, CellDim):
    
    '''
    Finds optimal Coordinates for the placement of a single new atom/molecule using 
    a dual annealing optimiser. 
    For multiple points global optimiser should be used instead.
    
    Args:
        Position (pd.DataFrame): Atom positions with 'Element' and fractional 
            coordinates ('x', 'y', 'z').
        CellDim (pd.DataFrame): A 3x3 DataFrame defining cell dimensions in angstroms.
    
    Returns:
        OptimalCoords (np.array): Fractional coordinates of point furthest away
            from each other atom. 
    '''
    
    #Fractional boundaries applied
    bounds = [(0, 1), (0, 1), (0, 1)]
    
    def ObjectiveFunction_MinimumDistance(NewPoint, FracCoords, CellDim):
        #Finds the cartesian distance to the nearest neighbour from proposed point.
        #Can be optimised by not calculating whole distance matrix.
        #For current workload optimisation not needed (1-2 seconds).
        
        #Last point will always be proposed point for optimiser.    
        FracCoords = np.append(FracCoords, [NewPoint], axis = 0)
        NearestNeighbour = -np.sort(MinimumDistancePBCVectorised(FracCoords, 
                                                                 CellDim)[-1])[1]

        return NearestNeighbour
    
    FracCoords = Positions[['x', 'y', 'z']].to_numpy()
    CellDim = CellDim.to_numpy()
    
    results = opt.dual_annealing(func = ObjectiveFunction_MinimumDistance, 
                                 bounds = bounds,
                                 args = (FracCoords, CellDim),
                                 initial_temp = 2500,
                                 maxiter = 500,
                                 )
    
    return results.x


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



    
# %%

