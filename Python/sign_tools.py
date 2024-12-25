import numpy as np
import sys
from scipy.linalg import lu
from collections import defaultdict
import itertools
import cmath
import random

N = 0

def transpose_permutations(permutations):
    row_number = len(permutations)
    col_number = len(permutations[0])
    PermMatrix = np.zeros((col_number , row_number))
    for i in range(row_number):
        for j in range(col_number):
            PermMatrix[j][i] = permutations[i][j]
    return PermMatrix

def binary_to_indices(binary_vector):
    indices = []
    for i in range(len(binary_vector)):
        if binary_vector[i] == 1:
            indices.append(i)
    return indices

def indices_to_binary(IndicesVector, NumOfSpins):
    """
    Converts a list of indices into a binary vector of given length.

    Parameters:
        indices (list of int): Indices where the binary vector should have 1s.
        number_of_spins (int): The length of the resulting binary vector.

    Returns:
        list of int: Binary vector with 1s at the specified indices.
    """
    BinaryVector = [0] * NumOfSpins  # Initialize with zeros
    for index in IndicesVector:
        if 0 <= index < NumOfSpins:  # Ensure index is within bounds
            BinaryVector[index] = 1
        else:
            raise ValueError(f"Index {index} is out of bounds for vector of length {NumOfSpins}")
    return BinaryVector


def mod2_nullspace(BinaryVectors):
    """
    Finds the null space of a set of binary vectors mod 2.

    Parameters:
        binary_vectors (list of list of int]): A list of binary vectors.

    Returns:
        list of list of int: The null space vectors in mod 2 as a list of binary vectors.
    """
    # Convert to a numpy array and ensure all entries are mod 2
    A = np.array(transpose_permutations(BinaryVectors), dtype=int) % 2
    n_rows, n_cols = A.shape

    # Augment the matrix with an identity matrix to track nullspace
    augmented_matrix = np.hstack((A.T, np.eye(n_cols, dtype=int)))

    # Perform Gaussian elimination over GF(2)
    for col in range(min(n_rows, n_cols)):
        # Find a pivot row
        pivot_row = -1
        for row in range(col, n_cols):
            if augmented_matrix[row, col] == 1:
                pivot_row = row
                break

        if pivot_row == -1:
            # No pivot in this column, move to the next
            continue

        # Swap rows to move the pivot row to the top
        augmented_matrix[[col, pivot_row]] = augmented_matrix[[pivot_row, col]]

        # Eliminate all other rows in this column
        for row in range(n_cols):
            if row != col and augmented_matrix[row, col] == 1:
                augmented_matrix[row] = (augmented_matrix[row] + augmented_matrix[col]) % 2

    # Extract the null space from the right half of the matrix
    null_space = []
    for row in range(n_cols):
        if np.all(augmented_matrix[row, :n_rows] == 0):  # Row corresponds to a null space vector
            null_space.append(augmented_matrix[row, n_rows:].tolist())

    return null_space

def evaluate_diagonal(State , ZStringIndices):
    """
    
    The input state is a binary np.array, i.e. a computational basis state
    The input z_string_indices is an np.array of integers representing which particle there is a pauli-Z action on.
    
    """
    diag = 1.0
    for particle_no in ZStringIndices:
        diag *= (-1.0)**State[particle_no]
    return diag

def permute_state(State , XStringIndices):
    """
    
    The input state is a binary np.array, i.e. a computational basis state
    The input z_string_indices is an np.array of integers representing which particle there is a pauli-X action on.
    
    """
    for particle_no in XStringIndices:
        State[particle_no] = (State[particle_no] + 1)%2 

def single_cycle_weight(InitialState , CycleDiags , CyclePerms):
    """

    The input InitialState is a binary np.array specifying a computational state |z>
    The input cycle_diags is the diagonals for each cycle_perm permutation. Each diagonal term is represented by a linear combination of Z-string operators. Each Z-string operator is a vector of integers representing
        which particle there is a pauli-Z action on.
    The input cycle_perms are binary vectors specifying the string of pauli-X. Each X-string operator is a vector of integers representing which particle there is a pauli-X action on.

    """
    weight = 1.0
    state = InitialState
    for i in range(len(CycleDiags)):
        diag_j = 0.0
        for j in range(len(CycleDiags[i][0])):
            diag_j += CycleDiags[i][0][j]*evaluate_diagonal(state , CycleDiags[i][1][j])
        weight *= diag_j
        permute_state(state , CyclePerms[i])
    return weight


def parse_pauli_file(filename):
    """

    Parses the input file and returns a list of coefficients and corresponding binary vectors.

    """

    coefficients = []
    binary_vectors = []

    # Read the file once to determine the largest particle_number (i.e., N)
    with open(filename, 'r') as file:
        max_particle = 0
        for line in file:
            tokens = line.split()
            particle_numbers = [int(tokens[i]) for i in range(1, len(tokens), 4)]
            max_particle = max(max_particle, *particle_numbers)

    N = max_particle  # Total number of particles in the system

    print(f'The total number of particles are: {N}')

    # Read the file again to construct the binary vectors and coefficients
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.split()
            coefficient = complex(tokens[0])
            binary_vector = np.zeros(2 * N, dtype=int)

            # Process each operator in the line
            for i in range(1, len(tokens), 4):
                particle_number = int(tokens[i]) - 1  # Convert to zero-based index
                pauli_matrix = tokens[i + 1]
                power = int(tokens[i + 2])  # Not used explicitly in this code
                spin = tokens[i + 3]       # Not used explicitly in this code

                if pauli_matrix in ['1', 'X']:
                    # Pauli-X
                    binary_vector[particle_number] = (binary_vector[particle_number] + 1) % 2
                elif pauli_matrix in ['2', 'Y']:
                    # Pauli-Y
                    binary_vector[particle_number] = (binary_vector[particle_number] + 1) % 2
                    binary_vector[N + particle_number] = (binary_vector[N + particle_number] + 1) % 2
                    coefficient *= 1j
                elif pauli_matrix in ['3', 'Z']:
                    # Pauli-Z
                    binary_vector[N + particle_number] = (binary_vector[N + particle_number] + 1) % 2

            coefficients.append(coefficient)
            binary_vectors.append(binary_vector)

    return coefficients, binary_vectors , N

def convert_diagonal_to_indices(DiagonalsBinary):
    DiagonalsIndices = DiagonalsBinary.copy()
    for i in range(len(DiagonalsIndices)):
        term = DiagonalsIndices[i]
        z_string_indices = []
        for Zstring in term[1]:
            z_string_indices.append(binary_to_indices(Zstring))
        DiagonalsIndices[i] = (term[0] , z_string_indices)
    return DiagonalsIndices

def convert_indices_to_diagonal(DiagonalsIndices, NumOfSpins):
    """
    Converts a list of coefficients and indices for the action of Pauli-Z 
    into coefficients and binary vectors.

    Parameters:
        DiagonalsIndices (list of tuples): Each tuple contains:
            - List of coefficients (list of complex or float).
            - List of lists of indices where Pauli-Z acts.
        number_of_spins (int): The length of each binary vector.

    Returns:
        list of tuples: Each tuple contains:
            - List of coefficients.
            - List of binary vectors representing Pauli-Z actions.
    """
    DiagonalsBinary = DiagonalsIndices.copy()
    for i in range(len(DiagonalsBinary)):
        term = DiagonalsBinary[i]
        z_string_binaries = []
        for Zstring_indices in term[1]:
            z_string_binaries.append(indices_to_binary(Zstring_indices, NumOfSpins))
        DiagonalsBinary[i] = (term[0], z_string_binaries)
    return DiagonalsBinary


def convert_binary_cycles_to_indices(NullspaceBinary , PermutationIndices , OffdiagonalIndices , NumOfParticles):
    NullspaceIndices = []
    OffdiagonalCycles = []
    M = len(PermutationIndices)
    for j in range(len(NullspaceBinary)):
        NullPermj = []
        NullDiagj = []
        for i in range(M):
            if NullspaceBinary[j][i]==1:
                NullPermj.append(PermutationIndices[i])
                NullDiagj.append(OffdiagonalIndices[i])
        NullspaceIndices.append(NullPermj)
        OffdiagonalCycles.append(NullDiagj)

    return NullspaceIndices , OffdiagonalCycles

def process_pauli_terms(Coefficients, BinaryVectors , NumOfParticles):

    """

    Processes the Pauli terms to group by their Pauli X action (first N binary coefficients),
    and combines terms with matching Z-action vectors.

    Parameters:
        coefficients (list of complex): Coefficients from the Pauli file.
        binary_vectors (list of numpy arrays): Binary vectors from the Pauli file.

    Returns:
        list: Permutations - unique binary vectors of N indices representing the Pauli X action.
        list: Diagonals - tuples of (list of coefficients, list of Z-action binary vectors).

    """
    N = NumOfParticles
    # Group terms by their Pauli X action
    grouped_terms = defaultdict(lambda: defaultdict(complex))
    for coeff, vector in zip(Coefficients, BinaryVectors):
        x_action = tuple(vector[:N])  # First N indices
        z_action = tuple(vector[N:])  # Last N indices
        grouped_terms[x_action][z_action] += coeff  # Add coefficients for matching Z-action

    # Build permutations and diagonals
    permutations = []
    off_diagonals = []
    diagonals = []

    for x_action, z_terms in grouped_terms.items():
        perm = np.array(x_action)
        coeff_list = []
        z_vectors = []
        for z_action, coeff in z_terms.items():
            coeff_list.append(coeff)
            z_vectors.append(list(z_action))
        if np.all(perm == 0):
            diagonals.append((coeff_list, z_vectors))
        else:
            permutations.append(list(x_action))
            off_diagonals.append((coeff_list, z_vectors))  # Store as tuple (coefficients, binary vectors)
    return permutations, off_diagonals , diagonals

def generate_permutations(arr):
    AllPermutations = []
    All = list(itertools.permutations(arr))
    for perm in All:
        AllPermutations.append(list(perm))
    return AllPermutations

def is_cyclic_equivalent(perm1, perm2):
    n = len(perm1)
    return any(perm1[i:] + perm1[:i] == perm2 for i in range(n))

def filter_cyclic_equivalents(permutations):
    unique_cyclic = []
    for perm in permutations:
        if not any(is_cyclic_equivalent(perm, unique) for unique in unique_cyclic):
            unique_cyclic.append(perm)
    return unique_cyclic

def generate_cyclic_permutations(arr):
    AllPerms = []
    for cyc in arr:
        AllPerms += generate_permutations(cyc)
    UniqueCyclic = filter_cyclic_equivalents(AllPerms)
    return UniqueCyclic

def find_perm_index(P , AllPs):
    for i in range(len(AllPs)):
        if np.array_equal(P , AllPs[i]):
            return i
    return "There is a problem: The permutation was not found!"

def generate_cycle_diagonals(Cycle , AllPerms , AllDs):
    CycleDs = []
    for Perm in Cycle:
        PermIndex = find_perm_index(Perm , AllPerms)
        CycleDs.append(AllDs[PermIndex])
    
    return CycleDs

def generate_cyclic_permutations_with_offdiagonals(PermCycs , AllPs , AllDs):
    # Generate all permutations of the paired array
    UniquePermutations = generate_cyclic_permutations(PermCycs)
    UniqueDiagonals = [generate_cycle_diagonals(Cycle , AllPs , AllDs) for Cycle in UniquePermutations]
    return UniquePermutations , UniqueDiagonals


def int_to_binary_array(n , N):
    binary_string = bin(n)[2:]  # Convert to binary and remove '0b' prefix
    binary_array = [int(bit) for bit in binary_string]  # Convert to an array of integers
    return binary_array[::-1] + [0]*(N-len(binary_array))

def total_cost_of_cycle(CycleDiags , CyclePerm , TotalParticles):
    N = TotalParticles
    SignFactor = (-1.0)**len(CycleDiags)
    cost = 0.0
    for i in range(2**N):
        State = int_to_binary_array(i , N)
        weight = single_cycle_weight(State , CycleDiags , CyclePerm)*SignFactor
        r , theta = cmath.polar(weight)
        #if r > 1E-6:
        #    print(f'The state is {State} and the permutation cycle is {CyclePerm}')
        #    print(f'The r is {r} and theta is {theta}')
        cost += r*(1.0-np.cos(theta))
    return cost

#def add_pair_to_cycle(Cycle , CycleDs , Permutation , Ds):
def add_pair_to_cycle(Cycle , CycleDs , Permutation , Ds):
    """
    
    The input Cycle is a length three cycle generator and CycleDs are the corresponding diagonal terms for the list of permutations in Cycle.
    This function adds pairs of Permutation in between the permutations appearing in the cycle so that no two identical permutations are next to each other.

    """
    # We just need to add the diagonals accordingly ... !!!!!!!!
    HigherCycles = []
    HigherCycleDs = []
    # Iterate over all positions to insert the two identical Permutations
    for i in range(len(Cycle) + 1):
        for j in range(i+1, len(Cycle) + 1):
            # Create a new list by inserting the Permutations
            NewCycle = Cycle[:i] + [Permutation] + Cycle[i:j] + [Permutation] + Cycle[j:]
            NewDs = CycleDs[:i] + [Ds] + CycleDs[i:j] + [Ds] + CycleDs[j:]
            # Check for the condition: no two neighboring elements are identical
            NearNeighbor = [not np.array_equal(NewCycle[k] , NewCycle[k+1]) for k in range(len(NewCycle)-1)]
            if all(NearNeighbor):
                HigherCycles.append(NewCycle)
                HigherCycleDs.append(NewDs)
    
    return HigherCycles , HigherCycleDs

def generate_higher_cycles(PermCycInds, OffDsCycInds , PermIndices , OffDsIndices):
    HighCycles = []
    HighOffDiags = []

    # Generating length 4 cycles:
    # Build length 4 cycles:
        # Recipe:
            # Take any two different permutation, e.g. P1 and P2 , then the only contributing cycle is P1 P2 P1 P2
            # Do this for all unique pairs

    for i in range(len(PermIndices)):
        for j in np.arange(i+1 , len(PermIndices)):
            HighCycles.append([PermIndices[i] , PermIndices[j] , PermIndices[i] , PermIndices[j]])
            HighOffDiags.append([OffDsIndices[i] , OffDsIndices[j] , OffDsIndices[i] , OffDsIndices[j]])
    
    # Generating length 5 cycles:
    # Build length 5 cycles:
        # Recipe: 
            # Take any length 3 fundamental cycle generator from the nullspace
            # Add pairs of permutations in a non-trivial order, e.g. lets say we have a fund cycle generator of P1 P2 P3 = 1, then
            #   take any other permutation P4, and create length 5 cycle generators by creating permutations such as
            #   P4 P1 P2 P4 P3 , P4 P1 P2 P3 P4 , ... , P1 P4 P2 P4 P3 , P1 P2 P4 P3 P4 , ...

    for FundCyc , FundCycDs in zip(PermCycInds , OffDsCycInds):
        if len(FundCyc) < 4:
            for Perm , Ds in zip(PermIndices , OffDsIndices):
                CycPs , CycDs = add_pair_to_cycle(FundCyc , FundCycDs , Perm , Ds)
                HighCycles += CycPs
                HighOffDiags += CycDs
    return HighCycles , HighOffDiags


def get_all_cycles_from_file(filename):
    coefficients, binary_vectors , NumOfParticles = parse_pauli_file(filename)
    #print(f'binary vectors are {binary_vectors}')
    permutations_binary , offdiagonals_binary , pure_diagonals = process_pauli_terms(coefficients , binary_vectors , NumOfParticles)
    #print(f'The permutation binary is {permutations_binary}')
    PermutationIndices = []

    Cycles_q = {}

    for permutation in permutations_binary:
        PermutationIndices.append(binary_to_indices(permutation))
    OffDiagonalsIndices = convert_diagonal_to_indices(offdiagonals_binary)
    NullSpace = mod2_nullspace(permutations_binary)
  
    PermCycleIndices , OffDiagCycleIndices = convert_binary_cycles_to_indices(NullSpace , PermutationIndices , OffDiagonalsIndices , NumOfParticles)
    FundCyclesIndices , FundCycOffDiagsIndices = generate_cyclic_permutations_with_offdiagonals(PermCycleIndices , PermutationIndices , OffDiagonalsIndices)

    # Generate all fundamental cycles up to length 5:
    # Take in AllCyclesIndices, AllOffDiagsIndices, PermutationIndices , OffDiagonalsIndices and output all cycles of length > 2
    HighCycles , HighOffDiags = generate_higher_cycles(FundCyclesIndices , FundCycOffDiagsIndices , PermutationIndices , OffDiagonalsIndices)

    AllCycles = FundCyclesIndices + HighCycles
    AllCycOffDs = FundCycOffDiagsIndices + HighOffDiags

    for i in range(len(AllCycles)):
        q = len(AllCycles[i])
        if q not in Cycles_q :
            Cycles_q[q] = {'Permutation Cycles':[] , 'Diagonal Cycles': []}
        Cycles_q[q]['Permutation Cycles'].append(AllCycles[i])
        Cycles_q[q]['Diagonal Cycles'].append(AllCycOffDs[i])

    return Cycles_q , NumOfParticles


def total_hamiltonian_cost(AllCycleDiags , AllCyclePerms , NumOfParticles):
    cost = 0.0
    for i in range(len(AllCyclePerms)):
        cost += total_cost_of_cycle(AllCycleDiags[i] , AllCyclePerms[i] , NumOfParticles)
    return cost

# ========================= Clifford rotation functions ====================================

def permutation_found(Perm , PermList):
    for i in range(len(PermList)):
        if Perm == PermList[i]:
            return True , i
        
    return False , -1

def Sbinary_xvec_zvec_onspins(Xvec , Zvec , Spins):
    """ 
    binary operations on the X and Z vectors inducing a Hadamard rotation

    """
    Xvecfinal = Xvec.copy()
    Zvecfinal = Zvec.copy()
    phase = 1.0
    for spin in Spins:
        BinXZ = ( Xvec[spin] + Zvec[spin] ) % 2
        Zvecfinal[spin] = BinXZ
        phase *= ((1.0j)**(Xvec[spin]))
    return Xvecfinal , Zvecfinal , phase

def Hbinary_xvec_zvec_onspins(Xvec , Zvec , Spins):
    """ 
    binary operations on the X and Z vectors inducing a S rotation
    
    """
    Xvecfinal = Xvec.copy()
    Zvecfinal = Zvec.copy()
    phase = 1.0
    for spin in Spins:
        Xvecfinal[spin] = Zvec[spin]
        Zvecfinal[spin] = Xvec[spin]
        phase *= ((-1)**(Xvec[spin] & Zvec[spin]))
    
    return Xvecfinal , Zvecfinal , phase

def apply_single_body(AllPerms, AllDiags , Spins , SingleBodyType):
    """

    Apply single body rotations on specified Spins.

    """
    AllPermsTransformed = []
    AllDiagsTransformed = []

    for i in range(len(AllDiags)):
        for j in range(len(AllDiags[i][1])):
            if SingleBodyType in ['H' , 'Hadamard' , 'hadamard']:
                NewPermutation , NewDiagonal , phase = Hbinary_xvec_zvec_onspins(AllPerms[i] , AllDiags[i][1][j] , Spins)
            elif SingleBodyType in ['S' , 'Sgate' , 'S-gate']:
                NewPermutation , NewDiagonal , phase = Sbinary_xvec_zvec_onspins(AllPerms[i] , AllDiags[i][1][j] , Spins)
            else:
                raise ValueError("The specified rotation does not exist.. There are only Hadamard and S-gate single body rotations available.")

            PermFound , index = permutation_found(NewPermutation , AllPermsTransformed)
            coefficient = phase*AllDiags[i][0][j]
            if not PermFound:
                AllPermsTransformed.append(NewPermutation)
                AllDiagsTransformed.append([[coefficient] , [NewDiagonal]])
            else:
                AllDiagsTransformed[index][0].append(coefficient)
                AllDiagsTransformed[index][1].append(NewDiagonal)

    return AllPermsTransformed, AllDiagsTransformed

def CNOT_xvec_zvec_onspins(Xvec , Zvec , CNOTPairs):
    """
    This functino applies CNOT on the binary x and z vectors of pauli string
    """
    Xvecfinal = Xvec.copy()
    Zvecfinal = Zvec.copy()
    for pair in CNOTPairs:
        control = pair[0]
        target = pair[1]
        Xvecfinal[target] = (Xvecfinal[control] + Xvecfinal[target])%2
        Zvecfinal[control] = (Zvecfinal[control] + Zvecfinal[target])%2

    return Xvecfinal , Zvecfinal

def apply_CNOT(AllPerms, AllDiags , CNOTPairs):
    """
    Apply CNOT rotations on specified pair of Spins.
    """
    AllPermsTransformed = []
    AllDiagsTransformed = []

    for i in range(len(AllDiags)):
        for j in range(len(AllDiags[i][1])):
            NewPermutation , NewDiagonal = CNOT_xvec_zvec_onspins(AllPerms[i] , AllDiags[i][1][j] , CNOTPairs)
            PermFound , index = permutation_found(NewPermutation , AllPermsTransformed)
            coefficient = AllDiags[i][0][j]
            if not PermFound:
                AllPermsTransformed.append(NewPermutation)
                AllDiagsTransformed.append([[coefficient] , [NewDiagonal]])
            else:
                AllDiagsTransformed[index][0].append(coefficient)
                AllDiagsTransformed[index][1].append(NewDiagonal)
    return AllPermsTransformed, AllDiagsTransformed


# This function needs to be updated! 
def Toff_xvec_zvec_onspins(Xvec , Zvec , ToffTruple):
    """
    This functino applies CNOT on the binary x and z vectors of pauli string
    """
    Xvecfinal = Xvec.copy()
    Zvecfinal = Zvec.copy()

    control1 = ToffTruple[0]
    control2 = ToffTruple[1]
    target = ToffTruple[2]
    Xvecfinal[target] = (Xvecfinal[control1] + Xvecfinal[control2] + Xvecfinal[target])%2
    Zvecfinal[target] = (Zvecfinal[control1] + Zvecfinal[control2] + Zvecfinal[target])%2

    return Xvecfinal , Zvecfinal

def apply_Toff(AllPerms, AllDiags , ToffTruple):
    """
    Apply Toffolli rotations on specified pair of Spins.
    """
    AllPermsTransformed = []
    AllDiagsTransformed = []

    for i in range(len(AllDiags)):
        for j in range(len(AllDiags[i][1])):
            NewPermutation , NewDiagonal = Toff_xvec_zvec_onspins(AllPerms[i] , AllDiags[i][1][j] , ToffTruple)
            PermFound , index = permutation_found(NewPermutation , AllPermsTransformed)
            coefficient = AllDiags[i][0][j]
            if not PermFound:
                AllPermsTransformed.append(NewPermutation)
                AllDiagsTransformed.append([[coefficient] , [NewDiagonal]])
            else:
                AllDiagsTransformed[index][0].append(coefficient)
                AllDiagsTransformed[index][1].append(NewDiagonal)
    return AllPermsTransformed, AllDiagsTransformed

def generate_random_spins(N):
    """
    Generate a random array of unique integers from 0 to N-1.

    Parameters:
        N (int): The range of integers to choose from (0 to N-1).

    Returns:
        list: A shuffled list of unique integers from 0 to N-1.
    """
    if N < 1:
        raise ValueError("N must be at least 1 to generate a random array.")

    array = list(range(N))
    random.shuffle(array)
    return array

def generate_random_pairs(N):
    """
    Generate a random set of pairs of unique non-identical integers from 0 to N-1.

    Parameters:
        N (int): The range of integers to choose from (0 to N-1).

    Returns:
        set: A set of tuples, where each tuple contains two unique integers.
    """
    if N < 2:
        raise ValueError("N must be at least 2 to generate unique pairs.")

    # Calculate the maximum possible number of unique pairs
    max_pairs = N * (N - 1) // 2

    # Choose a random multiple of 2 that does not exceed max_pairs
    num_pairs = random.randint(1, max_pairs // 2) * 2

    pairs = set()

    while len(pairs) < num_pairs:
        pair = tuple(sorted(random.sample(range(N), 2)))
        pairs.add(pair)

    return pairs


def generate_random_triple(N):
    """
    Generate a random set of triples of non-identical integers from 0 to N-1.

    Parameters:
        N (int): The range of integers to choose from (0 to N-1).

    Returns:
        set: A set of tuples, where each tuple contains three unique integers.
    """
    if N < 3:
        raise ValueError("N must be at least 3 to generate unique triples.")

    return random.sample(range(N), 3)

def apply_random_transformation(Probabilities , AllPerms , AllDiags , NumOfParticles):
    """
    The input 'Probabilities' specifies the probability of single, two, or three body rotations
    """
    ProbOneBody , ProbTwoBody , ProbThreeBody = Probabilities
    p = random.random()
    if p < ProbOneBody:
        # Apply single body rotation
        p1 = random.random()
        Spins = generate_random_spins(NumOfParticles)
        if p1 <= 0.5:
            # Apply hadamard at a randomly picked spin
            RotationType = 'H'
            print(f'Hadamard gates on spins {Spins} applied!')
            print(' ')
        else:
            # Apply S gate at a randomly picked spin
            RotationType = 'S'
            print(f'S-gates on spins {Spins} applied!')
            print(' ')
        AllPermsT , AllDiagsT = apply_single_body(AllPerms, AllDiags , Spins , RotationType)
    elif p < ProbOneBody + ProbTwoBody:
        # Apply two body rotation (CNOT)
        # randomly pick a tuple (i , j) and apply CNOT with control on i spin , and target at j spin...
        
        CNOTPairs = generate_random_pairs(NumOfParticles)
        print(f'CNOT gate applied on the CNOT pairs {CNOTPairs}') # The first term of the pair is the control spin and the second is the target!
        AllPermsT , AllDiagsT = apply_CNOT(AllPerms, AllDiags , CNOTPairs)
    else: 
        # Apply two body rotation (CCNOT)
        # randomly pick a tuple (i , j , k) and apply CCNOT with control on i and j spin , and target at k spin...
        ToffTriple = generate_random_triple(NumOfParticles)
        print(f'CNOT gate applied on the Toffoli triples {ToffTriple}') # The first term two spins are the control spin and the third is the target!
        AllPermsT , AllDiagsT = apply_Toff(AllPerms, AllDiags , ToffTriple)

    return AllPermsT , AllDiagsT

# ======================== Writing the output files from the permutation data ==============================
import numpy as np

def generate_pauli_file_from_pmr_data(output_filename, permutations, off_diagonals, diagonals):
    """
    Generates a text file from the given permutations and diagonal terms, handling complex coefficients.

    Parameters:
        output_filename (str): The name of the output file.
        permutations (list): List of binary vectors representing Pauli-X action.
        off_diagonals (list): List of tuples (list of coefficients, list of binary Z-action vectors).
        diagonals (list): List of tuples (list of coefficients, list of binary Z-action vectors).

    """
    def get_pauli_action(x_val, z_val):
        """
        Determines the Pauli action based on X and Z values.

        Parameters:
            x_val (int): Value in the X part of the binary vector.
            z_val (int): Value in the Z part of the binary vector.

        Returns:
            str: Pauli action ('X', 'Y', or 'Z').
        """
        if x_val == 1 and z_val == 0:
            return 'X'
        elif x_val == 1 and z_val == 1:
            return 'Y'
        elif x_val == 0 and z_val == 1:
            return 'Z'
        else:
            return None  # No Pauli action (identity)

    def format_line(coefficient, binary_vector):
        """
        Formats a line for the output file.

        Parameters:
            coefficient (complex): Coefficient of the Pauli term.
            binary_vector (list): Combined binary vector (X and Z actions).

        Returns:
            str: Formatted line for the output file.
        """
        N = len(binary_vector) // 2
        x_part = binary_vector[:N]
        z_part = binary_vector[N:]

        adjusted_coefficient = coefficient
        line = []

        for i, (x, z) in enumerate(zip(x_part, z_part)):
            pauli_action = get_pauli_action(x, z)
            if pauli_action:
                if pauli_action == 'Y':
                    adjusted_coefficient *= -1.0j
                #line.extend([f"{i + 1}", pauli_action, "1", "2"])
                line.extend([f"{i + 1}", pauli_action])
        
        # Format the coefficient with both real and imaginary parts
        coeff_str = f"{adjusted_coefficient.real:.6f}" + (f"{adjusted_coefficient.imag:+.6f}j" if adjusted_coefficient.imag != 0 else "")
        return f"{coeff_str} " + " ".join(line)

    with open(output_filename, 'w') as file:
        # Process off-diagonal terms
        for perm, (coeff_list, z_vectors) in zip(permutations, off_diagonals):
            for coeff, z_vec in zip(coeff_list, z_vectors):
                combined_vector = np.concatenate((perm, z_vec))
                line = format_line(coeff, combined_vector)
                file.write(line + '\n')

        # Process diagonal terms
        #for (coeff_list, z_vectors) in diagonals:
        (coeff_list , z_vectors) = diagonals
        for coeff , z_vec in zip(coeff_list, z_vectors):
            combined_vector = np.zeros(len(z_vec) * 2 , dtype=int)
            combined_vector[len(z_vec):] = z_vec  # Only Z actions for diagonals
            line = format_line(coeff, combined_vector)
            file.write(line + '\n')