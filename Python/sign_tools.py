import numpy as np
import sys
from scipy.linalg import lu
from collections import defaultdict



N = 0


def transpose_permutations(permutations):
    row_number = len(permutations)
    col_number = len(permutations[0])
    PermMatrix = np.zeros((col_number , row_number))
    for i in range(row_number):
        for j in range(col_number):
            PermMatrix[j][i] = permutations[i][j]

    return PermMatrix

import numpy as np

def mod2_nullspace(binary_vectors):
    """
    Finds the null space of a set of binary vectors mod 2.

    Parameters:
        binary_vectors (list of list of int]): A list of binary vectors.

    Returns:
        list of list of int: The null space vectors in mod 2 as a list of binary vectors.
    """
    # Convert to a numpy array and ensure all entries are mod 2
    A = np.array(transpose_permutations(binary_vectors), dtype=int) % 2
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



def parse_pauli_file(filename):
    global N
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

    return coefficients, binary_vectors

def process_pauli_terms(coefficients, binary_vectors):
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

    # Group terms by their Pauli X action
    grouped_terms = defaultdict(lambda: defaultdict(complex))
    for coeff, vector in zip(coefficients, binary_vectors):
        x_action = tuple(vector[:N])  # First N indices
        z_action = tuple(vector[N:])  # Last N indices
        grouped_terms[x_action][z_action] += coeff  # Add coefficients for matching Z-action

    # Build permutations and diagonals
    permutations = []
    diagonals = []

    for x_action, z_terms in grouped_terms.items():
        permutations.append(list(x_action))
        coeff_list = []
        z_vectors = []
        for z_action, coeff in z_terms.items():
            coeff_list.append(coeff)
            z_vectors.append(list(z_action))
        diagonals.append((coeff_list, z_vectors))  # Store as tuple (coefficients, binary vectors)

    return permutations, diagonals


# Example usage:

filename = sys.argv[1]
coefficients, binary_vectors = parse_pauli_file(filename)
print( "Coefficients: " , coefficients)
print( "Binary Vectors: " , binary_vectors)

permutations , diagonals = process_pauli_terms(coefficients , binary_vectors)
PermMatrix = transpose_permutations(permutations)

print( "Permutations are: " , permutations )
print( "Diagonals are: " , diagonals )

#print("Null space mod 2:", null_space)
print(" The cycles are : " , mod2_nullspace(permutations))