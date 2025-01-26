import numpy as np
from scipy.linalg import eigvals
from itertools import product

I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)

Z = np.array([[1,  0],
              [0, -1]], dtype=complex)

pauli_dict = {
    'I': I,
    'X': X,
    'Y': Y,
    'Z': Z
}

def are_unitarily_similar(A, B):
    """
    Determines if two square matrices A and B are unitarily similar by comparing eigenvalues.

    Parameters:
    A (np.ndarray): A square matrix.
    B (np.ndarray): Another square matrix.

    Returns:
    bool: True if A and B are unitarily similar, False otherwise.
    """
    # Check if matrices are square and of the same size
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        return False

    # Compute eigenvalues
    eigvals_A = np.sort(eigvals(A))
    eigvals_B = np.sort(eigvals(B))

    # Compare eigenvalues, allowing for some numerical tolerance
    return np.allclose(eigvals_A, eigvals_B, atol=1e-5)

def convert_to_format(input_data, n):
    # Initialize the result list
    result = []

    # Parse each line of the input data
    for line in input_data.strip().split("\n"):
        parts = line.split()
        coefficient = complex(parts[0])  # First value is the coefficient

        # Create a list of "I"s (identity operators) with length 5
        operators = ["I"] * n

        # Update the operators based on the parsed data
        for i in range(1, len(parts), 2):
            position = int(parts[i]) - 1  # Convert to 0-based index
            operator = parts[i + 1]       # Get the operator (X, Y, Z)
            operators[position] = operator

        # Append the tuple to the result list
        result.append((coefficient, *operators))

    return result

def tensor_product(pauli_list):
    """
    Given a list of single-qubit Pauli labels (e.g. ['X','I','Z']),
    return the corresponding multi-qubit operator as a NumPy array.
    """
    # Start with the first Pauli
    result = pauli_dict[pauli_list[0]]
    # Sequentially take the Kronecker product with the next Pauli
    for p in pauli_list[1:]:
        result = np.kron(result, pauli_dict[p])
    return result

def pauli_sum_to_matrix(pauli_terms):
    """
    Given a list of terms in the form:
       [
         (coefficient, 'X', 'X', 'I', 'I', ...),
         (coefficient, 'Y', 'Y', 'I', 'I', ...),
         ...
       ]
    return the full matrix of the sum of these tensor products.
    """
    n_qubits = len(pauli_terms[0]) - 1
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    # Build each term and add it
    for term in pauli_terms:
        coef = term[0]
        paulis = term[1:]
        op_matrix = tensor_product(paulis)
        H += coef * op_matrix
    
    return H

def generate_single_qubit_paulis():
    """
    Returns a list of (label, matrix) for single-qubit Pauli operators:
    I, X, Y, Z.
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [
        ("I", I),
        ("X", X),
        ("Y", Y),
        ("Z", Z),
    ]

def generate_n_qubit_paulis(n):
    """
    Yields (label, matrix) for all n-qubit Pauli operators (4^n in total).
    The label is a string of length n over {I, X, Y, Z}.
    The matrix is the corresponding 2^n x 2^n NumPy array.
    """
    single_qubit_paulis = generate_single_qubit_paulis()
    for combo in product(single_qubit_paulis, repeat=n):
        # combo is a tuple like (("X", X_matrix), ("Z", Z_matrix), ...)
        label = ""
        for i, c in enumerate (combo):
            if c[0] != "I":
                label += str(i+1)+" "+c[0]+" "
        # Build the full n-qubit Pauli by Kronecker product
        mat = combo[0][1]
        for c in combo[1:]:
            mat = np.kron(mat, c[1])
        yield (label, mat)

def matrix_to_pauli_sum(M, tol=1e-12):
    """
    Given a 2^n x 2^n NumPy array M, returns a string with the expansion of M
    in the n-qubit Pauli basis.
    
    Parameters:
    -----------
    M   : 2D NumPy array of shape (2^n, 2^n)
    tol : float, threshold below which coefficients are considered 0.
    
    Returns:
    --------
    A string of the form "c_1 * P1 + c_2 * P2 + ...",
    where Pk is an n-character string from {I, X, Y, Z}.
    """
    # Determine n from the shape of M (M is 2^n x 2^n)
    dim = M.shape[0]
    n = int(np.log2(dim))
    
    terms = []
    for label, pauli_mat in generate_n_qubit_paulis(n):
        # Compute the coefficient c = (1/2^n) Tr(Pauli * M)
        coeff = np.trace(pauli_mat @ M) / (2**n)
        
        # Only keep terms whose magnitude of coefficient is above threshold
        if abs(coeff) > tol:
            # Format real or complex coefficients suitably
            if np.isclose(coeff.imag, 0.0, atol=tol):
                # effectively real
                coeff_str = f"{coeff.real:.4g}"
            else:
                coeff_str = f"{coeff.real:.4g}{coeff.imag:+.4g}j"
            
            terms.append(f"{coeff_str} {label}")
    
    # Join all non-zero terms with " + "
    if not terms:
        return "0"
    return "\n".join(terms)

def NSpinTriangularHeisPauliString(N):
	PauliString = []
	
	for i in range (N-2):
		for j in range (2,4):
			PauliString.append("1.0 %d X %d X"%(i+1,i+j))
			PauliString.append("1.0 %d Y %d Y"%(i+1,i+j))
			PauliString.append("1.0 %d Z %d Z"%(i+1,i+j))
			
	PauliString.append("1.0 %d X %d X"%(N-1,N))
	PauliString.append("1.0 %d Y %d Y"%(N-1,N))
	PauliString.append("1.0 %d Z %d Z"%(N-1,N))
	
	return PauliString

def diagonalizing_unitary(matrix):
    """
    This function returns a unitary matrix that diagonalizes the given Hermitian matrix.
    
    Parameters:
    matrix (np.array): A Hermitian matrix.
    
    Returns:
    tuple: A tuple containing the unitary matrix and the diagonal matrix of eigenvalues.
    """
    # Check if the matrix is Hermitian
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("The matrix is not Hermitian")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # The eigenvectors form a unitary matrix
    return eigenvectors

def NSpinTriangularHeisPauliString(N):
	PauliString = []
	
	for i in range (N-2):
		for j in range (2,4):
			PauliString.append("1.0 %d X %d X"%(i+1,i+j))
			PauliString.append("1.0 %d Y %d Y"%(i+1,i+j))
			PauliString.append("1.0 %d Z %d Z"%(i+1,i+j))
			
	PauliString.append("1.0 %d X %d X"%(N-1,N))
	PauliString.append("1.0 %d Y %d Y"%(N-1,N))
	PauliString.append("1.0 %d Z %d Z"%(N-1,N))

	return "\n".join(PauliString)

# H1 Pauli strings (coefficients and terms)
H1_pauli = """1.0 1 X 2 X
1.0 1 Y 2 Y
1.0 1 Z 2 Z
1.0 1 X 3 X
1.0 1 Y 3 Y
1.0 1 Z 3 Z
1.0 2 X 3 X
1.0 2 Y 3 Y
1.0 2 Z 3 Z
1.0 2 X 4 X
1.0 2 Y 4 Y
1.0 2 Z 4 Z
1.0 3 X 4 X
1.0 3 Y 4 Y
1.0 3 Z 4 Z
1.0 3 X 5 X
1.0 3 Y 5 Y
1.0 3 Z 5 Z
1.0 4 X 5 X
1.0 4 Y 5 Y
1.0 4 Z 5 Z
"""
# Construct H1
H1 = pauli_sum_to_matrix(convert_to_format(H1_pauli, 5))

H2_pauli = """1.000000 1 X 2 Z 3 Z 4 Z 5 Z
1.000000 1 X 3 Z 4 Z 5 Z
-1.000000 1 X 5 Z
1.000000 1 X 3 Z 5 Z
1.000000 1 Y 2 Y 3 Z 4 Y 5 Y
-1.000000 4 X 5 Z
1.000000 2 Z 4 X 5 Z
-1.000000 1 Y 2 Y 3 Z 4 Z 5 X
1.000000 1 X 2 Z 3 Z 4 Y 5 X
1.000000 3 Y 5 Z
1.000000 2 Z 4 X 5 Y
1.000000 2 Z 3 Z 4 X 5 Y
1.000000 1 Y 2 Y 3 X 4 Z 5 Y
-1.000000 1 Z 2 Y 3 X 4 Z 5 X
1.000000 1 Z 2 X 4 X 5 X
-1.000000 1 Z 2 Y 5 Y
1.000000 1 X 2 Z 3 X 4 Y 5 Y
1.000000 1 Y 2 X 3 X 4 Y
-1.000000 1 Z 2 X 3 Y 4 Y 5 Z
1.000000 2 Z
1.000000 3 Z
"""

# Construct H2
H2 = pauli_sum_to_matrix(convert_to_format(H2_pauli, 5))

print(are_unitarily_similar(H1, H2))

"""
# Get U_2
U_2 = np.array([
    [1, 0, 0, 0],
    [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],
    [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
    [0, 0, 0, 1]
], dtype=complex)

# Create the full U matrix 
U_full = np.kron(np.kron(I, U_2), I)

# Conjugate H1 by U_full
H1_conjugated = U_full @ H1 @ U_full.conj().T
"""

# Prints 0 (or a number very close to 0) if equal 
#print(np.linalg.norm(H1_conjugated - H2))