import numpy as np
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
        label = "".join(c[0] for c in combo)
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
                coeff_str = f"({coeff.real:.4g}{coeff.imag:+.4g}j)"
            
            terms.append(f"{coeff_str} * {label}")
    
    # Join all non-zero terms with " + "
    if not terms:
        return "0"
    return "\n".join(terms)

# H1 Pauli strings (coefficients and terms)
H1_pauli = [
  (1.0, 'X', 'X', 'I', 'I', 'I', 'I') ,
(1.0, 'Y', 'Y', 'I', 'I', 'I', 'I') ,
(1.0, 'Z', 'Z', 'I', 'I', 'I', 'I') ,
(1.0, 'X', 'I', 'X', 'I', 'I', 'I') ,
(1.0, 'Y', 'I', 'Y', 'I', 'I', 'I') ,
(1.0, 'Z', 'I', 'Z', 'I', 'I', 'I') ,
(1.0, 'I', 'X', 'X', 'I', 'I', 'I') ,
(1.0, 'I', 'Y', 'Y', 'I', 'I', 'I') ,
(1.0, 'I', 'Z', 'Z', 'I', 'I', 'I') ,
(1.0, 'I', 'X', 'I', 'X', 'I', 'I') ,
(1.0, 'I', 'Y', 'I', 'Y', 'I', 'I') ,
(1.0, 'I', 'Z', 'I', 'Z', 'I', 'I') ,
(1.0, 'I', 'I', 'X', 'X', 'I', 'I') ,
(1.0, 'I', 'I', 'Y', 'Y', 'I', 'I') ,
(1.0, 'I', 'I', 'Z', 'Z', 'I', 'I') ,
(1.0, 'I', 'I', 'X', 'I', 'X', 'I') ,
(1.0, 'I', 'I', 'Y', 'I', 'Y', 'I') ,
(1.0, 'I', 'I', 'Z', 'I', 'Z', 'I') ,
(1.0, 'I', 'I', 'I', 'X', 'X', 'I') ,
(1.0, 'I', 'I', 'I', 'Y', 'Y', 'I') ,
(1.0, 'I', 'I', 'I', 'Z', 'Z', 'I') ,
(1.0, 'I', 'I', 'I', 'X', 'I', 'X') ,
(1.0, 'I', 'I', 'I', 'Y', 'I', 'Y') ,
(1.0, 'I', 'I', 'I', 'Z', 'I', 'Z') ,
(1.0, 'I', 'I', 'I', 'I', 'X', 'X') ,
(1.0, 'I', 'I', 'I', 'I', 'Y', 'Y') ,
(1.0, 'I', 'I', 'I', 'I', 'Z', 'Z')
]

# Construct H1
H1 = pauli_sum_to_matrix(H1_pauli)

# Get U_2
U_2 = np.array([
    [1, 0, 0, 0],
    [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],
    [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
    [0, 0, 0, 1]
], dtype=complex)

# Create the full U matrix (I ⊗ U_2 ⊗ I ⊗ I)
U_full = np.kron(np.kron(np.kron(I, U_2), U_2), I)

# Conjugate H1 by U_full
H1_conjugated = U_full @ H1 @ U_full.conj().T

#print(matrix_to_pauli_sum(H1_conjugated))

w = 1/np.sqrt(2)
H2_pauli = [
(w, 'X', 'X', 'I', 'I', 'I', 'I') ,
(w, 'Y', 'Y', 'I', 'I', 'I', 'I') ,
(w, 'X', 'X', 'Z', 'I', 'I', 'I') ,
(w, 'Y', 'Y', 'Z', 'I', 'I', 'I') ,
(-w, 'X', 'Z', 'X', 'I', 'I', 'I') ,
(-w, 'Y', 'Z', 'Y', 'I', 'I', 'I') ,
(w, 'X', 'I', 'X', 'I', 'I', 'I') ,
(w, 'Y', 'I', 'Y', 'I', 'I', 'I') ,
(0.25, 'I', 'I', 'Z', 'X', 'X', 'I') ,
(0.25, 'I', 'I', 'Z', 'Y', 'Y', 'I') ,
(0.25, 'I', 'Z', 'I', 'X', 'X', 'I') ,
(0.25, 'I', 'Z', 'I', 'Y', 'Y', 'I') ,
(-0.25, 'I', 'X', 'X', 'I', 'Z', 'I') ,
(-0.25, 'I', 'X', 'X', 'Z', 'I', 'I') ,
(-0.25, 'I', 'Y', 'Y', 'I', 'Z', 'I') ,
(-0.25, 'I', 'Y', 'Y', 'Z', 'I', 'I') ,
(0.25, 'I', 'X', 'X', 'X', 'X', 'I') ,
(0.25, 'I', 'X', 'X', 'Y', 'Y', 'I') ,
(0.25, 'I', 'Y', 'Y', 'X', 'X', 'I') ,
(0.25, 'I', 'Y', 'Y', 'Y', 'Y', 'I') ,
(0.5, 'I', 'X', 'I', 'X', 'I', 'I') ,
(0.5, 'I', 'Y', 'I', 'Y', 'I', 'I') ,
(0.5, 'I', 'X', 'Z', 'X', 'I', 'I') ,
(0.5, 'I', 'Y', 'Z', 'Y', 'I', 'I') ,
(0.5, 'I', 'X', 'Z', 'X', 'Z', 'I') ,
(0.5, 'I', 'Y', 'Z', 'Y', 'Z', 'I') ,
(-0.5, 'I', 'X', 'I', 'Z', 'X', 'I') ,
(-0.5, 'I', 'Y', 'I', 'Z', 'Y', 'I') ,
(-0.5, 'I', 'X', 'Z', 'Z', 'X', 'I') ,
(-0.5, 'I', 'Y', 'Z', 'Z', 'Y', 'I') ,
(0.5, 'I', 'X', 'Z', 'I', 'X', 'I') ,
(0.5, 'I', 'Y', 'Z', 'I', 'Y', 'I') ,
(-0.5, 'I', 'Z', 'X', 'X', 'I', 'I') ,
(-0.5, 'I', 'Z', 'Y', 'Y', 'I', 'I') ,
(0.5, 'I', 'I', 'X', 'X', 'I', 'I') ,
(0.5, 'I', 'I', 'Y', 'Y', 'I', 'I') ,
(0.5, 'I', 'I', 'X', 'X', 'Z', 'I') ,
(0.5, 'I', 'I', 'Y', 'Y', 'Z', 'I') ,
(0.5, 'I', 'Z', 'X', 'Z', 'X', 'I') ,
(0.5, 'I', 'Z', 'Y', 'Z', 'Y', 'I') ,
(-0.5, 'I', 'I', 'X', 'Z', 'X', 'I') ,
(-0.5, 'I', 'I', 'Y', 'Z', 'Y', 'I') ,
(0.5, 'I', 'I', 'X', 'I', 'X', 'I') ,
(0.5, 'I', 'I', 'Y', 'I', 'Y', 'I') ,
(w, 'I', 'I', 'I', 'X', 'I', 'X') ,
(w, 'I', 'I', 'I', 'Y', 'I', 'Y') ,
(w, 'I', 'I', 'I', 'X', 'Z', 'X') ,
(w, 'I', 'I', 'I', 'Y', 'Z', 'Y') ,
(-w, 'I', 'I', 'I', 'Z', 'X', 'X') ,
(-w, 'I', 'I', 'I', 'Z', 'Y', 'Y') ,
(w, 'I', 'I', 'I', 'I', 'X', 'X') ,
(w, 'I', 'I', 'I', 'I', 'Y', 'Y') ,
(1.0, 'I', 'I', 'Z', 'I', 'I', 'I') ,
(-1.0, 'I', 'Z', 'I', 'I', 'I', 'I') ,
(1.0, 'Z', 'I', 'Z', 'I', 'I', 'I') ,
(1.0, 'Z', 'Z', 'I', 'I', 'I', 'I') ,
(1.0, 'I', 'Z', 'Z', 'I', 'I', 'I') ,
(0.75, 'I', 'I', 'Z', 'I', 'Z', 'I') ,
(0.75, 'I', 'I', 'Z', 'Z', 'I', 'I') ,
(0.75, 'I', 'Z', 'I', 'I', 'Z', 'I') ,
(0.75, 'I', 'Z', 'I', 'Z', 'I', 'I') ,
(1.0, 'I', 'I', 'I', 'Z', 'Z', 'I') ,
(1.0, 'I', 'I', 'I', 'I', 'Z', 'Z') ,
(1.0, 'I', 'I', 'I', 'Z', 'I', 'Z') ,
(1.0, 'I', 'I', 'I', 'I', 'Z', 'I') ,
(-1.0, 'I', 'I', 'I', 'Z', 'I', 'I')
]

# Construct H2
H2 = pauli_sum_to_matrix(H2_pauli)

# Prints 0 (or a number very close to 0) if equal 
print(np.linalg.norm(H1_conjugated - H2))