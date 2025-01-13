import numpy as np

"""
Random Cliffords are generated using the Python script outlined in "Hadamard-free circuits
expose the structure of the Clifford group" by S. Bravyi and D. Maslov (2020), https://arxiv.org/pdf/2003.09412
"""

def gf2_add(a, b):
    return np.logical_xor(a, b)

def gf2_mul(a, b):
    return np.logical_and(a, b)

def binary_matmul_inner(A, B):
    m, kA = A.shape
    kB, n = B.shape
    if kA != kB:
        raise ValueError("Inner dimensions {kA} and {kB} must match for GF(2) multiplication.")
    
    C = np.zeros((m, n), dtype=bool)
    for i in range(m):
        for j in range(n):
            C[i,j] = np.logical_xor.reduce(np.logical_and(A[i,:], B[:,j]))
    return C

def swap_rows_inner(mat, r1, r2):
    mat[[r1, r2], :] = mat[[r2, r1], :]

def replace_row_inner(mat, dst_row, src_row):
    mat[dst_row, :] = src_row

def calc_inverse_matrix_inner(L, lower_tri=True):
    n, m = L.shape
    if n != m:
        raise ValueError(f"Dimensions {n} and {m} must match to compute the matrix inverse.")

    Inv = np.eye(n, dtype=bool)

    for i in range(n):
        for j in range(i+1, n):
            if L[j, i]:
                L[j, :] = gf2_add(L[j, :], L[i, :])
                Inv[j, :] = gf2_add(Inv[j, :], Inv[i, :])

    for i in reversed(range(n)):
        for j in range(i):
            if L[j, i]:
                L[j, :] = gf2_add(L[j, :], L[i, :])
                Inv[j, :] = gf2_add(Inv[j, :], Inv[i, :])
    return Inv

def inverse_tril(mat):
    return calc_inverse_matrix_inner(mat, lower_tri=True)

def sample_qmallows(n, rng):
    had = np.zeros(n, dtype=bool)
    perm = np.zeros(n, dtype=int)
    inds = list(range(n))

    for i in range(n):
        m = n - i
        eps = 4.0 ** (-m)
        r = rng.random()
        index = int(-np.ceil(np.log2(r + (1.0 - r) * eps)))

        had[i] = (index < m)

        if index < m:
            k = index
        else:
            k = 2 * m - index - 1

        perm[i] = inds[k]
        del inds[k]

    return had, perm

def fill_tril(mat, rng, symmetric):
    n = mat.shape[0]
    for i in range(n):
        for j in range(i):
            val = rng.integers(0, 2) == 1
            mat[i, j] = val
            if symmetric:
                mat[j, i] = val

def random_clifford_tableau(num_qubits, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    had, perm = sample_qmallows(num_qubits, rng)

    gamma1 = np.zeros((num_qubits, num_qubits), dtype=bool)
    for i in range(num_qubits):
        gamma1[i, i] = (rng.integers(0, 2) == 1)
    fill_tril(gamma1, rng, symmetric=True)

    gamma2 = np.zeros((num_qubits, num_qubits), dtype=bool)
    for i in range(num_qubits):
        gamma2[i, i] = (rng.integers(0, 2) == 1)
    fill_tril(gamma2, rng, symmetric=True)

    delta1 = np.zeros((num_qubits, num_qubits), dtype=bool)
    for i in range(num_qubits):
        delta1[i, i] = True
    fill_tril(delta1, rng, symmetric=False)

    delta2 = np.zeros((num_qubits, num_qubits), dtype=bool)
    for i in range(num_qubits):
        delta2[i, i] = True
    fill_tril(delta2, rng, symmetric=False)

    zero = np.zeros((num_qubits, num_qubits), dtype=bool)
    prod1 = binary_matmul_inner(gamma1, delta1)
    prod2 = binary_matmul_inner(gamma2, delta2)

    inv1 = inverse_tril(delta1).T.copy()
    inv2 = inverse_tril(delta2).T.copy()

    top1 = np.concatenate([delta1, zero], axis=1)
    bot1 = np.concatenate([prod1, inv1], axis=1)
    table1 = np.concatenate([top1, bot1], axis=0)

    top2 = np.concatenate([delta2, zero], axis=1)
    bot2 = np.concatenate([prod2, inv2], axis=1)
    table2 = np.concatenate([top2, bot2], axis=0)

    table = np.zeros((2 * num_qubits, 2 * num_qubits), dtype=bool)
    for i in range(num_qubits):
        replace_row_inner(table, i,     table2[perm[i], :])
        replace_row_inner(table, i + num_qubits, table2[perm[i] + num_qubits, :])

    for i in range(num_qubits):
        if had[i]:
            swap_rows_inner(table, i, i + num_qubits)

    random_symplectic_mat = binary_matmul_inner(table1, table)

    random_phases = rng.integers(0, 2, size=(2 * num_qubits, 1)).astype(bool)

    random_tableau = np.concatenate([random_symplectic_mat, random_phases], axis=1)

    return random_tableau