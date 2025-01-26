import numpy as np

"""
Random Cliffords are sampled using the Python script provided in "Hadamard-free circuits
expose the structure of the Clifford group" by S. Bravyi and D. Maslov (2020), https://arxiv.org/pdf/2003.09412
"""

def sample_qm_allows(n):
    h = np.zeros(n, dtype=int)
    S = np.zeros(n, dtype=int)
    A = list(range(n))
    for i in range(n):
        m = n - i
        r = np.random.uniform(0, 1)
        index = int(np.ceil(np.log2(1 + ((1 - r) * (4 ** (-m))))))
        h[i] = 1 if (index < m) else 0
        if index < m:
            k = index
        else:
            k = 2 * m - index - 1
        S[i] = A[k]
        del A[k]
    return h, S

def random_clifford_tableau(n):
    assert(n <= 200)

    ZR = np.zeros((n, n), dtype=int)
    ZR2 = np.zeros((2 * n, 2 * n), dtype=int)
    I = np.identity(n, dtype=int)
    h, S = sample_qm_allows(n)
    Gamma1 = np.copy(ZR)
    Delta1 = np.copy(I)
    Gamma2 = np.copy(ZR)
    Delta2 = np.copy(I)
    for i in range(n):
        Gamma2[i, i] = np.random.randint(2)
        if h[i]:
            Gamma1[i, i] = np.random.randint(2)

    for j in range(n):
        for i in range(j+1, n):
            b = np.random.randint(2)
            Gamma2[i, j] = b
            Gamma2[j, i] = b
            Delta2[i, j] = np.random.randint(2)
            if h[i] == 1 and h[j] == 1:
                b = np.random.randint(2)
                Gamma1[i, j] = b
                Gamma1[j, i] = b
            if h[i] == 1 and h[j] == 0 and S[i] < S[j]:
                b = np.random.randint(2)
                Gamma1[i, j] = b
                Gamma1[j, i] = b
            if h[i] == 0 and h[j] == 1 and S[i] > S[j]:
                b = np.random.randint(2)
                Gamma1[i, j] = b
                Gamma1[j, i] = b
            if h[i] == 0 and h[j] == 1:
                Delta1[i, j] = np.random.randint(2)
            if h[i] == 1 and h[j] == 1 and S[i] > S[j]:
                Delta1[i, j] = np.random.randint(2)
            if h[i] == 0 and h[j] == 0 and S[i] < S[j]:
                Delta1[i, j] = np.random.randint(2)
    # compute stabilizer tableau auxiliaries
    PROD1 = np.matmul(Gamma1, Delta1)
    PROD2 = np.matmul(Gamma2, Delta2)
    INV1 = np.linalg.inv(np.transpose(Delta1))
    INV2 = np.linalg.inv(np.transpose(Delta2))
    F1 = np.block([[Delta1, ZR], [PROD1, INV1]])
    F2 = np.block([[Delta2, ZR], [PROD2, INV2]])
    F1 = F1.astype(int) % 2
    F2 = F2.astype(int) % 2
    # compute the full stabilizer tableau
    U = np.copy(ZR2)
    # apply qubit permutation S to F2
    for i in range(n):
        U[i, :] = F2[S[i], :]
        U[i + n, :] = F2[S[i] + n, :]
    # apply layer of Hadamards
    for i in range(n):
        if h[i] == 1:
            U[(i, i + n), :] = U[(i + n, i), :]
    # apply F1
    return np.matmul(F1, U) % 2
