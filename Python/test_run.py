from sign_tools import *


# Example usage:
filename = sys.argv[1]
coefficients, binary_vectors = parse_pauli_file(filename)
print("Coefficients: " , coefficients)
print("Binary Vectors: " , binary_vectors)
print(" ")
permutations_binary , offdiagonals_binary , pure_diagonals = process_pauli_terms(coefficients , binary_vectors)

# Convert the binary permutations to integer particle number permutations
permutation_indices = []
for permutation in permutations_binary:
    permutation_indices.append(binary_to_indices(permutation))

offdiagonals_indices = convert_diagonal_to_indices(offdiagonals_binary)

print("Permutations are: " , permutations_binary )
print("Permutations in indices are: " , permutation_indices )
print("Diagonals are: " , offdiagonals_binary )
print("Diagonals in indices are: " , offdiagonals_indices)
print(" ")
null_space = mod2_nullspace(permutations_binary)
print("The cycles are : " , null_space)

NullspaceIndices , OffdiagonalCycles = convert_binary_cycles_to_indices(null_space , permutation_indices , offdiagonals_indices)

#perm_cycle = [permutation_indices[i] for i in range(N) if null_space[0][i]==1 ]
#offdiag_cycle = [offdiagonals_indices[i] for i in range(N) if null_space[0][i]==1]
#print(f"The permutation cycle is {perm_cycle}")
#offdiag_cycle = [[[3 , 3] , [[0,1] , [2]]] , [[2] , [[0]]] , [[-7] , [[2]]]]

trial_state = [0]*N
trial_state[1] = 1
print(" ")
print("The cycle is " , NullspaceIndices[0])
print("The off diagonal cycle is " , OffdiagonalCycles[0])

trial_weight = cycle_weight(trial_state , OffdiagonalCycles[0] , NullspaceIndices[0])

print(" ")
print("The trial state is " , trial_state)
print("The trial weight is " , trial_weight)