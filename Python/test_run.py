from sign_tools import *


def total_cost_from_binary_operators(AllPermsBinary , AllDiagsBinary):
    NumOfParticles = len(AllPermsBinary[0])    

    # Remove the identity if it is in AllPermsBinary
    AllPermsBinaryWOIden = AllPermsBinary.copy()
    AllDiagsBinaryWOIden = AllDiagsBinary.copy()

    AllPermsBinaryWOIden = [AllPermsBinaryWOIden[i] for i in range(len(AllPermsBinary)) if not AllPermsBinaryWOIden[i]==[0]*NumOfParticles ]
    if len(AllPermsBinaryWOIden) < len(AllPermsBinary):
        AllDiagsBinaryWOIden = AllDiagsBinaryWOIden[:len(AllDiagsBinaryWOIden)-1]

    # Convert binary permutations into vector of int indices
    PermutationIndices = []
    for permutation in AllPermsBinaryWOIden:
        PermutationIndices.append(binary_to_indices(permutation))
    OffDiagonalsIndices = convert_diagonal_to_indices(AllDiagsBinaryWOIden)
    NullSpace = mod2_nullspace(AllPermsBinaryWOIden)

    # Generating all cycles of up to length 5:
    PermCycleIndices , OffDiagCycleIndices = convert_binary_cycles_to_indices(NullSpace , PermutationIndices , OffDiagonalsIndices , NumOfParticles)

    FundCyclesIndices , FundCycOffDiagsIndices = generate_cyclic_permutations_with_offdiagonals(PermCycleIndices , PermutationIndices , OffDiagonalsIndices)
    
    HighCycles , HighOffDiags = generate_higher_cycles(FundCyclesIndices , FundCycOffDiagsIndices , PermutationIndices , OffDiagonalsIndices)

    AllCycles = FundCyclesIndices + HighCycles
    AllCycOffDs = FundCycOffDiagsIndices + HighOffDiags
    
    CyclesQ = {}
    for i in range(len(AllCycles)):
        q = len(AllCycles[i])
        if q not in CyclesQ :
            CyclesQ[q] = {'Permutation Cycles':[] , 'Diagonal Cycles': []}
        CyclesQ[q]['Permutation Cycles'].append(AllCycles[i])
        CyclesQ[q]['Diagonal Cycles'].append(AllCycOffDs[i])

    CostsQ = {}
    TotalCost = 0.0
    for q in np.arange(3,6):
        Cost = total_hamiltonian_cost(CyclesQ[q]['Diagonal Cycles'] , CyclesQ[q]['Permutation Cycles'] , NumOfParticles)
        TotalCost += Cost
        CostsQ[q] = Cost

    return TotalCost , CostsQ , CyclesQ

filename = sys.argv[1]
Coefficients, BinaryVectors , NumOfParticles = parse_pauli_file(filename)
AllPermsBinary , AllDiagsBinary , PureDiagonals = process_pauli_terms(Coefficients , BinaryVectors , NumOfParticles)
if len(PureDiagonals)>0:
    AllPermsBinary.append([0]*NumOfParticles)
    AllDiagsBinary.append(PureDiagonals[0])

InitialTotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinary , AllDiagsBinary)
CyclesOriginal , N = get_all_cycles_from_file(filename)
#print(f'The original cycles for q=4 is {CyclesOriginal[4]['Permutation Cycles']}')
#print(f'The diagonal cycles of the original cycle for q=4 is {CyclesOriginal[4]['Diagonal Cycles']}')
#print(f'The perm cycles for new q=4 is {CyclesQ[4]['Permutation Cycles']}')
#print(f'The diagonal cycles of the new cycle for q=4 is {CyclesQ[4]['Diagonal Cycles']}')
#print(f'The cost for q=4 of the new cycles {total_hamiltonian_cost(CyclesQ[4]['Diagonal Cycles'] , CyclesQ[4]['Permutation Cycles'] , N)}')
#print(f'The cost for q=4 of the original cycles are {total_hamiltonian_cost(CyclesOriginal[4]['Diagonal Cycles'] , CyclesOriginal[4]['Permutation Cycles'] , N)}')
#print(f'The Cycles for each q=4 is {CyclesQ[4]['Permutation Cycles']}')
print(f'The cost for each q is {CostsQ}')
print(f'The total cost of the Hamiltonian is {InitialTotalCost}')
print(' ')

Probabilities = [0.5 , 0.25 , 0.25]
TotalCost = InitialTotalCost
while TotalCost > InitialTotalCost/5.0:
    AllPermsBinary , AllDiagsBinary = apply_random_transformation(Probabilities , AllPermsBinary , AllDiagsBinary , NumOfParticles)
    TotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinary , AllDiagsBinary)
    print('After transformation ... ')
    #print(f'The Cycles for each q is {CyclesQ}')
    print(f'The cost for each q is {CostsQ}')
    print(f'The total cost of the Hamiltonian is {TotalCost}')
    print(' ')

# Apply a speicific transformation:
#   Apparently an all S tranformation cures the sign problem?!

# Get the pure diagonal term:
for i in range(len(AllPermsBinary)):
    if AllPermsBinary[i] == [0]*NumOfParticles:
        PureDiagonals = AllDiagsBinary[i]
        IdentityIndex = i

AllPermsBinary = AllPermsBinary[:IdentityIndex]+AllPermsBinary[IdentityIndex+1:]
AllDiagsBinary = AllDiagsBinary[:IdentityIndex]+AllDiagsBinary[IdentityIndex+1:]

print(f'The permutations are {AllPermsBinary}')
print(f'The diagonals are {AllDiagsBinary}')
print(f'The purely diagonals are {PureDiagonals}')

# Writing back into an ouptut file:

generate_pauli_file_from_pmr_data(filename.removesuffix(".txt")+'_sign_optimized.txt', AllPermsBinary , AllDiagsBinary , PureDiagonals)

#AllCycles , NumberOfParticles = get_all_cycles_from_file(filename)

#CostQ = {}
#for q in np.arange(3 , 6):
#    PermCyclesQ = AllCycles[q]['Permutation Cycles']
#    DiagCyclesQ = AllCycles[q]['Diagonal Cycles']
#    CostQ[q] = total_hamiltonian_cost(DiagCyclesQ , PermCyclesQ , NumberOfParticles)

# for q in np.arange(3,6):
#     print(f'For q = {q}:')
#     print(f'Cost is {CostQ[q]}')
#     print(' ')

# print(f'The total cost of the Hamiltonian is: ' , np.sum([CostQ[q] for q in np.arange(3, 6)]))

# Perms = [[0,1,1] , [1,0,0]]
# Diags = [[[1.0] , [[1,1,0]]] , [[2.5 , -1.0] , [ [0,0,0] , [1,0,1]]] ]

# PermsT , DiagsT = apply_single_body(Perms , Diags , [0,2] , 'H')
# print(f'The transformed Perms are: ' , PermsT)
# print(f'The transformed Diags are: ' , DiagsT)