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
if len(PureDiagonals) > 0:
    AllPermsBinary.append([0]*NumOfParticles)
    AllDiagsBinary.append(PureDiagonals[0])

InitialTotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinary , AllDiagsBinary)
CyclesOriginal , N = get_all_cycles_from_file(filename)
print(f'The cost for each q is {CostsQ}')
print(f'The total cost of the Hamiltonian is {InitialTotalCost}')
print(' ')

Probabilities = [0.5 , 0.5 , 0.0]
TotalCost = InitialTotalCost
BestCost = InitialTotalCost
AbsoluteBest = BestCost
AbsoluteBestFound = False
AllTransformations = []
BestTransformations = []

MaxIterations = 5000
Iteration = 0

# Simulated annealing:
while TotalCost > 0.0 and Iteration < MaxIterations:
    AllPermsBinaryNew , AllDiagsBinaryNew , Transformation = apply_random_transformation(Probabilities , AllPermsBinary , AllDiagsBinary , NumOfParticles)
    TotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinaryNew , AllDiagsBinaryNew)

    DeltaCost = TotalCost - BestCost
    TransitionProb = np.min([np.exp(-1.0*DeltaCost*(Iteration/10 + InitialTotalCost/10.0)) , 1])
    print(f'The new cost is {TotalCost}')
    print(f'The transition probability is {TransitionProb}')
    print(' ')
    p = random.random()
    Iteration += 1
    if TotalCost < AbsoluteBest:
        AbsoluteBest = TotalCost
        AbsoluteBestAllPermsBinary = AllPermsBinaryNew
        AbsoluteBestAllDiagsBinary = AllDiagsBinaryNew
        BestTransformations.append(Transformation)
        AbsoluteBestFound = True
    if p < TransitionProb:
        AllPermsBinary = AllPermsBinaryNew
        AllDiagsBinary = AllDiagsBinaryNew
        BestCost = TotalCost
        AllTransformations.append(Transformation)
        print('The transformation has been accepted!')
        print(' ')
        print(' ')


# while TotalCost > InitialTotalCost/5.0 and TotalCost > 5.0 and Iteration < MaxIterations:
#     AllPermsBinaryNew , AllDiagsBinaryNew = apply_random_transformation(Probabilities , AllPermsBinary , AllDiagsBinary , NumOfParticles)
#     TotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinaryNew , AllDiagsBinaryNew)
#     print(' ')
#     print('After transformation ... ')
#     #print(f'The Cycles for each q is {CyclesQ}')
#     print(f'The cost for each q is {CostsQ}')
#     print(f'The total cost of the Hamiltonian is {TotalCost}')
#     print(f'Iteration number: {Iteration}')
#     print(' ')
#     Iteration += 1
#     if TotalCost < InitialTotalCost:
#         AllPermsBinary = AllPermsBinaryNew
#         AllDiagsBinary = AllDiagsBinaryNew
#         print('The transformation is accepted!')
#         print(' ')
#         print(' ')

# Apply a speicific transformation:
#   Apparently an all S tranformation cures the sign problem?!
if AbsoluteBestFound:
    AllPermsBinary = AbsoluteBestAllPermsBinary
    AllDiagsBinary = AbsoluteBestAllDiagsBinary
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
print(' ')
print(' ')
# Writing back into an ouptut file:
print(f'The best cost is {BestCost}')
print(f'Best Transformations are {BestTransformations}')

generate_pauli_file_from_pmr_data(filename.removesuffix(".txt")+'_optimized.txt', AllPermsBinary , AllDiagsBinary , PureDiagonals)

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