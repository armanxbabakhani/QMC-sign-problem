import sys
from sign_tools import *
import numpy as np

N = int(sys.argv[1])
filename = './Inputs/Triangular_Ladder_Heisenberg/Triangular_Heis_n='+str(N)+'.txt'
Coefficients, BinaryVectors , NumOfParticles = parse_pauli_file(filename)
AllPermsBinary , AllDiagsBinary , PureDiagonals = process_pauli_terms(Coefficients , BinaryVectors , NumOfParticles)
InitialTotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinary , AllDiagsBinary)
if len(PureDiagonals) > 0:
    AllPermsBinary.append([0]*NumOfParticles)
    AllDiagsBinary.append(PureDiagonals[0])

print('Optimization is initialized with the following cost...')
print(f'The initial cost is {InitialTotalCost} and all costs are {CostsQ}')
print(' ')



# ================== Optimization =========================
TotalCost = InitialTotalCost
BestCost = InitialTotalCost
AbsoluteBest = BestCost
AbsoluteBestFound = False
AllTransformations = []
BestTransformations = []

MaxIterations = int(sys.argv[2])
Iteration = 0
CliffordProbability = 0.5
# ================= Simulated annealing 
TransformationSinceBest = []
#AllPermsBinary = AllPermsBinaryT
#AllDiagsBinary = AllDiagsBinaryT

while TotalCost > TotalCost/500 and Iteration < MaxIterations:
    AllPermsBinaryNew , AllDiagsBinaryNew , Transformation = apply_random_transformation(CliffordProbability , AllPermsBinary , AllDiagsBinary , NumOfParticles)
    #print(f'Transformation number {Iteration} generated! Now computing cost ... ')
    TotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinaryNew , AllDiagsBinaryNew)

    DeltaCost = TotalCost - BestCost
    TransitionProb = np.min([np.exp(-1.0*DeltaCost*(Iteration/10 + InitialTotalCost/10.0)/InitialTotalCost) , 1])
    p = random.random()
    Iteration += 1
    if p < TransitionProb:
        AllPermsBinary = AllPermsBinaryNew
        AllDiagsBinary = AllDiagsBinaryNew
        BestCost = TotalCost
        AllTransformations.append(Transformation)
        print(f'The new cost is {TotalCost}')
        print(f'The transition probability is {TransitionProb}')
        print(' ')
        print('The transformation has been accepted!')
        print(' ')
        print(' ')

# ====================== Writing output optimized file ===================================
PureDiagonals = []
for i in range(len(AllPermsBinary)):
    if AllPermsBinary[i] == [0]*NumOfParticles:
        PureDiagonals = AllDiagsBinary[i]
        IdentityIndex = i

print(' ')
print(f'Here is the final report of the costs and transformations ... ')
print(f'The best cost is {BestCost}')
print(f'The best transformations is {AllTransformations}')

AllPermsBinary = AllPermsBinary[:IdentityIndex]+AllPermsBinary[IdentityIndex+1:]
AllDiagsBinary = AllDiagsBinary[:IdentityIndex]+AllDiagsBinary[IdentityIndex+1:]

generate_pauli_file_from_pmr_data(filename.removesuffix(".txt")+'_full_optimized.txt', AllPermsBinary , AllDiagsBinary , PureDiagonals)
