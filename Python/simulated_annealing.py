from sign_tools import *

# ==================================== Simulated Annealing =================================================
# ********************************* Reading in the input file *********************************************
filename = sys.argv[1]
Coefficients, BinaryVectors , NumOfParticles = parse_pauli_file(filename)
AllPermsBinary , AllDiagsBinary , PureDiagonals = process_pauli_terms(Coefficients , BinaryVectors , NumOfParticles)
if len(PureDiagonals) > 0:
    AllPermsBinary.append([0]*NumOfParticles)
    AllDiagsBinary.append(PureDiagonals[0])


# ---------------------- Initial Cost -----------------------------
InitialTotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinary , AllDiagsBinary)
CyclesOriginal , N = get_all_cycles_from_file(filename)
print(f'The cost for each q is {CostsQ}')
print(f'The total cost of the Hamiltonian is {InitialTotalCost}')
print(' ')


# ---------------------- Simulation criteria -----------------------------
Probabilities = [0.5 , 0.5 , 0.0]
TotalCost = InitialTotalCost
BestCost = InitialTotalCost
AbsoluteBest = BestCost
AbsoluteBestFound = False
AllTransformations = []
BestTransformations = []

MaxIterations = 2000
Iteration = 0

# ================= Simulated annealing 
while TotalCost > 0.0 and Iteration < MaxIterations:
    AllPermsBinaryNew , AllDiagsBinaryNew , Transformation = apply_random_transformation(Probabilities , AllPermsBinary , AllDiagsBinary , NumOfParticles)
    TotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinaryNew , AllDiagsBinaryNew)

    DeltaCost = TotalCost - BestCost
    TransitionProb = np.min([np.exp(-5.0*DeltaCost*(Iteration/10 + InitialTotalCost/10.0)/InitialTotalCost) , 1])
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

if AbsoluteBestFound:
    AllPermsBinary = AbsoluteBestAllPermsBinary
    AllDiagsBinary = AbsoluteBestAllDiagsBinary
# Get the pure diagonal term:
PureDiagonals = []
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
