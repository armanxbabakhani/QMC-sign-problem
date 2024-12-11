from sign_tools import *


filename = sys.argv[1]
AllCycles , NumberOfParticles = get_all_cycles_from_file(filename)

CostQ = {}
for q in np.arange(3 , 6):
    PermCyclesQ = AllCycles[q]['Permutation Cycles']
    DiagCyclesQ = AllCycles[q]['Diagonal Cycles']
    CostQ[q] = total_hamiltonian_cost(DiagCyclesQ , PermCyclesQ , NumberOfParticles)


for q in np.arange(3,6):
    print(f'For q = {q}:')
    print(f'Cost is {CostQ[q]}')
    print(' ')

print(f'The total cost of the Hamiltonian is: ' , np.sum([CostQ[q] for q in np.arange(3, 6)]))