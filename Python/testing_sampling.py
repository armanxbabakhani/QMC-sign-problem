from sign_tools import *

filename = sys.argv[1]
Coefficients, BinaryVectors , NumOfParticles = parse_pauli_file(filename)
AllPermsBinary , AllDiagsBinary , PureDiagonals = process_pauli_terms(Coefficients , BinaryVectors , NumOfParticles)
if len(PureDiagonals) > 0:
    AllPermsBinary.append([0]*NumOfParticles)
    AllDiagsBinary.append(PureDiagonals[0])


# ---------------------- Initial Cost -----------------------------
InitialTotalCost , CostsQ , CyclesQ = total_cost_from_binary_operators(AllPermsBinary , AllDiagsBinary)

print(f'The initial total cost is {InitialTotalCost}')