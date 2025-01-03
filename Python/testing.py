from sign_tools import *

filename = sys.argv[1]
Coefficients, BinaryVectors , NumOfParticles = parse_pauli_file(filename)
AllPermsBinary , AllDiagsBinary , PureDiagonals = process_pauli_terms(Coefficients , BinaryVectors , NumOfParticles)
if len(PureDiagonals) > 0:
    AllPermsBinary.append([0]*NumOfParticles)
    AllDiagsBinary.append(PureDiagonals[0])


#AllPermsBinaryNew , AllDiagsBinaryNew , transformation = apply_random_transformation([0 , 1.0 , 0] , AllPermsBinary , AllDiagsBinary , NumOfParticles)
#AllPermsBinaryNew , AllDiagsBinaryNew = apply_single_body(AllPermsBinary, AllDiagsBinary , [1 , 2] , 'H')
AllPermsBinaryNew , AllDiagsBinaryNew = apply_single_body(AllPermsBinary, AllDiagsBinary , [] , 'S')
AllPermsBinaryNew , AllDiagsBinaryNew = apply_CNOT(AllPermsBinaryNew , AllDiagsBinaryNew , [tuple([0,1])])

IdentityIndex = -1
PureDiagonalsNew = []
for i in range(len(AllPermsBinaryNew)):
    if AllPermsBinaryNew[i] == [0]*NumOfParticles:
        PureDiagonalsNew = AllDiagsBinaryNew[i]
        IdentityIndex = i

if IdentityIndex >= 0:
    AllPermsBinaryNew = AllPermsBinaryNew[:IdentityIndex]+AllPermsBinaryNew[IdentityIndex+1:]
    AllDiagsBinaryNew = AllDiagsBinaryNew[:IdentityIndex]+AllDiagsBinaryNew[IdentityIndex+1:]

#print(f'The transformation is {transformation}')
print(' ')

generate_pauli_file_from_pmr_data(filename.removesuffix(".txt")+'_rotated.txt', AllPermsBinaryNew , AllDiagsBinaryNew , PureDiagonalsNew)