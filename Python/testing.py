from sign_tools import *

filename = sys.argv[1]
Coefficients, BinaryVectors , NumOfParticles = parse_pauli_file(filename)
AllPermsBinary , AllDiagsBinary , PureDiagonals = process_pauli_terms(Coefficients , BinaryVectors , NumOfParticles)
if len(PureDiagonals) > 0:
    AllPermsBinary.append([0]*NumOfParticles)
    AllDiagsBinary.append(PureDiagonals[0])

"""
def NSpinTriangularHeisPauliString(N):
	PauliString = []
	
	for i in range (N-2):
		for j in range (2,4):
			PauliString.append("1.0 %d X %d X"%(i+1,i+j))
			PauliString.append("1.0 %d Y %d Y"%(i+1,i+j))
			PauliString.append("1.0 %d Z %d Z"%(i+1,i+j))
			
	PauliString.append("1.0 %d X %d X"%(N-1,N))
	PauliString.append("1.0 %d Y %d Y"%(N-1,N))
	PauliString.append("1.0 %d Z %d Z"%(N-1,N))
	
	return PauliString

for l in NSpinTriangularHeisPauliString(9):
	print(l)
"""

#AllPermsBinaryNew , AllDiagsBinaryNew , transformation = apply_random_transformation([0 , 1.0 , 0] , AllPermsBinary , AllDiagsBinary , NumOfParticles)
#AllPermsBinaryNew , AllDiagsBinaryNew = apply_single_body(AllPermsBinary, AllDiagsBinary , [1 , 2] , 'H')
#AllPermsBinaryNew , AllDiagsBinaryNew = apply_single_body(AllPermsBinary, AllDiagsBinary , [] , 'S')
#AllPermsBinaryNew , AllDiagsBinaryNew = apply_CNOT(AllPermsBinaryNew , AllDiagsBinaryNew , [tuple([0,1])])
AllPermsBinaryNew , AllDiagsBinaryNew = apply_U2_rotation(AllPermsBinary , AllDiagsBinary , (1, 2))
AllPermsBinaryNew , AllDiagsBinaryNew = apply_U2_rotation(AllPermsBinaryNew , AllDiagsBinaryNew , (3, 4))

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

generate_pauli_file_from_pmr_data(filename.removesuffix(".txt")+'_U2.txt', AllPermsBinaryNew , AllDiagsBinaryNew , PureDiagonalsNew)