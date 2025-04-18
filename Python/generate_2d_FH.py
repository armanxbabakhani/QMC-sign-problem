import sys

def generate_fermi_hubbard_pauli_text(L, filename):
    N = L * L  # number of sites per spin sector
    lines = []

    # Map 2D (x, y) coordinates to 1D index
    def idx(x, y):
        return x * L + y

    # Generate hopping terms
    def hopping_terms(spin_offset):
        for x in range(L):
            for y in range(L):
                i = idx(x, y) + spin_offset
                # right neighbor
                if y + 1 < L:
                    j = idx(x, y + 1) + spin_offset
                    z_string = ''.join(f"Z {k} " for k in range(i + 1, j))
                    lines.append(f"-0.5 {z_string}X {i} X {j}")
                # down neighbor
                if x + 1 < L:
                    j = idx(x + 1, y) + spin_offset
                    z_string = ''.join(f"Z {k} " for k in range(i + 1, j))
                    lines.append(f"-0.5 {z_string}X {i} X {j}")

    # Add interaction terms
    for i in range(N):
        up = i
        down = i + N
        lines.append(f"2.0 Z {up} Z {down}")
        lines.append(f"-2.0 Z {up}")
        lines.append(f"-2.0 Z {down}")
        lines.append("2.0")

    # Add hopping terms for spin-up and spin-down
    hopping_terms(spin_offset=0)    # spin-up
    hopping_terms(spin_offset=N)    # spin-down

    # Write to file
    with open(filename, "w") as f:
        for line in lines:
            f.write(line.strip() + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_fh_pauli.py L")
        sys.exit(1)

    try:
        L = int(sys.argv[1])
    except ValueError:
        print("L must be an integer.")
        sys.exit(1)

    output_filename = f"FermiHubbard_2d_t=1_U=2_halffill_L={L}.txt"
    generate_fermi_hubbard_pauli_text(L, output_filename)
    print(f"Hamiltonian written to {output_filename}")
