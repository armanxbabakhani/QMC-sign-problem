import numpy as np
import sys

def generate_heisenberg_hamiltonian_file(N):
    """
    Generates a text file for the Hamiltonian of a Heisenberg model on a triangular graph.

    Args:
        N (int): Number of nodes in the graph.
        filename (str): Name of the output file.

    The file format will be:
        1.0 1 X 2 X
        1.0 1 Y 2 Y
        1.0 1 Z 2 Z
        ...
        1.0 (N-1) X N X
        1.0 (N-1) Y N Y
        1.0 (N-1) Z N Z
    """

    filename="./Inputs/Triangular_Ladder_Heisenberg/Triangular_Heis_n="+N+".txt"

    if int(N) < 2:
        raise ValueError("N must be at least 2 to form a triangular graph.")

    interactions = []
    for i in range(3, int(N)+1):
        for axis in ["X", "Y", "Z"]:
            interactions.append(f"1.0 {i-2} {axis} {i} {axis}")
            interactions.append(f"1.0 {i-1} {axis} {i} {axis}")

    with open(filename, "w") as file:
        file.write("\n".join(interactions) + "\n")


N = sys.argv[1]
generate_heisenberg_hamiltonian_file(N)