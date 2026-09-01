import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import SparsePauliOp, DensityMatrix

"""
DWave hardware:
Ferromagnetic is negative
Antiferromagnetic is positive
"""

def create_1D(nodes):
	G = nx.Graph()
	for idx in range(nodes-1):
		G.add_edge(idx, idx+1)
	G.add_edge(0, nodes-1)
	return G
def create_ANNNI(nodes):
	G = create_1D(nodes)
	NN = list(G.edges())
	NNN = []
	for idx in range(nodes-2):
		G.add_edge(idx, idx+2)
		NNN.append((idx, idx+2))
	G.add_edge(0, nodes-2)
	G.add_edge(1, nodes-1)
	NNN.append((0, nodes-2))
	NNN.append((1, nodes-1))
	return G, NN, NNN

beta = 0.1
J1 = 1
J2 = 0.9
Gamma = 1
N = 8

G, NN, NNN = create_ANNNI(N)
J = {}
for edge in NN:
	J[edge] = -1*J1
for edge in NNN:
	J[edge] = J2

terms = []

for i in range(N):
	terms.append(("X", [i], -Gamma))

for (i, j), Jij in J.items():
	terms.append(("ZZ", [i, j], Jij))

H = SparsePauliOp.from_sparse_list(terms, num_qubits=N)
H_matrix = H.to_matrix()

eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
print("Spectral range:", max(eigenvalues), min(eigenvalues))
boltzmann_factors = np.exp(-beta * eigenvalues, dtype=np.complex128)
Z = np.sum(boltzmann_factors, dtype=np.complex128)
thermal_weights = boltzmann_factors / Z

rho_matrix = (eigenvectors @ np.diag(thermal_weights) @ eigenvectors.conj().T)

rho = DensityMatrix(rho_matrix)

Pz = rho.probabilities()

print(len(Pz), max(Pz))

for index, p in enumerate(Pz):
	bitstring = format(index, f"0{N}b")
	print(bitstring, p)
