import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dwave.cloud import Client
import minorminer.subgraph


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


def Start_DWave_connection(device):
    client = Client.from_config()
    DWave_solver = client.get_solver(device)
    A = DWave_solver.undirected_edges
    connectivity_graph = nx.Graph(list(A))
    return connectivity_graph, DWave_solver

N = 8

G, NN, NNN = create_ANNNI(N)


# QA_device = "Advantage_system6"
#QA_device = "Advantage_system4"
QA_device = "Advantage2_system1"

result_map = {}

for rand_restart in range(10):
	hardware_Graph, solver = Start_DWave_connection(QA_device)
	unique_string_id = str(solver.graph_id)
	list_of_disjoint_embeddings = []
	for _ in range(1000):
		start_hardware_nodes = len(list(hardware_Graph.nodes()))
		s = minorminer.subgraph.find_subgraph(G, hardware_Graph, timeout=9000)
		print(s)
		if s == {}:
			print("Empty")
			continue
		used_qubits = list(s.values())
		print(used_qubits)
		hardware_Graph.remove_nodes_from(used_qubits)
		end_hardware_nodes = len(list(hardware_Graph.nodes()))
		print("remaining hardware nodes:", len(list(hardware_Graph.nodes())))
		assert start_hardware_nodes-N == end_hardware_nodes
		list_of_disjoint_embeddings.append(s)
		print(len(list_of_disjoint_embeddings))
	result_map[len(list_of_disjoint_embeddings)] = list_of_disjoint_embeddings

print("Most independent embeddings found", max(result_map))

file = open(f"embeddings/{QA_device}_{unique_string_id}_{N}.txt", "w")
file.write(str(result_map[max(result_map)]))
file.close()


"""
Embeddings can then be read in:
file = open("embeddings/{QA_device}_{unique_string_id}_{N}.txt", "r")
embeddings = ast.literal_eval(file.read())
file.close()
"""
