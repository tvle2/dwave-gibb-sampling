import ast
import json
import os
import time
import zlib
from pathlib import Path

import networkx as nx
from dwave.cloud import Client

def start_dwave_connection(device):
    print(f"Attempting to connect to D-Wave solver: {device}...")
    try:
        client = Client.from_config()
        solver = client.get_solver(device)
        hardware_graph = nx.Graph(list(solver.undirected_edges))
        print(
            f"Successfully connected. Hardware graph has "
            f"{hardware_graph.number_of_nodes()} nodes and "
            f"{hardware_graph.number_of_edges()} edges."
        )
        return hardware_graph, solver
    except Exception as error:
        print(f"Error connecting to {device}: {error}")
        return None, None

def run_dwave(params, path, h, J, solver, num_jobs):

    filename_out = path + "_solutions.json.zip"
    if os.path.exists(filename_out):
        print(f"Skipping, results already exist: {filename_out}")
        return
    
    all_samples = []
    total_qpu_time = 0.0

    for job_index in range(num_jobs):
        print(f"Submitting job {job_index + 1}/{num_jobs}")
        while True:
            try:
                future = solver.sample_ising(h, J, answer_mode="raw", **params)
                all_samples.extend(future.samples)
                total_qpu_time += future["timing"]["qpu_access_time"] / 1_000_000
                break
            except Exception as error:
                print(f"- D-Wave API Error: {error}")
                print("- Retrying in 2 seconds...", flush=True)
                time.sleep(2)

    with open(path + "_QPU_time.txt", "w") as file:
        file.write(str(total_qpu_time))
    print(f"    - Total QPU Time: {total_qpu_time:.6f} seconds")


    data  = json.dumps(all_samples).encode('utf-8')
    compressed_data = zlib.compress(data, 1)

    temp_file = filename_out + ".tmp"
    with open(temp_file, "wb") as file:
        file.write(compressed_data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_file, filename_out)
    print(f"Successfully saved results to {filename_out}")

def read_results(path):
    with open(path + "_solutions.json.zip", "rb") as file:
        return json.loads(
            zlib.decompress(file.read())
        )

def load_embeddings(path, n):

    path = Path(path).resolve()
    parts = path.stem.rsplit("_", 2)
    if len(parts) != 3 or int(parts[2]) != n:
        raise ValueError(f"embedding filename must end in _<graph-id>_{n}.txt")
    with open(path, "r") as file:
        mappings = ast.literal_eval(file.read())
    if not mappings or any(set(mapping) != set(range(n)) for mapping in mappings):
        raise ValueError(f"invalid logical mapping in {path.name}")
    physical_qubits = [qubit for mapping in mappings for qubit in mapping.values()]
    if len(physical_qubits) != len(set(physical_qubits)):
        raise ValueError(f"embeddings in {path.name} are not vertex-disjoint")
    return mappings, parts[1]