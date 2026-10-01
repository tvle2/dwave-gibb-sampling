import ast
from pathlib import Path


def start_dwave_connection(device):
    import networkx as nx
    from dwave.cloud import Client
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

def run_dwave(params, path, h, J, solver, num_jobs, metadata=None):
    from annni_selection.acquisition import acquire_jobs
    return acquire_jobs(params, path, h, J, solver, num_jobs,
                        metadata=metadata, compression_level=1, consolidate=True)

def read_results(path):
    """Compatibility convenience; use the streaming decoder for large datasets."""
    from annni_selection.io import iter_sample_rows, read_json
    legacy = Path(path + "_solutions.json.zip")
    if legacy.exists():
        return list(iter_sample_rows(legacy))
    manifest_path = Path(path + "_acquisition.json")
    manifest = read_json(manifest_path)
    if manifest.get("sample_file"):
        return list(iter_sample_rows(manifest_path.parent / manifest["sample_file"]))
    return [row for job in manifest["jobs"]
            for row in iter_sample_rows(manifest_path.parent / job["file"])]

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
