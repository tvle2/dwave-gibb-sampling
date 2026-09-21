import json
import os
from pathlib import Path

import helpers as dh


# Directory paths
SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
EMBEDDINGS_DIRECTORY = PROJECT_ROOT / "embeddings"
# OUTPUT_DIRECTORY = PROJECT_ROOT / "data"
OUTPUT_DIRECTORY = Path("/Volumes/ThinhLe/data")

# QPU devices
QPU_DEVICES = ["Advantage_system6", "Advantage2_system1"]

# Hyperparameters
N = 8
J1 = 1.0
J2_VALUES = [0.1, 0.5, 0.9]
ENERGY_SCALES = [1.0, 0.1, 0.01, 0.001]
ANNEALING_TIMES_US = (
    [i / 1000 for i in range(5, 100)]
    + [i / 1000 for i in range(100, 1001, 10)]
    + [float(i) for i in range(2, 10)]
    + [float(i) for i in range(10, 110, 10)]
    + [float(i) for i in range(200, 2100, 100)]
)

def annni_couplings(j2):
    couplings = {}
    for i in range(N):
        couplings[(i, (i + 1) % N)] = -J1
    for i in range(N):
        couplings[(i, (i + 2) % N)] = j2
    return couplings

def map_all_embeddings(mappings, logical_couplings, energy_scale):
    h_physical = {}
    J_physical = {}
    for mapping in mappings:
        h_physical.update({mapping[i]: 0.0 for i in range(N)})
        J_physical.update(
            {
                (mapping[u], mapping[v]): coupling * energy_scale
                for (u, v), coupling in logical_couplings.items()
            }
        )
    return h_physical, J_physical

def reads_and_jobs(annealing_time_us):
    if annealing_time_us <= 10:
        return 2000, 40
    return 250, 160


def main():
    for device in QPU_DEVICES:

        # Connect to D-Wave and get the hardware graph
        hardware_graph, solver = dh.start_dwave_connection(device)
        if solver is None:
            raise RuntimeError(f"Could not connect to {device}")

        # Load embeddings
        embedding_file = EMBEDDINGS_DIRECTORY / (
            f"{device}_{solver.graph_id}_{N}.txt"
        )

        if not embedding_file.is_file():
            raise FileNotFoundError(
                f"No embedding file found for {device}, "
                f"graph_id={solver.graph_id}, N={N}: {embedding_file}"
            )

        
        mappings, saved_graph_id = dh.load_embeddings(embedding_file, N)
        if str(solver.graph_id) != saved_graph_id:
            raise ValueError(
                f"Stale embedding for {device}: saved graph_id={saved_graph_id}, "
                f"current graph_id={solver.graph_id}"
            )

        output_directory = OUTPUT_DIRECTORY / device / str(solver.graph_id)
        os.makedirs(output_directory, exist_ok=True)
        print(f"Using {len(mappings)} embeddings from {embedding_file.name}")

        active_qubits_by_embedding = [
            [mapping[logical_qubit] for logical_qubit in range(N)]
            for mapping in mappings
        ]

        embedding_metadata = {
            "device": device,
            "graph_id": str(solver.graph_id),
            "N": N,
            "embedding_file": embedding_file.name,
            "num_embeddings": len(mappings),
            "logical_qubit_order": list(range(N)),
            "active_qubits_by_embedding": active_qubits_by_embedding,
        }

        with open(
            output_directory / f"{device}_embedding_metadata.json",
            "w",
        ) as file:
            json.dump(embedding_metadata, file, indent=2)

        for j2 in J2_VALUES:
            logical_couplings = annni_couplings(j2)
            for energy_scale in ENERGY_SCALES:
                h, J = map_all_embeddings(
                    mappings,
                    logical_couplings,
                    energy_scale,
                )
                unavailable_couplers = [
                    (u, v) for u, v in J if not hardware_graph.has_edge(u, v)
                ]
                if unavailable_couplers:
                    raise ValueError(
                        f"Embedding {embedding_file.name} uses unavailable "
                        f"coupler {unavailable_couplers[0]}"
                    )


                for annealing_time_us in ANNEALING_TIMES_US:
                    reads, jobs = reads_and_jobs(annealing_time_us)
                    params = {
                        "num_reads": reads,
                        "auto_scale": False,
                        "annealing_time": annealing_time_us,
                        "fast_anneal": True,
                    }
                    base_path = output_directory / (
                        f"{device}_ANNNI_N{N}_J2{j2}_es{energy_scale}_at{annealing_time_us}"
                    )
                    print(
                        f"Running {device}: J2={j2}, scale={energy_scale}, "
                        f"time={annealing_time_us} us, {jobs} jobs x {reads} reads"
                    )
                    dh.run_dwave(
                        params,
                        str(base_path),
                        h,
                        J,
                        solver,
                        num_jobs=jobs,
                    )

if __name__ == "__main__":
    main()