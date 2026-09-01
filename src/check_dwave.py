# Check D-Wave graph id

from dwave.cloud import Client

devices = [
    "Advantage_system6",
    "Advantage2_system1",
    "Advantage_system4",
]

with Client.from_config(client="qpu") as client:
    for device in devices:
        solver = client.get_solver(device)

        print(
            device,
            "graph_id =",
            solver.graph_id,
        )