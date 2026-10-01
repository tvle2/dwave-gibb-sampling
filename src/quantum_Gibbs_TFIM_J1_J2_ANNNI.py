"""Standalone full-Hamiltonian example using the analysis package's basis convention.

Run ``python src/quantum_Gibbs_TFIM_J1_J2_ANNNI.py --beta 1.5 --gamma .2``.
No Qiskit dependency or import-time calculation is required.
"""
import argparse
from annni_selection.states import StateSpace
from annni_selection.models import quantum_gibbs
from annni_selection.observables import summarize


def create_1D(nodes):
    import networkx as nx
    graph = nx.Graph()
    graph.add_edges_from((i, (i + 1) % nodes) for i in range(nodes))
    return graph


def create_ANNNI(nodes):
    graph = create_1D(nodes)
    nn = list(graph.edges())
    nnn = sorted({tuple(sorted((i, (i + 2) % nodes))) for i in range(nodes)})
    graph.add_edges_from(nnn)
    return graph, nn, nnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--kappa', type=float, default=.9)
    parser.add_argument('--n', type=int, default=8)
    args = parser.parse_args()
    space = StateSpace(args.n)
    probabilities = quantum_gibbs(space, args.beta, args.gamma, args.kappa)
    print('Basis: logical spin 0 is the most significant bit; bit 0=-1, bit 1=+1')
    print('States:', len(probabilities), 'normalization:', probabilities.sum())
    for index, probability in enumerate(probabilities):
        print(format(index, f'0{args.n}b'), f'{probability:.12g}')
    if args.n == 8:
        result = summarize(space, probabilities)
        print({key: result[key] for key in ('R1', 'R2', 'I_L', 'I_L_pair', 'W')})


if __name__ == '__main__':
    main()
