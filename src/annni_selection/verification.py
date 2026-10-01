"""Reproducible numerical examples from the revised proposal."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from .states import StateSpace
from .models import classical_gibbs, quantum_gibbs, projected_gibbs, emulating_pairwise, conditional
from .observables import summarize, w_variance_coefficient
from .readout import ReadoutChannel, leading_classical_bias


def exact_report():
    s = StateSpace()
    energies, counts = np.unique(s.energy(.5), return_counts=True)
    components, labels = connected_components(s.flip_adjacency[np.ix_(s.ground, s.ground)], directed=False)
    full = []
    for x, gamma, beta in [(.3, .05, 6), (.3, .2, 1.5), (.3, .4, .75),
                           (.6, .1, 6), (.6, .3, 2), (.6, .4, 1.5), (1, .4, 2.5)]:
        stats = summarize(s, quantum_gibbs(s, beta, gamma))
        full.append({"x": x, "beta": beta, "gamma": gamma,
                     "R1": stats["R1"], "R2": stats["R2"],
                     "D_chi": stats["D_chi_kappa_half"],
                     "projected_R2": summarize(s, projected_gibbs(s, x))["R2"]})
    projected = []
    for x in (.05, .3, .6, 1, 1.5):
        p = projected_gibbs(s, x)
        stats = summarize(s, p)
        pair = emulating_pairwise(s, stats["R1"], stats["R2"])
        projected.append({"x": x, **{k: stats[k] for k in ("R1", "R2", "I_L", "I_L_pair", "W")},
                          "nG_VarW_iid": w_variance_coefficient(s, p),
                          "sector_reweighting_to_zero_W": float(np.exp(-stats["W"] / 2)),
                          "conditional_pairwise_max_error": float(abs(conditional(p, s.target) - conditional(pair, s.target)).max())})
    readout = []
    for beta in (1.5, 2, 6):
        stats = summarize(s, ReadoutChannel.symmetric(8, .01).apply(classical_gibbs(s, beta)))
        readout.append({"beta": beta, "epsilon": .01, "R1": stats["R1"], "R2": stats["R2"],
                        "first_derivatives_at_zero": leading_classical_bias(beta).tolist()})
    return {"energy_degeneracies": dict(zip(map(str, energies), map(int, counts))),
            "ground_states": int(s.ground.sum()), "target_states": int(s.target.sum()),
            "orbits": {k: int(v.sum()) for k, v in s.orbits.items()},
            "symmetry_orbits": int(len(np.unique(s.symmetry_orbits))),
            "correlation_classes": len(s.class_values),
            "ground_graph_components": sorted(np.bincount(labels).tolist()),
            "full_quantum": full, "projected": projected, "classical_readout": readout,
            "interpretation": "exact/ideal-model calculations, not QPU measurements"}
