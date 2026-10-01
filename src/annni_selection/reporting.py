"""Optional standalone scientific plots; no model fitting occurs here."""
from pathlib import Path
import numpy as np
from .io import read_json


def plot_report(report_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = read_json(report_path)
    rows = report["embedding_observables"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    index = np.arange(len(rows))
    for key, ax in zip(("R1", "R2", "I_L", "W"), axes.flat):
        point = np.array([r["estimates"][key] if r["estimates"][key] is not None else np.nan for r in rows])
        ax.scatter(index, point, s=12, label=key)
        for e, row in enumerate(rows):
            interval = row["intervals"].get(key)
            if interval is not None:
                ax.vlines(e, interval[0], interval[1], color="C0", lw=.8)
        if key == "I_L":
            ax.scatter(index, [np.nan if r["estimates"]["I_L_pair"] is None else r["estimates"]["I_L_pair"] for r in rows],
                       marker="x", color="C1", label="pairwise-required curvature")
            ax.legend(fontsize=8)
        ax.axhline(0, color="0.5", lw=.6)
        ax.set(xlabel="Embedding", ylabel=key)
    figure.suptitle("Observed held-out contrasts; intervals retain job blocks\nReadout and mixtures can violate latent-model identities", fontsize=11)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
