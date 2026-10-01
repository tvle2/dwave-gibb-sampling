"""Cluster-aware summaries. Independent-read formulas are planning tools only."""
import numpy as np
from .observables import summarize


def split_jobs(n_jobs, block_size=1, seed=0):
    if n_jobs < 4 or block_size < 1:
        raise ValueError("Need at least four jobs and a positive block size")
    blocks = np.arange(n_jobs) // block_size
    unique = np.unique(blocks)
    if len(unique) < 4:
        raise ValueError("Need at least four independent job blocks for a split")
    shuffled = np.random.default_rng(seed).permutation(unique)
    train_blocks = shuffled[:len(unique) // 2]
    return {"train": np.flatnonzero(np.isin(blocks, train_blocks)).tolist(),
            "test": np.flatnonzero(~np.isin(blocks, train_blocks)).tolist(),
            "block_ids": blocks.tolist(), "block_size": block_size, "seed": seed}


def grouped_folds(indices, block_ids, n_folds=3):
    indices, block_ids = np.asarray(indices), np.asarray(block_ids)
    blocks = np.unique(block_ids[indices])
    n_folds = min(n_folds, len(blocks))
    if n_folds < 2:
        raise ValueError("Grouped CV needs at least two training blocks")
    for group in np.array_split(blocks, n_folds):
        val = indices[np.isin(block_ids[indices], group)]
        train = indices[~np.isin(block_ids[indices], group)]
        yield train, val


def block_resample(indices, block_ids, rng):
    indices, block_ids = np.asarray(indices), np.asarray(block_ids)
    groups = [indices[block_ids[indices] == k] for k in np.unique(block_ids[indices])]
    if len(groups) < 2:
        raise ValueError("Bootstrap needs at least two independent job blocks")
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def score_jobs(counts, probabilities, mask=None, eligible=None):
    """Eq.25: equal embedding weights within job, then equal job weights.

    Zero conditional occupancy is omitted identically for every candidate.
    Unsupported observed events give -inf, never a hidden probability floor.
    """
    counts, p = np.asarray(counts), np.asarray(probabilities, float)
    if counts.ndim != 3 or p.shape != counts.shape[1:]:
        raise ValueError("Expected counts[J,E,S] and probabilities[E,S]")
    if np.any(p < 0) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError("Predictions must be normalized full distributions")
    if mask is not None:
        counts, p = counts[:, :, mask], p[:, mask]
        sums = p.sum(1, keepdims=True)
        p = np.divide(p, sums, out=np.zeros_like(p), where=sums > 0)
    occupied = counts.sum(2)
    use = occupied > 0
    if eligible is not None:
        use &= np.asarray(eligible)[None, :]
    logp = np.log(p, out=np.full_like(p, -np.inf), where=p > 0)
    # 0 log 0 = 0; observations assigned probability 0 retain infinite loss.
    weighted = np.zeros(counts.shape, float)
    np.multiply(counts, logp[None, :, :], out=weighted, where=counts > 0)
    scores = np.divide(weighted.sum(2), occupied, out=np.zeros_like(occupied, float), where=occupied > 0)
    values = np.sum(np.where(use, scores, 0), axis=1)
    return np.divide(values, use.sum(1), out=np.full(len(counts), np.nan), where=use.sum(1) > 0)


def score_mean(scores):
    scores = np.asarray(scores)
    values = scores[~np.isnan(scores)]
    return float(values.mean()) if len(values) else float("nan")


def paired_score_interval(counts, a, b, indices, block_ids, *, mask=None,
                          eligible=None, repeats=500, seed=0):
    """a,b have [draw,E,S]; one calibration draw applies to a WHOLE resample."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("Paired predictions must share draw/embedding/state axes")
    rng = np.random.default_rng(seed)
    jobs_a = np.array([score_jobs(counts, x, mask, eligible) for x in a])
    jobs_b = np.array([score_jobs(counts, x, mask, eligible) for x in b])
    estimates = np.array([score_mean(x[indices]) - score_mean(y[indices]) for x, y in zip(jobs_a, jobs_b)])
    if not np.isfinite(estimates).all():
        return {"difference": None, "interval": None, "reason": "zero predictive support or no eligible observations"}
    boot = []
    for _ in range(repeats):
        idx = block_resample(indices, block_ids, rng)
        d = int(rng.integers(len(a)))
        boot.append(score_mean(jobs_a[d, idx]) - score_mean(jobs_b[d, idx]))
    return {"difference": float(estimates[0]), "calibration_range": [float(estimates.min()), float(estimates.max())],
            "interval": np.quantile(boot, [.025, .975]).tolist(), "repeats": repeats,
            "interpretation": "paired whole-block bootstrap with calibration sensitivity draws"}


def observable_intervals(space, counts, indices, block_ids, repeats=500, seed=0, min_train_counts=None):
    """Embedding-resolved log invariants; never logs of a pooled-embedding histogram."""
    rng = np.random.default_rng(seed)
    keys = ["R1", "R2", "I_L", "I_L_pair", "W", "D_chi_kappa_half", "P_T", "P_G"]
    total = counts[indices].sum(0)
    base = [summarize(space, c) for c in total]
    draws = np.full((repeats, len(total), len(keys)), np.nan)
    for r in range(repeats):
        idx = block_resample(indices, block_ids, rng)
        for e, c in enumerate(counts[idx].sum(0)):
            stats = summarize(space, c)
            draws[r, e] = [np.nan if stats[k] is None else stats[k] for k in keys]
    result = []
    for e, point in enumerate(base):
        record = {"embedding": e, "estimates": point, "intervals": {}, "valid_draw_fraction": {}}
        for k, key in enumerate(keys):
            valid = np.isfinite(draws[:, e, k])
            record["valid_draw_fraction"][key] = float(valid.mean())
            eligible = True if min_train_counts is None else bool(min_train_counts[e])
            if key in {"I_L", "I_L_pair", "W"} and not eligible:
                record["intervals"][key] = None
            elif valid.all():
                record["intervals"][key] = np.quantile(draws[:, e, k], [.025, .975]).tolist()
            else:
                record["intervals"][key] = None
        # These four quantities share the SAME bootstrap draws and covariance.
        cols = [keys.index(k) for k in ("I_L", "I_L_pair", "W")]
        cov_draws = draws[:, e, cols]
        record["curvature_covariance_order"] = ["I_L", "I_L_pair", "W"]
        record["curvature_covariance"] = np.cov(cov_draws.T).tolist() if np.isfinite(cov_draws).all() and repeats > 1 else None
        result.append(record)
    return result


def holm(pvalues, alpha=0.05):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0,1]; use 1 for missing prespecified tests")
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return {"adjusted": result.tolist(), "reject": (result <= alpha).tolist(), "alpha": alpha}


def wild_cluster_omnibus(counts, probabilities, indices, block_ids, mask, eligible, *, repeats=999, seed=0):
    """Exploratory multiplier test of a frozen conditional prediction.

    Does not assume independent embeddings/reads. Independent BLOCKS and a
    correctly specified nuisance envelope are still required; finite-sample size
    must be validated by the synthetic study before confirmatory interpretation.
    """
    counts, p = counts[:, :, mask], probabilities[:, mask]
    if np.any(p[np.asarray(eligible, bool)].sum(1) <= 0):
        return {"pvalue": None, "reason": "zero conditional predictive support"}
    totals = p.sum(1, keepdims=True)
    p = np.divide(p, totals, out=np.zeros_like(p), where=totals > 0)
    eligible = np.asarray(eligible, bool)
    if not eligible.any() or np.any(p[eligible] <= 0):
        return {"pvalue": None, "reason": "empty eligibility or zero conditional support"}
    counts, p = counts[:, eligible], p[eligible]
    n = counts.sum(2)
    # Equal embedding weights; zero-occupancy jobs contribute zero residual.
    freq = np.divide(counts, n[:, :, None], out=np.zeros_like(counts, float), where=n[:, :, None] > 0)
    residual = np.where((n > 0)[:, :, None], freq - p, 0) / np.sqrt(p)
    ids, blocks = np.asarray(indices), np.asarray(block_ids)
    groups = [ids[blocks[ids] == k] for k in np.unique(blocks[ids])]
    if len(groups) < 4:
        return {"pvalue": None, "reason": "fewer than four test blocks"}
    u = np.array([residual[g].sum(0).ravel() for g in groups])
    observed = float(np.dot(u.sum(0), u.sum(0)))
    centered = (u - u.mean(0)) * np.sqrt(len(u) / (len(u) - 1))
    gram = centered @ centered.T
    rng = np.random.default_rng(seed)
    multipliers = rng.choice([-1.0, 1.0], size=(repeats, len(u)))
    null = np.einsum("bi,ij,bj->b", multipliers, gram, multipliers)
    return {"pvalue": float((1 + np.sum(null >= observed)) / (repeats + 1)),
            "statistic": observed, "blocks": len(groups), "repeats": repeats,
            "interpretation": "exploratory cluster multiplier test; requires size calibration"}


def dependence_diagnostics(space, state_ids, offsets, max_lag=5):
    """Read and job order summaries without treating parallel copies as replications."""
    states = np.asarray(state_ids)
    if states.ndim != 2 or len(states) != offsets[-1]:
        raise ValueError("Invalid synchronized state-ID array")
    read_mean = np.empty(len(states))
    sums = np.zeros(states.shape[1])
    for start in range(0, len(states), 4096):
        indicator = space.target[states[start:start + 4096]]
        read_mean[start:start + len(indicator)] = indicator.mean(1)
        sums += indicator.sum(0)
    means = sums / len(states)
    variances = means * (1 - means)
    e = states.shape[1]
    off_cov = ((e * e * read_mean.var() - variances.sum()) / (e * (e - 1))) if e > 1 else None
    def acf(series):
        series = series - series.mean()
        den = float(series @ series)
        return [float(series[:-k] @ series[k:] / den) if den > 0 and k < len(series) else None
                for k in range(1, max_lag + 1)]
    job_mean = np.array([read_mean[a:b].mean() for a, b in zip(offsets[:-1], offsets[1:])])
    return {"mean_off_diagonal_embedding_covariance_of_T": off_cov,
            "read_acf_mean_T": acf(read_mean), "job_acf_mean_T": acf(job_mean),
            "job_T_occupancy": job_mean.tolist(), "time_units": "read/job index, not physical correlation time"}
