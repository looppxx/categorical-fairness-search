"""Seed-paired comparisons within ONE dataset/model/metric cell."""
import numpy as np
from scipy.stats import wilcoxon


def paired_comparison(baseline, candidate, seed=42, bootstrap_samples=5000):
    """Ordered arrays must contain matching seeds. Larger values mean better.

    Seeds quantify algorithm randomness, not independent dataset replications.
    Use a dataset-level hierarchical analysis for cross-dataset generalization.
    """
    a, b = np.asarray(baseline, float), np.asarray(candidate, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a + b)):
        raise ValueError("At least two finite matched observations required")
    d = b - a
    rng = np.random.default_rng(seed)
    samples = rng.choice(d, size=(bootstrap_samples, len(d)), replace=True).mean(axis=1)
    p = 1.0 if np.all(d == 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
    return {"pairs": len(d), "mean_paired_difference": float(d.mean()),
            "bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "wilcoxon_p": p,
            "paired_superiority_with_half_ties": float(np.mean(d > 0) + 0.5 * np.mean(d == 0))}


def holm_adjust(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite p-values in [0,1] required")
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result
