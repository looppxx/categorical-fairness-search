"""Leakage-aware multi-filter and wrapper feature selection for numeric tables.

    Continuous inputs are quantile-binned using TRAINING-fold edges. Supply
    categorical column indices explicitly; use original semantic columns, not IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import get_scorer, mutual_info_score
from sklearn.model_selection import StratifiedKFold


class TrainingBinner:
    def __init__(self, bins=8, categorical=()):
        self.bins, self.categorical = bins, tuple(categorical)

    def fit(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or not np.all(np.isfinite(X)) or self.bins < 2:
            raise ValueError("Finite numeric matrix and bins >= 2 required; impute inside training folds")
        self.rules_ = []
        for j, col in enumerate(X.T):
            if j in self.categorical:
                self.rules_.append((True, np.unique(col)))
            else:
                # Interior quantiles; remove extrema and duplicate edges.
                edges = np.unique(np.quantile(col, np.linspace(0, 1, self.bins + 1)[1:-1]))
                edges = edges[(edges > col.min()) & (edges < col.max())]
                self.rules_.append((False, edges))
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != len(self.rules_) or not np.all(np.isfinite(X)):
            raise ValueError("Input does not match fitted binner")
        out = np.empty(X.shape, int)
        for j, (cat, rule) in enumerate(self.rules_):
            if cat:
                mapping = {v: i for i, v in enumerate(rule)}
                out[:, j] = [mapping.get(v, len(rule)) for v in X[:, j]]
            else:
                out[:, j] = np.searchsorted(rule, X[:, j], side="right")
        return out


def entropy_bits(values):
    counts = np.unique(values, return_counts=True)[1]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def mrmr_filter(Z, y, top_n=20, redundancy_weight=1.0, pool=None):
    """Independent categorical mRMR-MID ranking; deterministic index tie breaks."""
    Z, y = np.asarray(Z), np.asarray(y)
    pool = list(range(Z.shape[1])) if pool is None else list(pool)
    relevance = {j: mutual_info_score(Z[:, j], y) for j in pool}
    selected, scores_at_selection = [], {}
    remaining, redundancy = set(pool), {j: 0.0 for j in pool}
    while remaining and len(selected) < top_n:
        scores = {j: relevance[j] - redundancy_weight * (
            redundancy[j] / len(selected) if selected else 0) for j in remaining}
        best = max(remaining, key=lambda j: (scores[j], -j))
        selected.append(best)
        scores_at_selection[best] = float(scores[best])
        remaining.remove(best)
        for j in remaining:
            redundancy[j] += mutual_info_score(Z[:, j], Z[:, best])
    return selected, scores_at_selection


def chi_square_filter(Z, y, top_n=20, pool=None):
    """Independent category-by-class Pearson chi-square ranking, not code sums."""
    Z, y = np.asarray(Z), np.asarray(y)
    pool = list(range(Z.shape[1])) if pool is None else list(pool)
    chi, sparse = {}, {}
    _, classes = np.unique(y, return_inverse=True)
    for j in pool:
        _, codes = np.unique(Z[:, j], return_inverse=True)
        table = np.zeros((codes.max() + 1, classes.max() + 1), int)
        np.add.at(table, (codes, classes), 1)
        if min(table.shape) <= 1:
            chi[j], sparse[j] = 0.0, True
        else:
            stat, _, _, expected = chi2_contingency(table, correction=False)
            chi[j], sparse[j] = float(stat), bool(np.any(expected < 5))
    order = sorted(pool, key=lambda j: (-chi[j], j))[:top_n]
    return order, chi, sparse


def union_top_features(mrmr_ranking, chi_ranking, top_n):
    """Sorted union of independently selected top-N feature indices."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    return np.array(sorted(set(mrmr_ranking[:top_n]) | set(chi_ranking[:top_n])), dtype=int)


@dataclass(frozen=True)
class FilterConfig:
    entropy_threshold: float = 0.05
    bins: int = 8
    top_k: int = 20
    fusion: str = "union"  # union, intersection, rank
    categorical: tuple[int, ...] = ()
    mandatory: tuple[int, ...] = ()
    redundancy_weight: float = 1.0

    def __post_init__(self):
        if not 0 <= self.entropy_threshold <= 1 or self.bins < 2 or self.top_k < 1:
            raise ValueError("Invalid filter settings")
        if self.fusion not in {"union", "intersection", "rank"} or self.redundancy_weight < 0:
            raise ValueError("Invalid fusion or redundancy setting")


class MultiFilter:
    """Entropy > threshold, then mRMR-MID and Pearson contingency chi-square.

    Chi-square is computed on a category-by-target contingency table, avoiding
    the invalid practice of treating signed continuous features as counts.
    Scores rank candidates; sparse expected counts are flagged, not used as
    calibrated significance tests. Mandatory columns bypass ALL drop stages.
    """
    def __init__(self, config=None):
        self.config = config or FilterConfig()

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        c = self.config
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError("Incompatible X, y")
        if any(j < 0 or j >= X.shape[1] for j in c.mandatory + c.categorical):
            raise ValueError("Column index outside X")
        self.binner_ = TrainingBinner(c.bins, c.categorical).fit(X)
        Z = self.binner_.transform(X)
        # Normalize by log2(max possible bins), not observed entropy itself.
        capacities = [max(2, len(rule) if cat else len(rule) + 1)
                      for cat, rule in self.binner_.rules_]
        self.entropy_ = np.array([entropy_bits(Z[:, j]) / np.log2(capacities[j])
                                  for j in range(Z.shape[1])])
        pool = [j for j, h in enumerate(self.entropy_) if h > c.entropy_threshold]
        self.entropy_survivors_ = tuple(pool)
        limit = len(pool) if c.fusion == "rank" else min(c.top_k, len(pool))
        selected, mrmr_scores = mrmr_filter(Z, y, limit, c.redundancy_weight, pool)
        chi_order, chi, sparse = chi_square_filter(Z, y, len(pool), pool)
        k = min(c.top_k, len(pool))
        if c.fusion == "union":
            keep = set(union_top_features(selected, chi_order, k)) if k else set()
        elif c.fusion == "intersection":
            keep = set(selected[:k]) & set(chi_order[:k])
        else:
            rank_m = {j: r for r, j in enumerate(selected)}
            rank_c = {j: r for r, j in enumerate(chi_order)}
            keep = set(sorted(pool, key=lambda j: (rank_m[j] + rank_c[j], j))[:k])
        self.selected_ = np.array(sorted(keep | set(c.mandatory)), dtype=int)
        self.mrmr_order_, self.chi_order_ = selected, chi_order
        self.chi_sparse_, self.mrmr_scores_ = sparse, mrmr_scores
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_]


class HybridFeatureSelector(TransformerMixin, BaseEstimator):
    """Inner folds refit filters independently; wrappers evaluate raw subsets.

    Feature filtering of a wrapper proposal uses ONLY each inner-training fold.
    Final filter is refit on the complete outer-training data. Evaluate the whole
    selector+estimator Pipeline on untouched outer folds for unbiased estimates.
    """
    def __init__(self, estimator, config=None, direction="forward", max_features=10,
                 min_features=1, tolerance=0.001, cv=3, scoring="balanced_accuracy", seed=42):
        self.estimator, self.config, self.direction = estimator, config, direction
        self.max_features, self.min_features = max_features, min_features
        self.tolerance, self.cv, self.scoring, self.seed = tolerance, cv, scoring, seed

    def fit(self, X, y, groups=None):
        X, y = np.asarray(X, float), np.asarray(y)
        c = self.config or FilterConfig()
        if self.direction not in {"forward", "backward"}:
            raise ValueError("direction must be forward or backward")
        if not 1 <= self.min_features <= self.max_features or self.tolerance < 0:
            raise ValueError("Invalid feature bounds or tolerance")
        if isinstance(self.cv, int):
            if groups is not None:
                raise ValueError("Provide a group-aware CV splitter when groups are supplied")
            split = StratifiedKFold(self.cv, shuffle=True, random_state=self.seed)
        else:
            split = self.cv
        scorer = get_scorer(self.scoring)
        folds = []
        # No global supervised filter is fit before inner validation.
        candidate_pool = set(c.mandatory)
        for train, val in split.split(X, y, groups):
            filt = MultiFilter(c).fit(X[train], y[train])
            allowed = set(filt.selected_.tolist())
            candidate_pool |= allowed
            folds.append((train, val, allowed))
        if not candidate_pool:
            raise ValueError("All features removed; relax threshold or add justified mandatory columns")
        cache = {}
        def evaluate(subset):
            key = tuple(sorted(subset))
            if key not in cache:
                values = []
                for train, val, allowed in folds:
                    use = sorted(set(key) & allowed)
                    if use:
                        model = clone(self.estimator).fit(X[train][:, use], y[train])
                        values.append(scorer(model, X[val][:, use], y[val]))
                    else:
                        dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), y[train])
                        values.append(scorer(dummy, np.zeros((len(val), 1)), y[val]))
                cache[key] = float(np.mean(values))
            return cache[key]
        mandatory = set(c.mandatory)
        chosen = mandatory.copy() if self.direction == "forward" else candidate_pool.copy()
        current = evaluate(chosen)
        self.history_ = [{"features": sorted(chosen), "score": current}]
        if len(mandatory) > self.max_features:
            raise ValueError("Mandatory feature count exceeds max_features")
        if self.direction == "forward":
            while len(chosen) < min(self.max_features, len(candidate_pool)):
                options = [(evaluate(chosen | {j}), j) for j in sorted(candidate_pool - chosen)]
                if not options:
                    break
                value, j = max(options, key=lambda z: (z[0], -z[1]))
                if len(chosen) >= self.min_features and value <= current + self.tolerance:
                    break
                chosen.add(j)
                current = value
                self.history_.append({"features": sorted(chosen), "score": current})
        else:
            while len(chosen) > max(self.min_features, len(mandatory)):
                options = [(evaluate(chosen - {j}), j) for j in sorted(chosen - mandatory)]
                value, j = max(options, key=lambda z: (z[0], -z[1]))
                if len(chosen) <= self.max_features and value < current - self.tolerance:
                    break
                chosen.remove(j)
                current = value
                self.history_.append({"features": sorted(chosen), "score": current})
        self.filter_ = MultiFilter(c).fit(X, y)
        # The full-training filter determines the deployed subset, never test data.
        self.selected_ = np.array(sorted(chosen & set(self.filter_.selected_.tolist())), dtype=int)
        if not len(self.selected_):
            raise ValueError("No selected feature survives refit; use stable features or relax filters")
        self.n_features_in_, self.wrapper_evaluations_ = X.shape[1], len(cache)
        self.inner_score_ = current  # selection score, not an unbiased final accuracy
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature schema changed after fitting")
        return X[:, self.selected_]

    def get_support(self, indices=False):
        if indices:
            return self.selected_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_] = True
        return mask
