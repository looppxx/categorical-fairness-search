"""Fixed subgroup semantics, budget enforcement, and auditable objective calls."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SubgroupSpace:
    """Conjunctions of integer categorical features; -1 denotes a wildcard.

    Fit bins and category dictionaries on development data only, then reuse them.
    This is proposed generic encoding, not a claim about native FairRec encoding.
    """
    cardinalities: tuple[int, ...]
    max_order: int = 3

    def __post_init__(self):
        if not self.cardinalities or any(c < 1 for c in self.cardinalities):
            raise ValueError("Positive cardinalities required")
        if not 1 <= self.max_order <= len(self.cardinalities):
            raise ValueError("Invalid max_order")

    @property
    def dimension(self):
        return len(self.cardinalities)

    @property
    def lower_bound(self):
        return -1

    def validate(self, candidate):
        raw = np.asarray(candidate)
        if raw.shape != (self.dimension,) or not np.all(np.isfinite(raw)):
            raise ValueError("Candidate must have one finite integer per feature")
        g = raw.astype(int)
        if not np.array_equal(g, raw):
            raise ValueError("Fractional category code")
        if any(v < -1 or v >= c for v, c in zip(g, self.cardinalities)):
            raise ValueError("Category outside fixed vocabulary")
        if not 1 <= np.count_nonzero(g >= 0) <= self.max_order:
            raise ValueError("Candidate order outside fixed grammar")
        return tuple(int(v) for v in g)

    def repair(self, candidate, rng):
        g = np.rint(candidate).astype(int)
        g = np.clip(g, -1, np.asarray(self.cardinalities) - 1)
        active = np.flatnonzero(g >= 0)
        if len(active) > self.max_order:
            g[rng.choice(active, len(active) - self.max_order, replace=False)] = -1
        if not np.any(g >= 0):
            j = int(rng.integers(self.dimension))
            g[j] = rng.integers(self.cardinalities[j])
        return g

    def sample(self, rng):
        g = np.full(self.dimension, -1)
        active = rng.choice(self.dimension, rng.integers(1, self.max_order + 1), replace=False)
        for j in active:
            g[j] = rng.integers(self.cardinalities[j])
        return g

    def mutate(self, g, rng):
        h = np.array(g, copy=True)
        j = int(rng.integers(self.dimension))
        choices = [v for v in range(-1, self.cardinalities[j]) if v != h[j]]
        h[j] = rng.choice(choices)
        return self.repair(h, rng)

    def enumerate(self):
        for order in range(1, self.max_order + 1):
            for dims in itertools.combinations(range(self.dimension), order):
                for vals in itertools.product(*(range(self.cardinalities[j]) for j in dims)):
                    g = [-1] * self.dimension
                    for j, v in zip(dims, vals):
                        g[j] = v
                    yield tuple(g)

    def mask(self, X, g):
        g = np.asarray(self.validate(g))
        active = np.flatnonzero(g >= 0)
        return np.all(X[:, active] == g[active], axis=1)


@dataclass(frozen=True)
class GroupResult:
    support: int
    complement_support: int
    utility: float | None
    reference: float | None
    gap: float | None
    valid: bool


class FixedOracle:
    """Immutable score data; greater utility is better; gap = complement - group.

    Replace this callable with an audited native score adapter for reproduction.
    Use one value per user to avoid weighting prolific users more heavily.
    """
    def __init__(self, X, utility, space, min_support=20):
        X = np.asarray(X)
        u = np.asarray(utility, dtype=float)
        if X.ndim != 2 or X.shape != (len(u), space.dimension) or not len(u):
            raise ValueError("Incompatible X and per-user utility")
        if not np.all(np.isfinite(X)) or not np.array_equal(X, X.astype(int)):
            raise ValueError("X must contain finite integer category codes")
        if any(np.any((X[:, j] < 0) | (X[:, j] >= c)) for j, c in enumerate(space.cardinalities)):
            raise ValueError("Data outside fixed category vocabulary")
        if not np.all(np.isfinite(u)) or min_support < 1:
            raise ValueError("Finite utilities and positive support required")
        self.X, self.utility = X.astype(int).copy(), u.copy()
        self.X.flags.writeable = self.utility.flags.writeable = False
        self.space, self.min_support = space, int(min_support)
        h = hashlib.sha256(self.X.tobytes() + self.utility.tobytes())
        h.update(repr((self.space, self.min_support, "complement-minus-group-v1")).encode())
        self._fingerprint = h.hexdigest()

    @property
    def fingerprint(self):
        return self._fingerprint

    def __call__(self, g):
        m = self.space.mask(self.X, g)
        n, r = int(m.sum()), int((~m).sum())
        if min(n, r) < self.min_support:
            return GroupResult(n, r, None, None, None, False)
        a, b = float(self.utility[m].mean()), float(self.utility[~m].mean())
        return GroupResult(n, r, a, b, b - a, True)


class BudgetExhausted(Exception):
    pass


class EvaluationSession:
    """Every proposal costs one budget unit, including repeats and infeasible groups.

    Log expensive unique calls separately. Cache is scoped to one run. All engines
    maximize score; sign=-1 searches the opposite end using the identical oracle.
    """
    def __init__(self, oracle, budget, sign=1, metadata=None, objective_field="gap", unique_budget=None):
        if budget < 1 or sign not in (-1, 1):
            raise ValueError("Positive budget and sign +/-1 required")
        self.oracle, self.budget, self.sign = oracle, int(budget), sign
        if unique_budget is not None and unique_budget < 1:
            raise ValueError("Unique budget must be positive")
        self.unique_budget = unique_budget
        if objective_field not in {"gap", "utility"}:
            raise ValueError("Objective must be gap or utility")
        self.objective_field = objective_field
        self.metadata = dict(metadata or {})
        self.cache, self.logs = {}, []
        self.best, self.best_group = -np.inf, None
        self.started = time.perf_counter()

    @property
    def remaining(self):
        calls_left = self.budget - len(self.logs)
        return calls_left if self.unique_budget is None else min(calls_left, self.unique_budget - len(self.cache))

    def evaluate(self, candidate):
        if not self.remaining:
            raise BudgetExhausted()
        key = self.oracle.space.validate(candidate)
        cached = key in self.cache
        if not cached:
            self.cache[key] = self.oracle(key)
        result = self.cache[key]
        score = self.sign * getattr(result, self.objective_field) if result.valid else -np.inf
        if score > self.best:
            self.best, self.best_group = float(score), key
        self.logs.append({**self.metadata, "proposal": len(self.logs) + 1,
                          "unique_calls": len(self.cache), "cached": cached,
                          "candidate": list(key), "valid": result.valid,
                          "support": result.support, "gap": result.gap,
                          "utility": result.utility, "sign": self.sign,
                          "objective_field": self.objective_field,
                          "best_objective": self.best if np.isfinite(self.best) else None,
                          "elapsed_seconds": time.perf_counter() - self.started,
                          "oracle_hash": self.oracle.fingerprint})
        return score

    def write_jsonl(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.logs:
                f.write(json.dumps(row, allow_nan=False) + "\n")


class NativeSearchAdapter:
    """Supply a verified native loop; route ALL fitness calls through session.

    runner(session, rng, initial) must respect the supplied encoding, oracle,
    budget, and frozen initialization. A different grammar needs a different
    oracle/space adapter, shared by every comparator in that experiment.
    """
    def __init__(self, name: str, runner: Callable, revision: str):
        if not revision.strip():
            raise ValueError("Record the upstream commit or artifact revision")
        self.name, self.runner, self.revision = name, runner, revision

    def run(self, session, rng, initial=None):
        return self.runner(session, rng, initial)
