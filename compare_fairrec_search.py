"""Controlled synthetic FairRec search comparison (Python 3.10+, NumPy).

Run: python compare_fairrec_search.py --runs 3
Optional exact native search (only fitness is replaced):
  python compare_fairrec_search.py --native-source C:/path/FairRec/fairrec.py

The default DPSO is an independent thermal double-ended reference, NOT an
exact FairRec replication. Native FairRec searches recommendation utility at
both ends; this mock instead searches a fixed synthetic disparity at both ends.
No result here measures fairness of a trained recommender or proves superiority.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Space:
    cardinalities: tuple[int, ...]

    def __post_init__(self):
        if not self.cardinalities or any(c < 1 for c in self.cardinalities):
            raise ValueError("Each categorical feature must have at least one value")

    @property
    def dimension(self):
        return len(self.cardinalities)

    def validate(self, vector):
        x = np.asarray(vector)
        if (x.shape != (self.dimension,) or not np.isfinite(x).all()
                or not np.equal(x, np.floor(x)).all()
                or np.any(x < 0) or np.any(x >= self.cardinalities)):
            raise ValueError(f"Invalid categorical subgroup: {vector}")
        return tuple(int(v) for v in x)

    def sample(self, rng, n=1):
        return rng.integers(0, self.cardinalities, size=(n, self.dimension))

    def neighbor(self, x, rng):
        y = x.copy()
        mutable = [j for j, c in enumerate(self.cardinalities) if c > 1]
        if not mutable:
            return y
        j = int(rng.choice(mutable))
        # Uniformly choose a DIFFERENT category (a bit flip for binary features).
        v = int(rng.integers(self.cardinalities[j] - 1))
        y[j] = v + (v >= y[j])
        return y


@dataclass(frozen=True)
class MockDisparity:
    """Immutable, deterministic categorical NK-style landscape in [0,1].

    Local tables couple each coordinate with its next two neighbors. A second
    term rewards complete three-feature patterns, with weak partial matches,
    creating deceptive combinatorial interactions. This is a synthetic score,
    not an estimate of a population fairness metric. Every cell is feasible.
    """
    space: Space
    tables: tuple[tuple[float, ...], ...]
    patterns: tuple[int, ...]

    @classmethod
    def create(cls, space, seed=2026):
        rng = np.random.default_rng(seed)
        c, d = space.cardinalities, space.dimension
        tables = tuple(tuple(float(v) for v in rng.random(
            c[j] * c[(j + 1) % d] * c[(j + 2) % d])) for j in range(d))
        patterns = tuple(int(rng.integers(v)) for v in c)
        return cls(space, tables, patterns)

    def __call__(self, subgroup_vector):
        g = self.space.validate(subgroup_vector)
        c, d = self.space.cardinalities, self.space.dimension
        local = []
        for j, table in enumerate(self.tables):
            k, l = (j + 1) % d, (j + 2) % d
            index = (g[j] * c[k] + g[k]) * c[l] + g[l]
            local.append(table[index])
        traps = []
        for start in range(0, d, 3):
            block = range(start, min(start + 3, d))
            matches = sum(g[j] == self.patterns[j] for j in block)
            n = len(block)
            traps.append(1.0 if matches == n else (n - 1 - matches) / n)
        return float(0.65 * sum(local) / d + 0.35 * sum(traps) / len(traps))

    @property
    def fingerprint(self):
        raw = json.dumps((self.space.cardinalities, self.tables, self.patterns))
        return hashlib.sha256(raw.encode()).hexdigest()


# All engines receive this same fixed callable through Tracker, never a
# method-specific objective. The default function is also directly importable.
DEFAULT_OBJECTIVE = MockDisparity.create(Space((2, 4, 3) * 8))


def evaluate_subgroup_disparity(subgroup_vector):
    return DEFAULT_OBJECTIVE(subgroup_vector)


class BudgetExhausted(Exception):
    pass


class Tracker:
    """Charge EVERY request, including repeats, initializers and scouts.

    Identical caching and archive policy for all engines. Traces are kept in
    memory during timing; file I/O is excluded from reported execution time.
    """
    def __init__(self, objective, budget, threshold, block_size):
        self.objective, self.space = objective, objective.space
        self.budget, self.threshold, self.block_size = budget, threshold, block_size
        self.cache, self.high, self.trace = {}, set(), []
        self.best, self.worst = -math.inf, math.inf
        self.best_group, self.best_query = None, 0

    @property
    def calls(self):
        return len(self.trace)

    def evaluate(self, vector):
        if self.calls >= self.budget:
            raise BudgetExhausted
        key = self.space.validate(vector)
        cached = key in self.cache
        if not cached:
            self.cache[key] = float(self.objective(key))
        score = self.cache[key]
        if not math.isfinite(score):
            raise ValueError("Objective must return a finite score")
        if score > self.best:
            self.best, self.best_group, self.best_query = score, key, self.calls + 1
        self.worst = min(self.worst, score)
        if score >= self.threshold:
            self.high.add(key)
        self.trace.append(dict(query=self.calls + 1, subgroup=key, score=score,
                               best=self.best, cached=cached,
                               diversity=len(self.high)))
        return score


class End:
    """A signed search view; the shared disparity function is never changed."""
    def __init__(self, tracker, stop, sign):
        self.tracker, self.stop, self.sign = tracker, stop, sign
        self.space = tracker.space

    @property
    def remaining(self):
        return self.stop - self.tracker.calls

    def evaluate(self, x):
        if self.remaining <= 0:
            raise BudgetExhausted
        return self.sign * self.tracker.evaluate(x)


def initialize(end, initial):
    pop = initial.copy()
    scores = np.array([end.evaluate(x) for x in pop])
    return pop, scores


def genetic_algorithm(end, rng, initial):
    pop, scores = initialize(end, initial)
    n, d = pop.shape
    def select():
        indices = rng.integers(n, size=3)
        return pop[indices[np.argmax(scores[indices])]]
    while end.remaining:
        children, values = [], []
        for _ in range(min(n, end.remaining)):
            a, b = select(), select()
            child = np.where(rng.random(d) < 0.5, a, b).copy()
            for j in np.flatnonzero(rng.random(d) < 1 / d):
                if end.space.cardinalities[j] == 1:
                    continue
                v = int(rng.integers(end.space.cardinalities[j] - 1))
                child[j] = v + (v >= child[j])
            children.append(child)
            values.append(end.evaluate(child))
        if len(children) == n:
            # One elite, using its existing score; no free objective queries.
            worst = int(np.argmin(values))
            if scores.max() > values[worst]:
                children[worst] = pop[int(np.argmax(scores))].copy()
                values[worst] = float(scores.max())
            pop, scores = np.array(children), np.array(values)


def differential_evolution(end, rng, initial):
    """DE/rand/1/bin on bounded random keys, categorical argmax decoding.

    F=0.7, CR=0.8, clipping repair to [0,1]. One key per category; nominal
    category codes are NEVER subtracted. One forced donor KEY does not imply
    a changed decoded category; duplicate decodings still consume queries.
    """
    pop, scores = initialize(end, initial)
    n = len(pop)
    offsets = np.r_[0, np.cumsum(end.space.cardinalities)]
    keys = rng.uniform(0, 0.49, size=(n, int(offsets[-1])))
    for i, x in enumerate(pop):
        for j, v in enumerate(x):
            keys[i, offsets[j] + v] = rng.uniform(0.51, 1)
    while end.remaining:
        old, old_scores = keys.copy(), scores.copy()
        for i in range(min(n, end.remaining)):
            others = np.delete(np.arange(n), i)
            a, b, c = rng.choice(others, 3, replace=False)
            donor = np.clip(old[a] + 0.7 * (old[b] - old[c]), 0, 1)
            mask = rng.random(keys.shape[1]) < 0.8
            mask[rng.integers(keys.shape[1])] = True
            trial = np.where(mask, donor, old[i])
            x = np.array([np.argmax(trial[offsets[j]:offsets[j + 1]])
                          for j in range(end.space.dimension)])
            score = end.evaluate(x)
            if score >= old_scores[i]:
                keys[i], scores[i] = trial, score


def ant_colony_optimization(end, rng, initial):
    pop, scores = initialize(end, initial)
    trails = [np.ones(c) for c in end.space.cardinalities]
    while True:
        for t in trails:
            t *= 0.8  # evaporation
        order = [i for i in np.argsort(-scores, kind="stable") if np.isfinite(scores[i])]
        order = order[:max(1, len(scores) // 3)]
        for rank, i in enumerate(order):
            for j, v in enumerate(pop[i]):
                trails[j][v] += 1 / (rank + 1)
        for t in trails:
            np.clip(t, 0.05, 20, out=t)
        if not end.remaining:
            return
        ants, values = [], []
        for _ in range(min(len(initial), end.remaining)):
            # Constant heuristic: constructing an ant performs no hidden queries.
            x = np.array([rng.choice(len(t), p=0.9 * t / t.sum() + 0.1 / len(t))
                          for t in trails])
            ants.append(x)
            values.append(end.evaluate(x))
        pop, scores = np.array(ants), np.array(values)


def artificial_bee_colony(end, rng, initial):
    pop, scores = initialize(end, initial)
    n = len(pop)
    failures = np.zeros(n, dtype=int)
    abandonment_limit = 10

    def attempt(i):
        x = pop[i].copy()
        mutable = [j for j, c in enumerate(end.space.cardinalities) if c > 1]
        if not mutable:
            end.evaluate(x)
            failures[i] += 1
            return
        j = int(rng.choice(mutable))
        k = int(rng.integers(n - 1))
        k += k >= i
        if pop[k, j] != x[j]:
            x[j] = pop[k, j]
        else:
            v = int(rng.integers(end.space.cardinalities[j] - 1))
            x[j] = v + (v >= x[j])
        score = end.evaluate(x)
        if score > scores[i]:
            pop[i], scores[i], failures[i] = x, score, 0
        else:
            failures[i] += 1

    while end.remaining:
        for i in range(n):  # employed bees
            if end.remaining:
                attempt(i)
        ranks = np.empty(n, dtype=float)
        ranks[np.argsort(scores, kind="stable")] = np.arange(1, n + 1)
        for _ in range(min(n, end.remaining)):  # rank-based onlookers
            attempt(int(rng.choice(n, p=ranks / ranks.sum())))
        for i in np.flatnonzero(failures >= abandonment_limit):
            if end.remaining:  # scouts: restarts are fully charged
                pop[i] = end.space.sample(rng)[0]
                scores[i] = end.evaluate(pop[i])
                failures[i] = 0


def simulated_annealing(end, rng, initial):
    pop, scores = initialize(end, initial)
    i = int(np.argmax(scores))
    x, score = pop[i].copy(), float(scores[i])
    total = max(1, end.remaining)
    while end.remaining:
        fraction = (total - end.remaining) / max(1, total - 1)
        temperature = 0.15 * (0.001 / 0.15) ** fraction
        y = end.space.neighbor(x, rng)
        value = end.evaluate(y)
        if value >= score or rng.random() < math.exp((value - score) / temperature):
            x, score = y, value


def thermal_double_ended_dpso(tracker, rng, initial):
    """Portable FairRec-inspired thermal DPSO with cached personal scores.

    Two interacting swarms, alpha=.09, c1=c2=2, velocity [-2,2], integer
    truncation and clipping. This cleans up state/score bookkeeping and aliases
    and therefore must not be called an exact native FairRec implementation.
    """
    x = np.array(initial, dtype=float)
    personal = x.copy()
    values = np.array([[tracker.evaluate(g) for g in swarm] for swarm in x])
    n, d = x.shape[1:]
    signs = (1, -1)
    while tracker.calls < tracker.budget:
        center = personal.mean(axis=1)
        flat = personal.reshape(-1, d)
        best = [flat[int(np.argmax(values))].copy(),
                flat[int(np.argmin(values))].copy()]
        for i in range(n):
            phi = rng.normal()
            for e, sign in enumerate(signs):
                if tracker.calls >= tracker.budget:
                    return
                velocity = (0.09 * np.abs(center[e] - x[e, i]) * phi
                            + 2 * rng.random() * (personal[e, i] - x[e, i])
                            + 2 * rng.random() * (best[e] - x[e, i]))
                x[e, i] = np.clip(np.trunc(x[e, i] + np.clip(velocity, -2, 2)),
                                  0, np.array(tracker.space.cardinalities) - 1)
                score = tracker.evaluate(x[e, i])
                if sign * score > sign * values[e, i]:
                    personal[e, i], values[e, i] = x[e, i].copy(), score
                if sign * score > sign * tracker.cache[tracker.space.validate(best[e])]:
                    best[e] = x[e, i].copy()


def load_native(path):
    """Load only the local FairRec class; avoid its model/import/CLI effects."""
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8-sig"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FairRec"]
    if len(nodes) != 1:
        raise ValueError("Expected exactly one FairRec class in native source")
    namespace = {"np": np, "random": random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["FairRec"], hashlib.sha256(raw).hexdigest()


def native_dpso(tracker, rng, population, native_class):
    # Exact native initialization/update logic; only the fitness boundary changes.
    # Every repeated native fitness request is charged, as for every challenger.
    native_class.fitness = lambda self, x: tracker.evaluate(x)
    params = dict(dimension=tracker.space.dimension, iter=tracker.budget,
                  size=population, bound_low=[0] * tracker.space.dimension,
                  bound_up=[c - 1 for c in tracker.space.cardinalities],
                  v_low=-2, v_high=2,
                  value_prob_list=[[list(range(c)), [1 / c] * c]
                                   for c in tracker.space.cardinalities])
    a, b = random.getstate(), np.random.get_state()
    seed = int(rng.integers(2**31 - 1))
    try:
        random.seed(seed)
        np.random.seed(seed)
        try:
            native_class(params, {}, {}).double_ended_dpso()
        except BudgetExhausted:
            pass
    finally:
        random.setstate(a)
        np.random.set_state(b)


ENGINES: dict[str, Callable] = {
    "GA": genetic_algorithm, "DE": differential_evolution,
    "ACO": ant_colony_optimization, "ABC": artificial_bee_colony,
    "SA": simulated_annealing,
}


def compare(objective, budget=4000, population=20, runs=3, seed=42,
            threshold=0.65, patience=10, native_source=None):
    if population < 4 or budget < 2 * population or runs < 1 or patience < 1:
        raise ValueError("Need population>=4, budget>=2*population, runs>=1, patience>=1")
    if not 0 <= threshold <= 1:
        raise ValueError("High-disparity threshold must lie in [0,1]")
    native_class, revision = load_native(native_source) if native_source else (None, None)
    baseline = "DPSO (native)" if native_class else "DPSO (thermal reference)"
    rows, traces = [], []
    for run in range(runs):
        initial_rng = np.random.default_rng(np.random.SeedSequence([seed, run, 0]))
        initial = objective.space.sample(initial_rng, 2 * population).reshape(
            2, population, objective.space.dimension)
        # Randomize execution order to reduce systematic wall-clock order effects.
        order = [baseline, *ENGINES]
        initial_rng.shuffle(order)
        for name in order:
            rng = np.random.default_rng(np.random.SeedSequence([seed, run, 1]))
            tracker = Tracker(objective, budget, threshold, population)
            start = perf_counter()
            if name == baseline:
                if native_class:
                    native_dpso(tracker, rng, population, native_class)
                else:
                    thermal_double_ended_dpso(tracker, rng, initial)
            else:
                # Equal two-end allocation preserves the double-ended study
                # design: half maximize, half minimize, same raw disparity.
                for e, sign in enumerate((1, -1)):
                    end = End(tracker, budget // 2 if e == 0 else budget, sign)
                    ENGINES[name](end, rng, initial[e])
            elapsed = perf_counter() - start
            assert tracker.calls == budget, (name, tracker.calls)
            rows.append(dict(
                algorithm=name, run=run, seed=seed,
                maximum_disparity=tracker.best,
                iterations_to_convergence=math.ceil(tracker.best_query / population),
                best_found_query=tracker.best_query,
                stagnation_confirmed=(budget - tracker.best_query >= patience * population),
                subgroup_diversity=len(tracker.high), execution_seconds=elapsed,
                queries=tracker.calls, unique_queries=len(tracker.cache),
                observed_range=tracker.best - tracker.worst,
                best_subgroup=json.dumps(tracker.best_group),
                objective_sha256=objective.fingerprint,
                native_source_sha256=revision or ""))
            traces.extend(dict(algorithm=name, run=run, **entry) for entry in tracker.trace)
    return rows, traces


def print_matrix(rows):
    def summary(values, digits=4):
        return f"{np.mean(values):.{digits}f} +/- {np.std(values, ddof=1) if len(values)>1 else 0:.{digits}f}"
    headers = ["Algorithm", "Maximum disparity", "Iterations to convergence*",
               "Subgroup diversity", "Execution time (s)"]
    table = []
    for name in dict.fromkeys([rows[0]["algorithm"], *[r["algorithm"] for r in rows]]):
        group = [r for r in rows if r["algorithm"] == name]
        table.append([name, *[summary([r[k] for r in group], digits) for k, digits in
            [("maximum_disparity", 4), ("iterations_to_convergence", 1),
             ("subgroup_diversity", 1), ("execution_seconds", 4)]]])
    widths = [max(len(row[j]) for row in [headers, *table]) for j in range(len(headers))]
    print(" | ".join(v.ljust(w) for v, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in table:
        print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))
    print("Values: mean +/- sample SD across independent runs.")
    print("* First attainment of final best, in common population-sized query blocks;")
    print("  a retrospective proxy, not a proof of convergence or native generation count.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--population", type=int, default=20, help="Per-end population / query-block size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--landscape-seed", type=int, default=2026)
    parser.add_argument("--cardinalities", nargs="+", type=int, default=[2, 4, 3] * 8)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--patience", type=int, default=10, help="Blocks without improvement for stagnation flag")
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/search_comparison"))
    args = parser.parse_args()
    objective = MockDisparity.create(Space(tuple(args.cardinalities)), args.landscape_seed)
    rows, traces = compare(objective, args.budget, args.population, args.runs,
                           args.seed, args.threshold, args.patience, args.native_source)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row) + "\n")
    config = dict(vars(args), objective_sha256=objective.fingerprint,
                  numpy_version=np.__version__,
                  initialization="Shared initial cells, except optional native initializer",
                  budget_policy="Every request including duplicates; equal split across ends",
                  diversity_definition="Distinct evaluated category tuples with score >= threshold",
                  timing="Includes initialization, search, scoring and in-memory logging; excludes output I/O")
    (args.out / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    print(f"SYNTHETIC benchmark: {objective.space.dimension} features, {args.runs} runs,")
    print(f"{args.budget} queries/method/run; high disparity >= {args.threshold:g}.")
    print_matrix(rows)
    print(f"Raw results, query traces and configuration: {args.out.resolve()}")


if __name__ == "__main__":
    main()
