"""Runnable synthetic experiments; no generated row is an empirical paper result."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import FixedOracle, SubgroupSpace, EvaluationSession
from .features import FilterConfig, HybridFeatureSelector
from .metrics import audit_groups, coverage_metrics
from .native import ExactCellSpace, NativeCellOracle, NativeFairRecSearch, NativeFairASSearch, DualEndedSearch
from .search import (RandomSearch, GeneticSearch, AntColonySearch, DiscretePSOReference,
                     SimulatedAnnealing, TabuSearch, HybridTestingFramework, GreedyLocalSearch)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest(out, args):
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).parent.glob("*.py")}
    dump_json(out / "manifest.json", {"synthetic": True, "arguments": vars(args),
              "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
              "scipy": scipy.__version__, "sklearn": sklearn.__version__, "source_hashes": sources,
              "baseline_policy": "Native loops use native initialization; generic demo shares candidates",
              "budget_policy": "Every scoring proposal charged, repeats included; unique calls also logged"})


def run_search_demo(out, seeds, budget):
    space = SubgroupSpace((3, 3, 3, 3, 3), max_order=3)
    def data(seed):
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 3, size=(1800, 5))
        utility = 0.75 + rng.normal(0, 0.04, len(X))
        utility -= 0.35 * ((X[:, 0] == 1) & (X[:, 1] == 2))
        utility -= 0.20 * ((X[:, 2] == 0) & (X[:, 3] == 1))
        return X, np.clip(utility, 0, 1)
    oracle = FixedOracle(*data(100), space, min_support=25)
    confirm = FixedOracle(*data(101), space, min_support=25)
    universe = list(space.enumerate())
    engines = [RandomSearch(), DiscretePSOReference(), GeneticSearch(), AntColonySearch(),
               SimulatedAnnealing(), TabuSearch(), HybridTestingFramework()]
    rows = []
    for seed in range(seeds):
        initial_rng = np.random.default_rng(seed + 500)
        initial = np.array([space.sample(initial_rng) for _ in range(20)])
        for engine in engines:
            session = EvaluationSession(oracle, budget, metadata={"engine": engine.name, "seed": seed,
                                                                   "synthetic": True})
            engine.run(session, np.random.default_rng(seed), initial.copy())
            if session.remaining:
                raise AssertionError("Engine did not consume its common proposal budget")
            session.write_jsonl(out / "traces" / f"{engine.name}_{seed}.jsonl")
            candidates = list(session.cache)
            audited = audit_groups(confirm, candidates)
            dump_json(out / f"confirmation_{engine.name}_{seed}.json", audited)
            hits = [r for r in audited if r["valid"] and r["disadvantaged"]]
            coverage = coverage_metrics(oracle, candidates, universe)
            rows.append({"synthetic": True, "engine": engine.name, "seed": seed,
                         "budget": budget, "proposals": len(session.logs), "unique_calls": len(session.cache),
                         "best_discovery_gap": session.best,
                         "confirmed_disadvantaged": len(hits),
                         "seconds": session.logs[-1]["elapsed_seconds"], **coverage})
    write_csv(out / "search_summary.csv", rows)


def run_feature_demo(out, seeds):
    rows = []
    for p, topk, limit in [(10, 3, 3), (20, 4, 4), (200, 8, 5)]:
        for seed in range(seeds):
            X, y = make_classification(n_samples=240, n_features=p, n_informative=min(5, p - 2),
                                       n_redundant=2, random_state=seed)
            # Constant-column failure case. It must be rejected by entropy.
            X[:, -1] = 1
            train, test = train_test_split(np.arange(len(X)), stratify=y, test_size=0.25, random_state=seed)
            estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
            baseline = sklearn.base.clone(estimator).fit(X[train], y[train])
            baseline_score = balanced_accuracy_score(y[test], baseline.predict(X[test]))
            for direction in ("forward", "backward"):
                started = time.perf_counter()
                selector = HybridFeatureSelector(estimator, FilterConfig(top_k=topk), direction=direction,
                                                 max_features=limit, cv=3, seed=seed)
                selector.fit(X[train], y[train])
                model = sklearn.base.clone(estimator).fit(selector.transform(X[train]), y[train])
                score = balanced_accuracy_score(y[test], model.predict(selector.transform(X[test])))
                rows.append({"synthetic": True, "dimensions": p, "seed": seed, "wrapper": direction,
                             "filtered": len(selector.filter_.selected_), "selected": len(selector.selected_),
                             "selected_indices": json.dumps(selector.selected_.tolist()),
                             "inner_selection_score": selector.inner_score_,
                             "heldout_balanced_accuracy": score, "all_feature_accuracy": baseline_score,
                             "subset_evaluations": selector.wrapper_evaluations_,
                             "seconds": time.perf_counter() - started})
    write_csv(out / "feature_summary.csv", rows)


def run_native_demo(out, repo, seeds, budget):
    """Actual local native search loops on a known synthetic exact-cell landscape.

    Native scoring/model inference is replaced with the SAME frozen fixture for
    ALL engines. This validates integration, not real DRS fairness effectiveness.
    """
    repo = Path(repo)
    space = ExactCellSpace((2, 6, 8))
    def score(g):
        a, b, c = g
        value = 0.7 + 0.03 * a + 0.01 * b + 0.005 * c
        if b in (1, 4) and c in (2, 6):
            value -= 0.35
        return value, 50
    oracle = NativeCellOracle(space, score, "synthetic_native_fixture_v1")
    engines = [NativeFairRecSearch(repo / "FairRec/fairrec.py", population=10),
               DualEndedSearch(GeneticSearch(population=10)),
               DualEndedSearch(AntColonySearch(population=10)),
               NativeFairASSearch(repo / "FairAS/run_s.py", ("gender", "age", "country")),
               NativeFairASSearch(repo / "FairAS/run_s.py", ("gender", "age", "country"), GreedyLocalSearch()),
               NativeFairASSearch(repo / "FairAS/run_s.py", ("gender", "age", "country"), SimulatedAnnealing()),
               NativeFairASSearch(repo / "FairAS/run_s.py", ("gender", "age", "country"), TabuSearch()),
               DualEndedSearch(HybridTestingFramework(global_search=GeneticSearch(population=10)))]
    all_values = [score(g)[0] for g in space.enumerate()]
    truth = max(all_values) - min(all_values)
    rows = []
    for seed in range(seeds):
        for engine in engines:
            session = EvaluationSession(oracle, budget, metadata={"synthetic": True, "engine": engine.name,
                                                                  "seed": seed}, objective_field="utility")
            engine.run(session, np.random.default_rng(seed))
            session.write_jsonl(out / "native_traces" / f"{engine.name}_{seed}.jsonl")
            scores = [v.utility for v in session.cache.values() if v.valid]
            found_range = max(scores) - min(scores) if scores else None
            rows.append({"synthetic": True, "engine": engine.name, "seed": seed, "budget": budget,
                         "proposals": len(session.logs), "unique_calls": len(session.cache),
                         "discovered_utility_range": found_range, "exhaustive_range": truth,
                         "range_regret": truth - found_range if scores else None,
                         "termination": session.metadata.get("termination", "budget_exhausted"),
                         "seconds": session.logs[-1]["elapsed_seconds"] if session.logs else 0})
    write_csv(out / "native_search_summary.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/demo")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--mode", choices=["all", "search", "features", "native"], default="all")
    parser.add_argument("--repo", default="C:/Users/DELL/Documents/repo")
    args = parser.parse_args()
    if args.seeds < 1 or args.budget < 2:
        parser.error("Positive seed count and budget >= 2 required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest(out, args)
    if args.mode in ("all", "search"):
        run_search_demo(out, args.seeds, args.budget)
    if args.mode in ("all", "features"):
        run_feature_demo(out, args.seeds)
    if args.mode in ("all", "native"):
        run_native_demo(out, args.repo, args.seeds, args.budget)
    print(f"Saved synthetic demonstration to {out.resolve()}")


if __name__ == "__main__":
    main()
