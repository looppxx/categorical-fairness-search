"""Train/load MovieLens NeuMF, search real utility gaps, confirm on test.

python run_live_fairness.py --data data/ml-1m --download --epochs 5
python run_live_fairness.py --data data/ml-1m --checkpoint outputs/live_fairness/neumf.pt

PyTorch is optional for the rest of the repository; see requirements-recommender.txt.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compare_fairrec_search import ENGINES
from fairness_lab.recommender import (NeuMF, RecommenderDisparityObjective,
    download_movielens, load_movielens_1m, train_neumf)


class LiveSearchSession:
    """Adapter for the existing GA/ACO/DE functions; invalid groups get -inf.

    Both disparity signs remain available: a negative *valid* score is an
    advantaged subgroup, distinct from invalid/undersupported cells. Every
    objective request counts, including invalid cells and duplicates.
    """
    def __init__(self, objective, budget, threshold=0.02):
        self.objective, self.space = objective, objective.space
        self.budget, self.threshold = budget, threshold
        self.logs, self.cache, self.high = [], {}, set()
        self.best, self.best_group, self.best_query = -np.inf, None, None

    @property
    def remaining(self):
        return self.budget - len(self.logs)

    def evaluate(self, vector):
        if self.remaining <= 0:
            raise RuntimeError("Search exceeded its common request budget")
        key = self.space.validate(vector)
        cached = key in self.cache
        # In live mode perform fresh forward passes even for repeated cells.
        if not cached or not self.objective.cache_utilities:
            self.cache[key] = self.objective.details(key)
        detail = self.cache[key]
        score = detail["disparity"]
        query = len(self.logs) + 1
        if detail["valid"]:
            if score > self.best:
                self.best, self.best_group, self.best_query = score, key, query
            if score >= self.threshold:
                self.high.add(key)
        self.logs.append(dict(query=query, **detail, cached=cached,
                             best=None if self.best_group is None else self.best,
                             diversity=len(self.high)))
        return float(score) if detail["valid"] else -np.inf


def benchmark(objective, algorithms=("GA", "ACO", "DE"), budget=1000,
              population=20, runs=3, seed=42, threshold=0.02):
    if population < 4 or budget < population or runs < 1 or threshold < 0:
        raise ValueError("Need population>=4, budget>=population, runs>=1, threshold>=0")
    if not algorithms or not set(algorithms) <= set(ENGINES):
        raise ValueError("Unknown algorithm")
    rows, traces = [], []
    for run in range(runs):
        rng_init = np.random.default_rng(np.random.SeedSequence([seed, run, 0]))
        initial = objective.space.sample(rng_init, population)
        order = list(algorithms)
        rng_init.shuffle(order)
        for name in order:
            session = LiveSearchSession(objective, budget, threshold)
            rng = np.random.default_rng(np.random.SeedSequence([seed, run, 1]))
            start = perf_counter()
            ENGINES[name](session, rng, initial)
            seconds = perf_counter() - start
            assert not session.remaining
            rows.append(dict(algorithm=name, run=run, search_seed=seed,
                maximum_disparity=None if session.best_group is None else session.best,
                iterations_to_convergence=None if session.best_query is None else
                    int(np.ceil(session.best_query / population)),
                best_found_query=session.best_query,
                subgroup_diversity=len(session.high), execution_seconds=seconds,
                requests=len(session.logs), unique_groups=len(session.cache),
                invalid_requests=sum(not r["valid"] for r in session.logs),
                best_subgroup=session.best_group, objective_sha256=objective.fingerprint,
                audit_sha256=getattr(objective, "audit_fingerprint", "")))
            if hasattr(objective, "audit_fingerprint"):
                from fairness_lab.expanded_bias import flatten_bias_report
                best = {} if session.best_query is None else session.logs[session.best_query - 1]
                rows[-1].update(flatten_bias_report(best.get("expanded_bias")))
            traces.extend(dict(algorithm=name, run=run, **row) for row in session.logs)
    return rows, traces


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (tuple, dict, list)) else v
                         for k, v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Folder containing users.dat and ratings.dat")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Load an existing matching NeuMF checkpoint")
    parser.add_argument("--out", type=Path, default=Path("outputs/live_fairness"))
    parser.add_argument("--features", nargs="+", default=["gender", "age", "occupation"])
    parser.add_argument("--positive-rating", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithms", nargs="+", choices=list(ENGINES), default=["GA", "ACO", "DE"])
    parser.add_argument("--metric", choices=["ndcg", "recall"], default="ndcg")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--live-inference", action="store_true", help="Recompute subgroup model predictions on every request")
    parser.add_argument("--expanded-bias", action="store_true", help="Audit frozen recommendation slates using MovieLens genres and training histories")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.download and not (args.data / "ratings.dat").exists():
        download_movielens(args.data)
    data = load_movielens_1m(args.data, args.positive_rating, tuple(args.features))
    args.out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        if checkpoint["data_sha256"] != data.source_sha256:
            raise ValueError("Checkpoint data/encoding/split settings differ")
        model = NeuMF(**checkpoint["model_config"]).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        training_history = checkpoint.get("training_history", [])
    else:
        model, training_history = train_neumf(data, args.epochs, args.negatives,
             args.batch_size, seed=args.seed, device=args.device, embedding_dim=args.embedding_dim)
        torch.save(dict(state_dict=model.state_dict(), model_config=model.config,
                        data_sha256=data.source_sha256, training_history=training_history,
                        training_settings=dict(epochs=args.epochs, negatives=args.negatives,
                                               batch_size=args.batch_size, seed=args.seed)), args.out / "neumf.pt")
    train_seconds = perf_counter() - start
    start = perf_counter()
    objective = RecommenderDisparityObjective(model, data, args.metric, args.k,
                   args.min_support, cache_utilities=not args.live_inference)
    item_features = None
    if args.expanded_bias:
        from fairness_lab.expanded_bias import ExpandedBiasEvaluator, ExpandedBiasObjective, movielens_genre_features
        item_features, genre_names = movielens_genre_features(data, args.data)
        objective = ExpandedBiasObjective(objective, ExpandedBiasEvaluator.from_neumf(objective, item_features))
    setup_seconds = perf_counter() - start
    print(f"Frozen {args.metric.upper()}@{args.k}: global={objective.global_utility:.6f}; "
          f"eligible users={len(data.user_ids)}; catalog={len(data.item_ids)}", flush=True)
    rows, traces = benchmark(objective, args.algorithms, args.budget, args.population,
                            args.runs, args.seed, args.threshold)
    # Instantiate the confirmation oracle only AFTER validation search finishes.
    confirmation = RecommenderDisparityObjective(model, data, args.metric, args.k,
                       args.min_support, split="test")
    if args.expanded_bias:
        confirmation = ExpandedBiasObjective(confirmation, ExpandedBiasEvaluator.from_neumf(confirmation, item_features))
    confirmations = []
    for row in rows:
        if row["best_subgroup"] is not None:
            confirmations.append(dict(algorithm=row["algorithm"], run=row["run"],
                                       **confirmation.details(row["best_subgroup"])))
    write_csv(args.out / "results.csv", rows)
    write_csv(args.out / "test_confirmation.csv", confirmations)
    with (args.out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    import pandas as pd
    utilities = pd.DataFrame(data.demographics, columns=data.feature_names)
    utilities.insert(0, "user_id", data.user_ids)
    utilities["validation_utility"] = objective.utilities
    utilities["test_utility"] = confirmation.utilities
    utilities.to_csv(args.out / "per_user_utilities.csv", index=False)
    config = dict(vars(args), model_sha256=objective.model_sha256,
                  model_config=model.config,
                  data_sha256=data.source_sha256, objective_sha256=objective.fingerprint,
                  categories=dict(zip(data.feature_names, data.categories)),
                  torch_version=str(torch.__version__), numpy_version=np.__version__,
                  training_or_loading_seconds=train_seconds, objective_setup_seconds=setup_seconds,
                  training_history=training_history,
                  audit_sha256=getattr(objective, "audit_fingerprint", ""),
                  objective="eligible-user global mean minus group mean; signed disparity",
                  invalid_policy="None from oracle; -inf for maximization search; every request charged",
                  search_policy="single-ended disparity maximization with identical initial cells and request budgets",
                  convergence="first final-best query / population, rounded up; retrospective proxy",
                  ranking="full catalog, one held-out positive, train masked; validation also masked on test",
                  timing="search includes objective calls and in-memory logging; training and global inference reported separately")
    (args.out / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    print(frame.groupby("algorithm")[["maximum_disparity", "iterations_to_convergence",
            "subgroup_diversity", "execution_seconds", "invalid_requests"]].agg(["mean", "std"]).to_string())
    print(f"Saved validation search results and separate test confirmation to {args.out.resolve()}")


if __name__ == "__main__":
    main()
