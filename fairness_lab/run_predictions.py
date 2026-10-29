"""Run native and replacement searches against a frozen real prediction CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .core import EvaluationSession
from .experiments import dump_json, write_csv
from .native import NativeFairRecSearch, NativeFairASSearch, DualEndedSearch
from .predictions import from_csv
from .search import GeneticSearch, AntColonySearch, RandomSearch, SimulatedAnnealing, TabuSearch, HybridTestingFramework, GreedyLocalSearch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--repo", default="C:/Users/DELL/Documents/repo")
    p.add_argument("--baseline", choices=["FairRec", "FairAS"], required=True)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--cardinalities", nargs="+", type=int, required=True)
    p.add_argument("--metric", choices=["mrr", "auc", "ndcg", "urd", "urp"], default="mrr")
    p.add_argument("--support", type=float, required=True, help="Absolute user threshold, not a fraction")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--request-cap", type=int, default=10000)
    p.add_argument("--unique-budget", type=int, default=200)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--initialization", choices=["uniform", "native"], default="native")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if len(args.features) != len(args.cardinalities) or args.seeds < 1 or min(args.cardinalities) < 1:
        p.error("Match features to positive cardinalities and use positive seed count")
    repo, out = Path(args.repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_source = repo / args.baseline / "rec_metrics.py"
    oracle = from_csv(args.predictions, metrics_source, args.features, args.cardinalities,
                      metric=args.metric, threshold_users=args.support,
                      threshold_rule="strict_greater" if args.baseline == "FairRec" else "greater_equal")
    frame = oracle.scorer.frame
    X = frame[args.features].to_numpy()
    if not np.all(np.isfinite(X)) or not np.array_equal(X, X.astype(int)):
        p.error("Sensitive columns must have fixed integer codes")
    if any(np.any((X[:, j] < 0) | (X[:, j] >= c)) for j, c in enumerate(args.cardinalities)):
        p.error("Sensitive category codes exceed supplied cardinalities")
    # Static group attributes are required for a user-group interpretation.
    if (frame.groupby("user_id")[args.features].nunique() > 1).to_numpy().any():
        p.error("Sensitive subgroup attributes vary within user; define the unit before testing")
    native_dpso = NativeFairRecSearch(repo / "FairRec/fairrec.py", args.population)
    probabilities = None
    if args.initialization == "native":
        # Native FairRec samples frequencies of interaction rows, as in set_dpso_params.
        native_dpso.value_prob_list = []
        for j, c in enumerate(args.cardinalities):
            frequency = np.bincount(X[:, j].astype(int), minlength=c).astype(float)
            native_dpso.value_prob_list.append([list(range(c)), (frequency / frequency.sum()).tolist()])
        if args.baseline == "FairAS":
            # Use original initialize_filter on the actual prediction table.
            from .native import source_definitions
            import contextlib
            import io
            ns = {"np": np, "feat_dict": dict(zip(args.features, args.cardinalities)), "test_data": frame}
            source_definitions(repo / "FairAS/utilis.py", {"create_single_rule", "create_sparse_rule_query"}, ns)
            source_definitions(repo / "FairAS/run_s.py", {"initialize_filter"}, ns)
            with contextlib.redirect_stdout(io.StringIO()):
                probabilities = ns["initialize_filter"]({"sensitive_features": args.features,
                                                         "threshold_num": args.support}, {})
            if any(sum(v) <= 0 for v in probabilities.values()):
                p.error("Native prefilter produced an empty sampling distribution")
    engines = [DualEndedSearch(RandomSearch()), DualEndedSearch(GeneticSearch(args.population)),
               DualEndedSearch(AntColonySearch(args.population)),
               DualEndedSearch(HybridTestingFramework(global_search=GeneticSearch(args.population)))]
    if args.baseline == "FairRec":
        engines.insert(0, native_dpso)
    else:
        engines = [NativeFairASSearch(repo / "FairAS/run_s.py", args.features, local,
                                      probabilities=probabilities)
                   for local in (None, GreedyLocalSearch(), SimulatedAnnealing(), TabuSearch())] + engines
    rows = []
    for seed in range(args.seeds):
        for engine in engines:
            session = EvaluationSession(oracle, args.request_cap, unique_budget=args.unique_budget,
                                         objective_field="utility", metadata={"engine": engine.name, "seed": seed,
                                         "synthetic": False, "metric": args.metric, "initialization": args.initialization})
            engine.run(session, np.random.default_rng(seed))
            session.write_jsonl(out / "traces" / f"{engine.name}_{seed}.jsonl")
            values = [r.utility for r in session.cache.values() if r.valid]
            rows.append({"engine": engine.name, "seed": seed, "requests": len(session.logs),
                         "unique_queries": len(session.cache), "valid_unique_groups": len(values),
                         "utility_range": max(values) - min(values) if values else None,
                         "seconds": session.logs[-1]["elapsed_seconds"] if session.logs else 0,
                         "termination": session.metadata.get("termination", "unique_budget" if
                             len(session.cache) >= args.unique_budget else "request_cap")})
    write_csv(out / "summary.csv", rows)
    dump_json(out / "manifest.json", {"arguments": vars(args), "synthetic": False,
              "oracle_hash": oracle.fingerprint, "native_metric_source_sha256": oracle.scorer.source_hash,
              "prediction_csv_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
              "metric_compatibility": "Native methods preserved; roc_auc_score symbol supplied explicitly",
              "initialization": "Native policies preserved only for native engines; alternatives use their declared categorical samplers",
              "timing": "Search over precomputed predictions; excludes inference and prefilter setup"})
    print(f"Saved prediction-table experiment to {out.resolve()}")


if __name__ == "__main__":
    main()
