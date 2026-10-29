"""Separate post hoc controls; never rewrite the support-20 evidence.

Utility labels are development-only per-user deficits, dichotomized for the
existing classification-based mRMR/Chi-square wrapper. They are not continuous
subgroup gaps or causal discrimination labels. All support cases are reported;
none is chosen using test persistence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from compare_fairrec_search import load_native
from .conference import ALGORITHMS, enumerate_cells, save_search, sha
from .empirical import atomic_csv, atomic_json, feature_frame, select_features
from .recommender import NeuMF, load_movielens_1m, ranking_utilities
from .subgroup_objectives import CategoricalUtilityGap


def utility_labels(development_utility, deficit_threshold=.02):
    q = np.asarray(development_utility, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or deficit_threshold < 0:
        raise ValueError("Finite development utilities and nonnegative threshold required")
    deficits = q.mean() - q
    return (deficits >= deficit_threshold).astype(int), deficits


def retained_cells(full_objective, projected, threshold):
    full = enumerate_cells(full_objective)
    high = [r for r in full if r["valid"] and r["disparity"] >= threshold]
    indices = [full_objective.feature_names.index(n) for n in projected.feature_names]
    kept = [r for r in high if projected.details(tuple(r["subgroup"][j] for j in indices))["disparity"] >= threshold]
    return dict(high_full_cells=len(high), high_cells_with_high_parent=len(kept),
                parent_threshold_retention=len(kept)/len(high) if high else None)


def bounded_radius(n, total):
    if not 0 < n <= total:
        raise ValueError("Invalid subgroup support")
    if n == total:
        return 0.
    return (1-n/total)*(math.sqrt(math.log(80)/(2*n))+math.sqrt(math.log(80)/(2*(total-n))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/final_benchmark"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/conference_extension"))
    parser.add_argument("--out", type=Path, default=Path("outputs/utility_support_extension"))
    parser.add_argument("--support-grid", nargs="+", type=int, default=[20, 50, 100, 200])
    parser.add_argument("--primary-support", type=int, default=50)
    parser.add_argument("--search-seeds", nargs="+", type=int, default=list(range(1001,1011)))
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=.02)
    args = parser.parse_args()
    if args.primary_support not in args.support_grid or min(args.support_grid) < 1 or len(set(args.support_grid)) != len(args.support_grid):
        parser.error("Distinct positive supports must include the primary support")
    if args.population < 4 or args.budget < 2*args.population or args.threshold < 0:
        parser.error("Invalid population, budget, or threshold")
    if len(set(args.search_seeds)) != len(args.search_seeds) or any(not 0 <= s < 2**32 for s in args.search_seeds):
        parser.error("Distinct uint32 search seeds required")
    torch.set_num_threads(4)
    config = json.loads((args.baseline / "config.json").read_text())
    split = json.loads((args.baseline / "user_split.json").read_text())
    native, revision = load_native(Path(config["dpso_source"]))
    assert revision == config["dpso_source_sha256"]
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = [args.baseline / "config.json", args.baseline / "user_split.json",
              Path(__file__), Path("run_utility_support_extension.py")]
    inputs += [Path("fairness_lab") / n for n in ("conference.py", "empirical.py", "categorical_selection.py",
               "features.py", "subgroup_objectives.py", "recommender.py")]
    inputs += [Path("compare_fairrec_search.py")]
    for seed in config["seeds"]:
        inputs += [args.baseline / f"seed_{seed}/neumf.pt", args.previous / f"model_{seed}/utilities.npz",
                   args.previous / f"model_{seed}/utilities.json"]
    protocol = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        input_sha256={str(p):sha(p) for p in inputs}, native_sha256=revision,
        labels="development mean NDCG@10 minus per-development-user NDCG@10 >= 0.02",
        feature_caps=[2,3], primary_support_choice="50 fixed before new test confirmation; report all grid values",
        design="post hoc; classification selector unchanged; no new training; test utility never selects labels, features, support, or rules")
    if (args.out / "protocol.json").exists():
        assert json.loads((args.out / "protocol.json").read_text()) == protocol, "Choose new output for a different protocol"
    atomic_json(args.out / "protocol.json", protocol)
    data = load_movielens_1m(Path(config["data"]), config["positive_rating"], tuple(config["features"]))
    assert data.source_sha256 == config["data_sha256"]
    indices = {int(u):i for i,u in enumerate(data.user_ids)}
    development = np.array([indices[u] for u in split["development_user_ids"]])
    audit = np.array([indices[u] for u in split["audit_user_ids"]])
    assert not set(development) & set(audit)
    frame = feature_frame(data, SimpleNamespace(feature_table=None))
    selection_rows, cases, searches = [], [], []
    for seed in config["seeds"]:
        root = args.out / f"model_{seed}"
        root.mkdir(parents=True, exist_ok=True)
        dev_path = root / "development_utilities.npz"
        dev_metadata = dev_path.with_suffix(".json")
        expected_dev = dict(checkpoint_sha256=sha(args.baseline / f"seed_{seed}/neumf.pt"),
            data_sha256=data.source_sha256, ranking_source_sha256=sha("fairness_lab/recommender.py"))
        if dev_path.exists():
            stored = json.loads(dev_metadata.read_text())
            assert all(stored[k] == v for k,v in expected_dev.items()) and stored["cache_sha256"] == sha(dev_path)
            with np.load(dev_path) as cache:
                assert np.array_equal(cache["user_ids"], data.user_ids[development])
                q_dev = cache["ndcg"]
        else:
            checkpoint = torch.load(args.baseline / f"seed_{seed}/neumf.pt", map_location="cpu", weights_only=True)
            assert checkpoint["data_sha256"] == data.source_sha256
            model = NeuMF(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval().requires_grad_(False)
            q_dev = ranking_utilities(model, data, "validation", 10, development)["ndcg"]
            np.savez_compressed(dev_path, ndcg=q_dev, user_ids=data.user_ids[development])
            atomic_json(dev_metadata, dict(expected_dev, cache_sha256=sha(dev_path)))
        labels, deficits = utility_labels(q_dev, .02)
        label_path = root / "utility_labels.csv"
        atomic_csv(label_path, [dict(user_id=int(u), ndcg=float(q), deficit=float(d), target=int(y))
            for u,q,d,y in zip(data.user_ids[development],q_dev,deficits,labels)])
        grammar = {}
        for target in ("activity", "utility"):
            for cap in (2,3):
                arm = f"{target}_cap{cap}"
                selection_args = SimpleNamespace(selection_labels=label_path if target=="utility" else None,
                    selection_labels_provenance="development-only per-user NDCG deficit >= .02; outcome-aligned proxy",
                    mandatory_features=[], selection_cv=3, top_n=10, max_features=cap)
                report = select_features(data, frame, development, selection_args, seed)
                report["warning"] = "Predictive proxy labels; retention of subgroup gaps must be evaluated separately."
                atomic_json(root / f"{arm}_selection.json", report)
                names = tuple(report["selected_features"])
                grammar[arm] = names
                selection_rows.append(dict(model_seed=seed, target=target, cap=cap, arm=arm,
                    selected_features=list(names), cells=math.prod(len(frame.iloc[audit][n].unique()) for n in names),
                    utility_positive_labels=int(labels.sum()) if target=="utility" else None,
                    development_users=len(development), development_mean_utility=float(q_dev.mean())))
        grammar["unfiltered"] = tuple(frame.columns)
        previous_metadata = json.loads((args.previous / f"model_{seed}/utilities.json").read_text())
        assert previous_metadata["cache_sha256"] == sha(args.previous / f"model_{seed}/utilities.npz")
        assert previous_metadata["checkpoint_sha256"] == expected_dev["checkpoint_sha256"]
        assert previous_metadata["data_sha256"] == data.source_sha256
        with np.load(args.previous / f"model_{seed}/utilities.npz") as cache:
            assert np.array_equal(cache["user_ids"], data.user_ids[audit])
            q_val, q_test = cache["validation"], cache["test"]
        pending = []
        for support in args.support_grid:
            base = CategoricalUtilityGap.from_frame(frame.iloc[audit], q_val, support,
                source_id=f"utility-support-model-{seed}-validation")
            for arm,names in grammar.items():
                objective = base.project(names)
                exact = enumerate_cells(objective)
                atomic_csv(root / f"{arm}_support_{support}_exact.csv", exact)
                valid = [r for r in exact if r["valid"]]
                best = max(valid, key=lambda r:r["disparity"]) if valid else None
                record = dict(model_seed=seed, arm=arm, features=list(names), min_support=support,
                    total_cells=len(exact), valid_cells=len(valid), global_utility=base.global_utility,
                    validation_gap=None if best is None else best["disparity"],
                    **retained_cells(base,objective,args.threshold))
                pending.append((record, names, best))
        # Search unique grammars only; aliases share exact same objective and runs.
        for names in dict.fromkeys(grammar.values()):
            aliases = [a for a,n in grammar.items() if n == names]
            objective = CategoricalUtilityGap.from_frame(frame.iloc[audit], q_val, args.primary_support,
                source_id=f"utility-support-model-{seed}-validation").project(names)
            exact = enumerate_cells(objective)
            if not any(r["valid"] for r in exact):
                continue
            for search_seed in args.search_seeds:
                order = list(ALGORITHMS)
                np.random.default_rng(np.random.SeedSequence([seed, search_seed, 999])).shuffle(order)
                for algorithm in order:
                    searches.append(save_search(root / "primary_search" / "_".join(names) / f"search_{search_seed}" / algorithm.replace(" ","_"),
                        objective, algorithm, args, search_seed, native,
                        dict(model_seed=seed, grammar=",".join(names), aliases=aliases, min_support=args.primary_support), exact))
            print(f"Model {seed}: support {args.primary_support}, grammar {names}, 50 searches complete", flush=True)
        # All validation-only rules and primary searches end before confirmation.
        for record,names,best in pending:
            if best is not None:
                test = CategoricalUtilityGap.from_frame(frame.iloc[audit], q_test, record["min_support"],
                    source_id=f"utility-support-model-{seed}-test").project(names).details(best["subgroup"])
                n = int(test["support"])
                radius = bounded_radius(n, len(audit))
                record.update(rule=best["demographic_values"], support=n, test_gap=test["disparity"],
                    sign_reversal=bool(best["disparity"] > 0 and test["disparity"] < 0),
                    absolute_val_test_difference=abs(best["disparity"]-test["disparity"]),
                    test_interval_lower=test["disparity"]-radius, test_interval_upper=test["disparity"]+radius)
            cases.append(record)
        atomic_csv(args.out / "selection_results.csv", selection_rows)
        atomic_csv(args.out / "support_results.csv", cases)
        atomic_csv(args.out / "primary_search_results.csv", searches)
        print(f"Model {seed}: all selection and support cases recorded", flush=True)
    atomic_json(args.out / "complete.json", dict(status="complete", model_seeds=config["seeds"],
        selection_arms=len(selection_rows), support_cases=len(cases), unique_primary_searches=len(searches),
        objective_requests=len(searches)*args.budget, primary_support=args.primary_support))


if __name__ == "__main__":
    main()
