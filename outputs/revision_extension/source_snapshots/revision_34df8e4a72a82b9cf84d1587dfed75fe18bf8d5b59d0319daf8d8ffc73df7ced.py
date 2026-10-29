"""Post hoc revision experiments; preserve all earlier experiment folders.

Add ABC/SA to recorded MovieLens and synthetic protocols, and audit three
supplied frozen DeepFM checkpoints. External splits are disjoint USERS from
the supplied candidate table, not new training replications or full catalogs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from compare_fairrec_search import load_native
from .categorical_selection import CategoricalUnionBackwardSelector
from .conference import enumerate_cells, landscape, save_search, sha
from .empirical import atomic_csv, atomic_json, feature_frame
from .recommender import load_movielens_1m
from .subgroup_objectives import CategoricalUtilityGap
from .utility_support import bounded_radius

METHODS = ("DPSO (native)", "GA", "ACO", "DE", "ABC", "SA", "Random")


class IndexedUtilityGap(CategoricalUtilityGap):
    """Identical cell scores with O(1) lookups after a single user aggregation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cells, inverse, counts = np.unique(self.demographics, axis=0, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=self.utilities)
        self._cells = {tuple(g): (int(n), float(s / n)) for g, n, s in zip(cells, counts, sums)}

    def project(self, names):
        base = super().project(names)
        return IndexedUtilityGap(base.demographics, base.utilities, base.feature_names,
                                 base.categories, base.min_support, source_id=base.source_id)

    def details(self, vector, one_hot=False):
        group = self.decode(vector, one_hot)
        n, q = self._cells.get(group, (0, None))
        valid = n >= self.min_support
        return dict(subgroup=group, demographic_values={name: values[v] for name, values, v
                    in zip(self.feature_names, self.categories, group)}, support=n, valid=valid,
                    subgroup_utility=q if valid else None, global_utility=self.global_utility,
                    disparity=self.global_utility-q if valid else None)


def selection(frame, labels, cap, seed=314159):
    if np.unique(labels, return_counts=True)[1].min() < 3 or len(np.unique(labels)) != 2:
        raise ValueError("Selection target needs two classes with three examples each")
    selector = CategoricalUnionBackwardSelector(top_n=10, max_features=cap, min_features=1,
        estimator=make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                DecisionTreeClassifier(max_depth=5, random_state=seed)),
        cv=3, tolerance=.001, seed=seed).fit(frame, labels)
    return dict(selected_features=list(selector.get_feature_names_out()),
                filter_union=[frame.columns[j] for j in selector.filter_union_],
                wrapper_history=selector.history_)


def ablation_record(base, names, threshold, **context):
    projected = base.project(names)
    full, reduced = enumerate_cells(base), enumerate_cells(projected)
    high = [r for r in full if r["valid"] and r["disparity"] >= threshold]
    columns = [base.feature_names.index(n) for n in names]
    kept = [r for r in high if projected.details(tuple(r["subgroup"][j] for j in columns))["disparity"] >= threshold]
    valid = [r for r in reduced if r["valid"]]
    return dict(**context, features=list(names), total_cells=len(reduced), valid_cells=len(valid),
        maximum_gap=max((r["disparity"] for r in valid), default=None),
        high_full_cells=len(high), retained_full_cells=len(kept),
        parent_threshold_retention=len(kept)/len(high) if high else None)


def add_existing(args, native):
    config = json.loads((args.baseline / "config.json").read_text())
    data = load_movielens_1m(Path(config["data"]), config["positive_rating"], tuple(config["features"]))
    assert data.source_sha256 == config["data_sha256"]
    split = json.loads((args.baseline / "user_split.json").read_text())
    lookup = {int(u): i for i, u in enumerate(data.user_ids)}
    audit = np.array([lookup[u] for u in split["audit_user_ids"]])
    frame = feature_frame(data, SimpleNamespace(feature_table=None)).iloc[audit]
    records = []
    for model_seed in config["seeds"]:
        source_root = args.previous / f"model_{model_seed}"
        metadata = json.loads((source_root / "utilities.json").read_text())
        assert metadata["cache_sha256"] == sha(source_root / "utilities.npz")
        with np.load(source_root / "utilities.npz") as cache:
            assert np.array_equal(cache["user_ids"], data.user_ids[audit])
            q = cache["validation"].copy()
        source = f"{metadata['model_sha256']}:{data.source_sha256}:audit-users:validation"
        base = IndexedUtilityGap.from_frame(frame, q, 20, source_id=source)
        report = json.loads((args.baseline / f"seed_{model_seed}/feature_selection.json").read_text())
        for grammar, names in (("selected", report["selected_features"]), ("unfiltered", list(frame))):
            objective = base.project(names)
            exact = enumerate_cells(objective)
            # Verify the new indexed implementation against every original score.
            previous = pd.read_csv(source_root / f"{grammar}_exact.csv")
            for old, new in zip(previous.to_dict("records"), exact):
                assert old["support"] == new["support"] and old["valid"] == new["valid"]
                if new["valid"]:
                    assert np.isclose(old["disparity"], new["disparity"], atol=1e-14, rtol=0)
            for seed in range(1001, 1011):
                for method in ("ABC", "SA"):
                    records.append(save_search(args.out / "movielens" / f"model_{model_seed}" / grammar /
                        f"search_{seed}" / method, objective, method, args, seed, native,
                        dict(model_seed=model_seed, grammar=grammar), exact))
            print(f"MovieLens {model_seed} {grammar}: ABC/SA complete", flush=True)
    atomic_csv(args.out / "movielens_added_results.csv", records)
    synthetic_records = []
    for family, instance, encoding in itertools.product(("additive", "nk_trap", "interaction"), range(2026, 2029), range(3)):
        rng = np.random.default_rng(np.random.SeedSequence([instance, encoding, 517]))
        mapping = [np.arange(c) if encoding == 0 else rng.permutation(c) for c in (2, 4, 3)*8]
        objective = landscape(family, instance, mapping)
        settings = SimpleNamespace(budget=args.budget, population=args.population, threshold=.65)
        for seed in range(1001, 1006):
            for method in ("ABC", "SA"):
                synthetic_records.append(save_search(args.out / "synthetic" / family / f"landscape_{instance}" /
                    f"encoding_{encoding}" / f"search_{seed}" / method, objective, method, settings, seed, native,
                    dict(family=family, landscape_seed=instance, encoding=encoding)))
        print(f"Synthetic {family} {instance} encoding {encoding}: ABC/SA complete", flush=True)
    atomic_csv(args.out / "synthetic_added_results.csv", synthetic_records)


def external(args, native):
    searches, ablations, confirmations, summaries = [], [], [], []
    for dataset in ("last-fm", "black-fri", "amazon"):
        export = args.exports / dataset
        metadata = json.loads((export / "metadata.json").read_text())
        assert sha(export / "user_utilities.csv") == metadata["utility_csv_sha256"]
        users = pd.read_csv(export / "user_utilities.csv")
        assert not users.user_id.duplicated().any()
        frame = users[metadata["audit_attributes"]].astype(str)
        order = np.random.default_rng(314159).permutation(len(users))
        n_dev = int(.30 * len(users))
        n_audit = int(.35 * len(users))
        dev, audit, confirm = order[:n_dev], order[n_dev:n_dev+n_audit], order[n_dev+n_audit:]
        root = args.out / dataset
        atomic_json(root / "user_split.json", dict(seed=314159, development=users.user_id.iloc[dev].tolist(),
            discovery=users.user_id.iloc[audit].tolist(), confirmation=users.user_id.iloc[confirm].tolist()))
        # Selection is fit only on development users. No test-based tuning.
        q_dev = users.ndcg10.to_numpy()[dev]
        labels = (q_dev.mean() - q_dev >= .02).astype(int)
        arms = {}
        for cap in (2, len(frame.columns)):
            key = f"utility_cap{cap}"
            arms[key] = selection(frame.iloc[dev], labels, cap)
        arms["unfiltered"] = dict(selected_features=list(frame.columns))
        atomic_json(root / "selection.json", dict(target="development mean NDCG@10 minus user NDCG@10 >= .02",
            target_positive=int(labels.sum()), development_users=len(dev), arms=arms))
        # Declare vocabulary from supplied metadata without using utilities;
        # categories absent from discovery remain explicit unsupported cells.
        categories = tuple(tuple(pd.unique(frame[n])) for n in frame)
        encoded = np.column_stack([pd.Categorical(frame[n], categories=v).codes
                                   for n, v in zip(frame, categories)])
        full = IndexedUtilityGap(encoded[audit], users.ndcg10.iloc[audit], tuple(frame), categories, 20,
                                 source_id=metadata["utility_csv_sha256"] + ":discovery")
        confirmed = IndexedUtilityGap(encoded[confirm], users.ndcg10.iloc[confirm], full.feature_names,
            full.categories, 1, source_id=metadata["utility_csv_sha256"] + ":confirmation")
        # Both selection caps + no selection, crossed with support sensitivity.
        for support in (5, 20, 50, 100):
            base = IndexedUtilityGap(full.demographics, full.utilities, full.feature_names,
                                     full.categories, support, source_id=full.source_id)
            for arm, report in arms.items():
                ablations.append(ablation_record(base, report["selected_features"], .02,
                                                 dataset=dataset, arm=arm, min_support=support))
        for grammar, names in (("selected", arms["utility_cap2"]["selected_features"]), ("unfiltered", list(frame))):
            objective = full.project(names)
            exact = enumerate_cells(objective)
            atomic_csv(root / f"{grammar}_exact.csv", exact)
            valid = [r for r in exact if r["valid"]]
            if not valid:
                raise ValueError(f"No supported primary cells for {dataset} {grammar}; report rather than retune")
            rule = max(valid, key=lambda r: r["disparity"])
            result = confirmed.project(names).details(rule["subgroup"])
            n = result["support"]
            radius = bounded_radius(n, len(confirm)) if n else None
            confirmations.append(dict(dataset=dataset, grammar=grammar, rule=rule["demographic_values"],
                discovery_gap=rule["disparity"], confirmation_gap=result["disparity"], support=n,
                lower=None if n == 0 else max(-1., result["disparity"]-radius),
                upper=None if n == 0 else min(1., result["disparity"]+radius),
                scope="one discovery-fixed rule; separate users; frozen supplied checkpoint; pointwise bounded interval"))
            for seed in range(1001, 1011):
                methods = list(METHODS)
                np.random.default_rng(np.random.SeedSequence([seed, 999])).shuffle(methods)
                for method in methods:
                    searches.append(save_search(root / grammar / f"search_{seed}" / method.replace(" ", "_"),
                        objective, method, args, seed, native, dict(dataset=dataset, grammar=grammar), exact))
            print(f"External {dataset} {grammar}: 70 searches complete", flush=True)
        summaries.append(dict(dataset=dataset, total_users=len(users), development_users=len(dev),
            discovery_users=len(audit), confirmation_users=len(confirm), rows=metadata["rows"],
            attributes=list(frame), cardinalities=list(full.space.cardinalities),
            candidate_min=metadata["candidate_count_min"], candidate_max=metadata["candidate_count_max"],
            global_discovery_ndcg=float(full.global_utility), checkpoint_replicates=1))
        atomic_csv(args.out / "external_search_results.csv", searches)
        atomic_csv(args.out / "external_ablation.csv", ablations)
        atomic_csv(args.out / "external_confirmation.csv", confirmations)
        atomic_json(args.out / "external_datasets.json", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["existing", "external"], required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/revision_extension"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/final_benchmark"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/conference_extension"))
    parser.add_argument("--exports", type=Path, default=Path("outputs/external_utilities_checked"))
    args = parser.parse_args()
    args.budget, args.population, args.threshold = 4000, 20, .02
    torch.set_num_threads(4)
    config = json.loads((args.baseline / "config.json").read_text())
    native, revision = load_native(Path(config["dpso_source"]))
    assert revision == config["dpso_source_sha256"]
    paths = [Path(__file__), Path("compare_fairrec_search.py"), Path("fairness_lab/conference.py"),
             Path("fairness_lab/empirical.py"), Path("fairness_lab/subgroup_objectives.py"),
             Path("fairness_lab/categorical_selection.py"), Path("fairness_lab/features.py")]
    protocol = dict(scope=args.scope, methods=METHODS, budget=4000, population_per_end=20,
        threshold=.02, synthetic_threshold=.65, source_sha256={str(p):sha(p) for p in paths},
        native_sha256=revision, split_seed=314159, min_support=20, support_grid=[5,20,50,100],
        external_search_seeds=list(range(1001,1011)), external_development_fraction=.30,
        external_discovery_fraction=.35, selection_cv=3, selection_depth=5, selection_tolerance=.001,
        note="post hoc; no test-based tuning; supplied DeepFM checkpoints are not new training replications")
    path = args.out / f"{args.scope}_protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol)), "Use a new output directory after changes"
    atomic_json(path, protocol)
    (add_existing if args.scope == "existing" else external)(args, native)
    atomic_json(args.out / f"{args.scope}_complete.json", dict(status="complete", protocol_sha256=sha(path)))


if __name__ == "__main__":
    main()
