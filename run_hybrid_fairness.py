"""Runnable mock hybrid: development-only feature selection, separate audit users.

python run_hybrid_fairness.py --algorithm both
python run_hybrid_fairness.py --native-source C:/Users/DELL/Documents/repo/FairAS/run_s.py

Mock per-user utilities emulate frozen recommendation utility, not a trained DRS.
Without --native-source the driver is explicitly FairAS-inspired, not native.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from fairness_lab.categorical_selection import CategoricalUnionBackwardSelector
from fairness_lab.hybrid import HybridFairnessTester
from fairness_lab.subgroup_objectives import CategoricalUtilityGap


def mock_users(n_users, n_features, seed):
    if n_users < 20 or n_features < 2:
        raise ValueError("Demo needs at least 20 users and two features")
    rng = np.random.default_rng(seed)
    categories = rng.integers(0, 3, size=(n_users, n_features))
    # Marginal and interacting harms; independent noise varies between users.
    utility = np.clip(0.85 - 0.4 * (categories[:, 0] == 0)
                      - 0.25 * ((categories[:, 0] == 1) & (categories[:, 1] == 1))
                      + rng.normal(0, 0.02, n_users), 0, 1)
    frame = pd.DataFrame(categories.astype(str), columns=[f"attribute_{j}" for j in range(n_features)])
    return frame, utility


def mock_slate_evaluator(audit, seed):
    """Synthetic slate metadata; never presented as NeuMF recommendations."""
    from fairness_lab.expanded_bias import ExpandedBiasEvaluator
    rng = np.random.default_rng(seed)
    features = np.eye(6)[np.arange(60) % 6]
    training_counts = np.arange(60, 0, -1) * 10
    preferences = rng.dirichlet(np.ones(6), size=len(audit))
    slates = []
    for i, value in enumerate(audit.attribute_0):
        weights = (features[:, 0] * 5 + 0.1) if value == "0" else preferences[i][np.arange(60) % 6] + 0.05
        slates.append(rng.choice(60, 8, replace=False, p=weights / weights.sum()))
    return ExpandedBiasEvaluator(np.asarray(slates), features, training_counts,
                    preferences, source_id=f"mock-slates-{seed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["GA", "ACO", "both"], default="both")
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--development-users", type=int, default=600)
    parser.add_argument("--audit-users", type=int, default=1500)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=15)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--local-budget", type=int, default=24)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--one-step-num", type=int, default=3)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--expanded-bias", action="store_true", help="Track synthetic calibration, exposure, popularity and ILD alongside utility")
    parser.add_argument("--out", type=Path, default=Path("outputs/hybrid_mock"))
    args = parser.parse_args()
    development, development_utility = mock_users(args.development_users, args.features, args.seed)
    audit, audit_utility = mock_users(args.audit_users, args.features, args.seed + 1)
    labels = (development_utility < 0.7).astype(int)
    objective = CategoricalUtilityGap.from_frame(audit, audit_utility, args.min_support,
                    source_id=f"mock-frozen-utility-seed-{args.seed+1}")
    if args.expanded_bias:
        from fairness_lab.expanded_bias import ExpandedBiasObjective
        objective = ExpandedBiasObjective(objective, mock_slate_evaluator(audit, args.seed + 2))
    estimator = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                              DecisionTreeClassifier(max_depth=4, random_state=args.seed))
    selector = CategoricalUnionBackwardSelector(top_n=args.top_n, max_features=args.max_features,
                                                estimator=estimator, cv=3, seed=args.seed)
    algorithms = ["GA", "ACO"] if args.algorithm == "both" else [args.algorithm]
    for algorithm in algorithms:
        tester = HybridFairnessTester(selector, algorithm, args.budget, args.population,
            args.local_budget, args.radius, args.one_step_num, args.seed, args.threshold,
            args.native_source)
        directory = args.out / algorithm
        discovered = tester.run(development, labels, objective=objective, out=directory)
        (directory / "demo_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
        summary = tester.search_adapter_.metadata_
        print(f"\n{summary['algorithm']} ({summary['backend']})")
        print(f"Selected: {', '.join(tester.selected_features_)}")
        print(f"Eliminated ({len(tester.eliminated_features_)}): {', '.join(tester.eliminated_features_)}")
        print(f"Requests={summary['requests']}/{args.budget}; local calls={summary['local_invocations']}; "
              f"stop={summary['stopping']}")
        if discovered:
            print(pd.DataFrame(discovered)[["rank", "demographic_values", "support", "disparity", "high_disparity"]]
                  .head(10).to_string(index=False))
        else:
            print("No supported subgroup found within this request budget.")
        print(f"Logs: {directory.resolve()}")


if __name__ == "__main__":
    main()
