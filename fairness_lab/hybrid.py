"""Feature union/backward selection -> adaptive FairAS with GA/ACO refinement."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.base import clone

from .categorical_selection import CategoricalUnionBackwardSelector
from .fairas_adapter import FairASSearchAdapter
from .subgroup_objectives import json_default


class HybridFairnessTester:
    """Fit selection on development data; search a separately supplied audit oracle.

    Supply exactly one of:
      objective: a live/table objective exposing project(selected_feature_names);
      objective_factory: callable(selected_names) returning an already reduced
        objective, e.g. a CallableSubgroupObjective for a custom live loop.

    Reduced subgroup predicates leave every eliminated attribute unconstrained.
    Selection labels must be development-only and discrete; final test utilities
    must never be used to construct them or to tune selection/search settings.
    """
    def __init__(self, selector=None, algorithm="GA", budget=500, population=12,
                 local_budget=36, radius=2, one_step_num=5, seed=42,
                 threshold=0.02, native_source=None, frozen_features=()):
        self.selector, self.algorithm, self.budget = selector, algorithm, budget
        self.population, self.local_budget, self.radius = population, local_budget, radius
        self.one_step_num, self.seed, self.threshold = one_step_num, seed, threshold
        self.native_source, self.frozen_features = native_source, tuple(frozen_features)

    def fit(self, X, y, *, objective=None, objective_factory=None, groups=None):
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any() or not all(isinstance(n, str) for n in X.columns):
            raise ValueError("Selection data needs a DataFrame with distinct string feature names")
        if (objective is None) == (objective_factory is None):
            raise ValueError("Supply exactly one projectable objective or objective_factory")
        start = perf_counter()
        self.selector_ = clone(self.selector) if self.selector is not None else CategoricalUnionBackwardSelector(seed=self.seed)
        self.selector_.fit(X, y, groups=groups)
        self.selected_features_ = tuple(self.selector_.get_feature_names_out().tolist())
        self.eliminated_features_ = tuple(name for name in X.columns if name not in self.selected_features_)
        if objective_factory is not None:
            projected = objective_factory(self.selected_features_)
        else:
            if not hasattr(objective, "project"):
                raise TypeError("Bare full-vector functions cannot project safely; supply objective_factory(selected_names)")
            projected = objective.project(self.selected_features_)
        if tuple(projected.feature_names) != self.selected_features_:
            raise ValueError("Projected objective must preserve selected feature names/order")
        self.objective_ = projected
        union = set(self.selector_.filter_union_.tolist())
        selected = set(self.selector_.selected_.tolist())
        self.feature_report_ = dict(input_features=list(X.columns),
            selected_features=list(self.selected_features_), eliminated_features=list(self.eliminated_features_),
            mrmr_order=[X.columns[j] for j in self.selector_.mrmr_order_],
            chi_square_order=[X.columns[j] for j in self.selector_.chi_order_],
            filter_union=[X.columns[j] for j in sorted(union)],
            removed_by_filter=[X.columns[j] for j in range(X.shape[1]) if j not in union],
            removed_after_union=[X.columns[j] for j in sorted(union - selected)],
            wrapper_history=self.selector_.history_,
            selection_seconds=perf_counter() - start,
            category_vocabulary=dict(zip(projected.feature_names, getattr(projected, "categories", []))),
            eliminated_attribute_policy="unconstrained; partial conjunctions over retained features",
            label_policy="caller-supplied discrete development-only labels")
        return self

    def search(self):
        if not hasattr(self, "objective_"):
            raise RuntimeError("Fit the feature pipeline before searching")
        self.search_adapter_ = FairASSearchAdapter(self.algorithm, self.population,
            self.local_budget, self.radius, self.one_step_num, self.native_source, self.frozen_features)
        self.ledger_ = self.search_adapter_.run(self.objective_, self.budget, self.seed, self.threshold)
        ordered = sorted(((key, detail) for key, detail in self.ledger_.cache.items() if detail["valid"]),
                         key=lambda pair: (-pair[1]["disparity"], pair[0]))
        self.subgroups_ = []
        for rank, (key, detail) in enumerate(ordered, start=1):
            predicates = detail["demographic_values"]
            full_rule = {name: predicates[name] if name in predicates else "*"
                         for name in self.feature_report_["input_features"]}
            self.subgroups_.append(dict(rank=rank, **detail, high_disparity=key in self.ledger_.high,
                                        full_rule=full_rule))
        return self.subgroups_

    def run(self, X, y, *, objective=None, objective_factory=None, groups=None, out=None):
        self.fit(X, y, objective=objective, objective_factory=objective_factory, groups=groups)
        subgroups = self.search()
        if out is not None:
            self.write_logs(out)
        return subgroups

    def write_logs(self, directory):
        if not hasattr(self, "subgroups_"):
            raise RuntimeError("Run subgroup search before exporting")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        def write_json(name, value):
            (directory / name).write_text(json.dumps(value, default=json_default, indent=2,
                                                    allow_nan=False), encoding="utf-8")
        write_json("feature_selection.json", self.feature_report_)
        write_json("subgroups.json", self.subgroups_)
        write_json("summary.json", dict(self.search_adapter_.metadata_, search_seed=self.seed,
            request_budget=self.budget, threshold=self.threshold, population=self.population,
            local_budget=self.local_budget, one_step_num=self.one_step_num,
            selected_features=self.selected_features_, eliminated_features=self.eliminated_features_))
        summary = self.search_adapter_.metadata_
        best_query = next((row["query"] for row in self.ledger_.logs
                           if summary["maximum_disparity"] is not None and
                           row["disparity"] == summary["maximum_disparity"]), None)
        result = dict(algorithm=summary["algorithm"], run=0, search_seed=self.seed,
            maximum_disparity=summary["maximum_disparity"],
            iterations_to_convergence=None if best_query is None else math.ceil(best_query / self.population),
            best_found_query=best_query, subgroup_diversity=summary["subgroup_diversity"],
            execution_seconds=summary["execution_seconds"], requests=summary["requests"],
            unique_groups=summary["unique_groups"], invalid_requests=summary["invalid_requests"],
            population=self.population, threshold=self.threshold, request_budget=self.budget,
            objective_sha256=self.objective_.fingerprint,
            audit_sha256=getattr(self.objective_, "audit_fingerprint", ""))
        if hasattr(self.objective_, "audit_fingerprint"):
            from .expanded_bias import flatten_bias_report
            best = {} if best_query is None else self.ledger_.logs[best_query - 1]
            result.update(flatten_bias_report(best.get("expanded_bias")))
        with (directory / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result))
            writer.writeheader()
            writer.writerow(result)
        fields = ["rank", "subgroup", "demographic_values", "support", "subgroup_utility",
                  "global_utility", "disparity", "high_disparity", "full_rule"]
        with (directory / "subgroups.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for record in self.subgroups_:
                writer.writerow({key: json.dumps(value, default=json_default) if isinstance(value, (tuple, dict, list))
                                 else value for key, value in record.items() if key in fields})
        with (directory / "trace.jsonl").open("w", encoding="utf-8") as f:
            for record in self.ledger_.logs:
                f.write(json.dumps(dict(algorithm=summary["algorithm"], run=0,
                    search_seed=self.seed, **record), default=json_default, allow_nan=False) + "\n")
        bias_rows = [dict(subgroup=record["subgroup"], support=record["support"],
                         metric=name, **values) for record in self.subgroups_
                     for name, values in record.get("expanded_bias", {}).items()]
        if bias_rows:
            with (directory / "expanded_bias.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(bias_rows[0]))
                writer.writeheader()
                writer.writerows(dict(row, subgroup=json.dumps(row["subgroup"], default=json_default)) for row in bias_rows)
            evaluator = self.objective_.evaluator
            write_json("expanded_bias_definitions.json", dict(definitions=evaluator.DEFINITIONS,
                global_means=evaluator.global_means, audit_sha256=evaluator.fingerprint))
