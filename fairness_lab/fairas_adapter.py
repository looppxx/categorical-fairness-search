"""Replace FairAS local_search with categorical GA/ACO under one query ledger.

Native mode executes source-loaded FairAS adaptive updates; portable mode is
an explicitly FairAS-inspired driver. The original numeric +/-1,2 neighborhood
is replaced by a bounded Hamming neighborhood (all categories are nominal).
"""
from __future__ import annotations

from collections import Counter
import contextlib
import copy
import io
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from .core import BudgetExhausted
from .native import ExactCellSpace, seeded_legacy_rng, source_definitions
from .search import GeneticSearch, AntColonySearch


class SearchStopped(Exception):
    pass


def default_cache_policy(objective):
    if hasattr(objective, "cache_utilities"):
        return bool(objective.cache_utilities)
    return getattr(objective, "score_users", None) is None


class SearchLedger:
    """One budget/archive for adaptive sampling AND every local-search call."""
    def __init__(self, objective, budget=500, threshold=0.02, cache_results=None):
        if budget < 1 or threshold < 0:
            raise ValueError("Positive request budget and nonnegative disparity threshold required")
        self.objective, self.budget, self.threshold = objective, budget, threshold
        self.cache_results = default_cache_policy(objective) if cache_results is None else cache_results
        self.cache, self.logs, self.high = {}, [], set()
        self.best, self.best_group = -np.inf, None
        self.started = perf_counter()

    @property
    def remaining(self):
        return self.budget - len(self.logs)

    def evaluate(self, vector, phase="adaptive_sample", **context):
        if not self.remaining:
            raise BudgetExhausted
        key = self.objective.space.validate(vector)
        cached = key in self.cache and self.cache_results
        if not cached:
            detail = dict(self.objective.details(key))
            score = detail["disparity"]
            if bool(detail["valid"]) != (score is not None) or (score is not None and not np.isfinite(score)):
                raise ValueError("Objective validity and finite/None disparity must agree")
            self.cache[key] = detail
        detail = self.cache[key]
        if detail["valid"]:
            if detail["disparity"] > self.best:
                self.best, self.best_group = float(detail["disparity"]), key
            if detail["disparity"] >= self.threshold:
                self.high.add(key)
        self.logs.append(dict(query=len(self.logs) + 1, phase=phase, candidate=key,
                    **detail, cached=cached, **context,
                    best_disparity=None if self.best_group is None else self.best,
                    elapsed_seconds=perf_counter() - self.started))
        return detail["disparity"]


class FairASLocalSearchAdapter:
    """Callable with the exact native FairAS local_search argument signature.

    objective: projectable/live objective or CallableSubgroupObjective wrapper.
    Scores retain their original signed disparity; direction only changes the
    optimizer's view. Native max searches disadvantage, min searches advantage.
    `key_format='native'` supports the upstream concatenated-string lists for
    direct legacy hook injection. The orchestrator uses collision-safe tuples.
    """
    def __init__(self, objective, algorithm="GA", population=12, local_budget=36,
                 radius=2, budget=500, seed=42, ledger=None, rng=None,
                 key_format="native", frozen_features=()):
        if algorithm not in {"GA", "ACO"} or population < 2 or local_budget < 1 or radius < 0:
            raise ValueError("Choose GA/ACO, population>=2, local_budget>=1, radius>=0")
        if key_format not in {"native", "tuple"}:
            raise ValueError("Unknown duplicate-key format")
        self.objective, self.algorithm = objective, algorithm
        self.population, self.local_budget, self.radius = population, local_budget, radius
        self.ledger = ledger if ledger is not None else SearchLedger(objective, budget)
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.key_format, self.frozen_features = key_format, tuple(frozen_features)
        if not set(self.frozen_features) <= set(objective.feature_names):
            raise ValueError("Frozen local attributes must exist in the search space")
        self.engine = GeneticSearch(population) if algorithm == "GA" else AntColonySearch(population)
        self.invocations = 0

    def token(self, candidate):
        key = tuple(int(v) for v in candidate)
        return key if self.key_format == "tuple" else "".join(map(str, key))

    def local_search(self, test_info, feat_dict, temp_sample, score_target, direction,
                     sampled_users_str, sampled_users, sampled_score):
        if direction not in {"min", "max"}:
            raise ValueError("FairAS direction must be min or max")
        names = tuple(test_info["sensitive_features"])
        if names != self.objective.feature_names or tuple(feat_dict[n] for n in names) != self.objective.space.cardinalities:
            raise ValueError("FairAS hook schema differs from objective schema")
        seed = np.array(self.objective.space.validate(temp_sample))
        self.invocations += 1
        invocation = self.invocations
        adapter, radius = self, self.radius
        frozen = [names.index(n) for n in self.frozen_features]

        class Neighborhood(ExactCellSpace):
            def repair(self, candidate, rng):
                g = super().repair(candidate, rng)
                g[frozen] = seed[frozen]
                changed = np.flatnonzero(g != seed)
                if len(changed) > radius:
                    reset = rng.choice(changed, len(changed) - radius, replace=False)
                    g[reset] = seed[reset]
                return g
            def sample(self, rng):
                return self.repair(np.array([rng.integers(c) for c in self.cardinalities]), rng)

        space = Neighborhood(self.objective.space.cardinalities)
        known = {tuple(g) for g in sampled_users}
        sign = 1 if direction == "max" else -1

        class LocalSession:
            oracle = type("LocalOracle", (), {"space": space})()
            calls = 0
            best_group = seed.copy()
            best_score = score_target
            @property
            def remaining(self):
                return min(adapter.local_budget - self.calls, adapter.ledger.remaining)
            def evaluate(self, vector):
                if not self.remaining:
                    raise BudgetExhausted
                group = adapter.objective.space.validate(vector)
                distance = int(np.count_nonzero(np.asarray(group) != seed))
                if distance > radius or any(group[j] != seed[j] for j in frozen):
                    raise ValueError("Local engine left its declared neighborhood")
                self.calls += 1
                score = adapter.ledger.evaluate(group, "local_search", local_seed=seed.tolist(),
                    hamming_distance=distance, direction=direction, local_invocation=invocation)
                token = adapter.token(group)
                if token not in sampled_users_str:
                    sampled_users_str.append(token)
                if score is None:
                    return -np.inf
                if group not in known:
                    known.add(group)
                    sampled_users.append(list(group))
                    sampled_score.append(score)
                if self.best_score is None or sign * score > sign * self.best_score:
                    self.best_group, self.best_score = np.asarray(group), score
                return sign * score

        session = LocalSession()
        initial = [seed] + [space.sample(self.rng) for _ in range(self.population - 1)]
        self.engine.run(session, self.rng, initial)
        return session.best_group.tolist(), session.best_score

    __call__ = local_search


class FairASSearchAdapter:
    """Adaptive outer discovery with GA/ACO replacing the local hook.

    native_source=None: portable FairAS-inspired batching and median adaptation.
    native_source=run_s.py: original sample_group/update_prob/one_step functions.
    Native mode additionally fixes ambiguous concatenated duplicate keys and
    bounds duplicate rejection; these declared changes prevent collisions/hangs.
    """
    def __init__(self, algorithm="GA", population=12, local_budget=36, radius=2,
                 one_step_num=5, native_source=None, frozen_features=()):
        if one_step_num < 1:
            raise ValueError("one_step_num must be positive")
        self.algorithm, self.population, self.local_budget = algorithm, population, local_budget
        self.radius, self.one_step_num = radius, one_step_num
        self.native_source = None if native_source is None else Path(native_source)
        self.frozen_features = tuple(frozen_features)

    def run(self, objective, budget=500, seed=42, threshold=0.02, cache_results=None, ledger=None):
        if cache_results is None:
            cache_results = default_cache_policy(objective)
        if ledger is None:
            ledger = SearchLedger(objective, budget, threshold, cache_results)
        elif ledger.objective is not objective or ledger.budget != budget or ledger.logs:
            raise ValueError("Injected ledger must be fresh and match objective/budget")
        rng = np.random.default_rng(seed)
        local = FairASLocalSearchAdapter(objective, self.algorithm, self.population,
            self.local_budget, self.radius, ledger=ledger, rng=rng, key_format="tuple",
            frozen_features=self.frozen_features)
        names = objective.feature_names
        info, dimensions = {"sensitive_features": list(names)}, dict(zip(names, objective.space.cardinalities))
        users, scores, keys = [], [], []
        p_min = {name: np.full(c, 1 / c).tolist() for name, c in dimensions.items()}
        p_max = copy.deepcopy(p_min)
        total_groups = math.prod(objective.space.cardinalities)
        metadata = dict(backend="native" if self.native_source else "portable_fairas_inspired",
                        algorithm="FairAS-" + self.algorithm, objective_sha256=objective.fingerprint,
                        audit_sha256=getattr(objective, "audit_fingerprint", ""),
                        local_radius=self.radius, frozen_local_features=list(self.frozen_features),
                        stopping="request_budget", duplicate_keys="tuples")

        def draw(probabilities, native_random=None):
            if len(keys) >= total_groups:
                metadata["stopping"] = "categorical_space_exhausted"
                raise SearchStopped
            seen = set(keys)
            for _ in range(max(1000, budget * 10)):
                if native_random:
                    group = tuple(native_random.choices(range(dimensions[name]), weights=probabilities[name], k=1)[0]
                                  for name in names)
                else:
                    group = tuple(int(rng.choice(dimensions[name], p=np.asarray(probabilities[name]) /
                                                  sum(probabilities[name]))) for name in names)
                if group not in seen:
                    keys.append(group)
                    return list(group)
            metadata["stopping"] = "duplicate_rejection_limit"
            raise SearchStopped

        try:
            if self.native_source is not None:
                namespace = dict(np=np, random=random, copy=copy, Counter=Counter,
                    time=__import__("time"), algorithm_params={"one_step_num": self.one_step_num},
                    feat_dict=dimensions, query_count=0, query_time=0.0, eval_time=0.0, unfairness_list=[])
                definitions = {"count_sensitive_one_direction", "sample_group", "update_prob_one_direction",
                               "update_prob", "one_step"}
                metadata["source_sha256"] = source_definitions(self.native_source, definitions, namespace)
                metadata["native_adjustments"] = ["injected disparity scorer", "GA/ACO local hook",
                                                   "tuple duplicate keys", "bounded duplicate rejection",
                                                   "request cap", "uniform initial category weights"]
                namespace["run_score"] = lambda test_info, group: ledger.evaluate(group)
                namespace["local_search"] = local.local_search
                namespace["sample_one_direction"] = lambda test_info, weights, seen: draw(weights, random)
                with seeded_legacy_rng(int(rng.integers(2**31 - 1))), contextlib.redirect_stdout(io.StringIO()):
                    iteration = 0
                    while ledger.remaining:
                        iteration += 1
                        p_min, p_max = namespace["one_step"](info, p_min, p_max,
                            self.one_step_num * iteration * 2, users, scores, keys)
            else:
                while ledger.remaining:
                    for direction, probabilities in (("min", p_min), ("max", p_max)):
                        for _ in range(self.one_step_num):
                            if not ledger.remaining:
                                break
                            group = draw(probabilities)
                            score = ledger.evaluate(group)
                            if score is None:
                                continue
                            users.append(group)
                            scores.append(score)
                            extreme = min(scores) if direction == "min" else max(scores)
                            if score == extreme and ledger.remaining:
                                local(info, dimensions, group, score, direction, keys, users, scores)
                    if scores:
                        median = float(np.median(scores))
                        # Same per-coordinate mass redistribution used by the
                        # inspected FairAS update_prob_one_direction function.
                        for direction, weights in (("min", p_min), ("max", p_max)):
                            keep = [g for g, s in zip(users, scores) if (s < median if direction == "min" else s > median)]
                            for j, name in enumerate(names):
                                counts = Counter(g[j] for g in keep)
                                mass = sum(weights[name][v] for v in counts)
                                for v, count in counts.items():
                                    weights[name][v] = count / len(keep) * mass
        except BudgetExhausted:
            pass
        except SearchStopped:
            pass
        metadata.update(requests=len(ledger.logs), unique_groups=len(ledger.cache),
                        invalid_requests=sum(not row["valid"] for row in ledger.logs),
                        local_invocations=local.invocations,
                        maximum_disparity=None if ledger.best_group is None else ledger.best,
                        best_subgroup=ledger.best_group,
                        subgroup_diversity=len(ledger.high), cache_results=cache_results,
                        execution_seconds=perf_counter() - ledger.started)
        self.ledger_, self.metadata_ = ledger, metadata
        return ledger
