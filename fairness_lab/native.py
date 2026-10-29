"""Integrate actual local FairRec/FairAS search definitions without editing repos.

AST loading skips TensorFlow imports, CLI parsing, and model/data side effects.
Search methods themselves come from the supplied local source. This permits
testing search against frozen native group scores or an injected native scorer.
"""
from __future__ import annotations
import ast
import contextlib
import copy
from collections import Counter
import hashlib
import io
import itertools
import random
from pathlib import Path
import time

import numpy as np

from .core import SubgroupSpace, GroupResult, BudgetExhausted


class ExactCellSpace(SubgroupSpace):
    """Native full conjunctions, no wildcard extension in controlled comparisons."""
    def __init__(self, cardinalities):
        super().__init__(tuple(cardinalities), len(cardinalities))

    @property
    def lower_bound(self):
        return 0

    def validate(self, candidate):
        g = super().validate(candidate)
        if any(v < 0 for v in g):
            raise ValueError("Native cell space does not allow wildcards")
        return g

    def sample(self, rng):
        return np.array([rng.integers(c) for c in self.cardinalities])

    def repair(self, candidate, rng):
        return np.clip(np.rint(candidate).astype(int), 0, np.asarray(self.cardinalities) - 1)

    def mutate(self, g, rng):
        h = np.array(g, copy=True)
        dims = [j for j, c in enumerate(self.cardinalities) if c > 1]
        if dims:
            j = rng.choice(dims)
            choices = [v for v in range(self.cardinalities[j]) if v != h[j]]
            h[j] = rng.choice(choices)
        return h

    def enumerate(self):
        return itertools.product(*(range(c) for c in self.cardinalities))


class NativeCellOracle:
    """Native scalar group metric, with support/invalid policy provided by caller.

    scorer(g) -> (score or None, unique_user_support). Preserve original metric
    aggregation exactly (pooled AUC is not mean user AUC). scorer_id must identify
    model/data/metric/support policy. Results use utility as the search objective.
    """
    def __init__(self, space, scorer, scorer_id):
        self.space, self.scorer = space, scorer
        self.fingerprint = hashlib.sha256(scorer_id.encode()).hexdigest()

    def __call__(self, g):
        key = self.space.validate(g)
        value, support = self.scorer(key)
        if value is None or not np.isfinite(value):
            return GroupResult(int(support), 0, None, None, None, False)
        return GroupResult(int(support), 0, float(value), None, None, True)


def source_definitions(path, names, namespace):
    path = Path(path)
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8-sig"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Local source API differs from inspected definitions")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return hashlib.sha256(raw).hexdigest()


@contextlib.contextmanager
def seeded_legacy_rng(seed):
    a, b = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(a)
        np.random.set_state(b)


class NativeFairRecSearch:
    name = "fairrec_native_dpso"

    def __init__(self, source, population=20, velocity=(-2, 2)):
        self.source, self.population, self.velocity = Path(source), population, velocity

    def run(self, session, rng, initial=None):
        if not isinstance(session.oracle.space, ExactCellSpace) or session.objective_field != "utility":
            raise ValueError("Native DPSO needs exact cells and original utility score")
        if initial is not None:
            raise ValueError("Native initializer is preserved; use policy-matched initialization in a separate ablation")
        ns = {"np": np, "random": random}
        revision = source_definitions(self.source, {"FairRec"}, ns)
        session.metadata["source_sha256"] = revision
        cls = ns["FairRec"]
        # Replace fitness boundary only; keep initializer and native updates.
        def fitness(instance, position):
            session.evaluate(position)
            result = session.cache[session.oracle.space.validate(position)]
            return result.utility if result.valid else -100
        cls.fitness = fitness
        space = session.oracle.space
        params = {"dimension": space.dimension, "iter": session.budget,
                  "size": self.population, "bound_low": [0] * space.dimension,
                  "bound_up": [c - 1 for c in space.cardinalities],
                  "v_low": self.velocity[0], "v_high": self.velocity[1],
                  "value_prob_list": [[list(range(c)), [1 / c] * c] for c in space.cardinalities]}
        # For data-faithful initialization, set probabilities from empirical
        # original sensitive-feature frequencies before running (see setter).
        if hasattr(self, "value_prob_list"):
            params["value_prob_list"] = self.value_prob_list
        with seeded_legacy_rng(int(rng.integers(2**31 - 1))):
            try:
                obj = cls(params, {}, {})
                obj.double_ended_dpso()
            except BudgetExhausted:
                pass


class DualEndedSearch:
    """Modern engine allocated half the total proposals for each utility end."""
    def __init__(self, engine):
        self.engine = engine
        self.name = engine.name + "_dual"

    def run(self, session, rng, initial=None):
        total = session.budget
        unique_total = session.unique_budget
        session.budget = max(1, total // 2)
        if unique_total is not None:
            session.unique_budget = max(1, unique_total // 2)
        try:
            self.engine.run(session, rng, initial)
        finally:
            session.budget = total
            session.unique_budget = unique_total
        session.sign, session.best, session.best_group = -1, -np.inf, None
        try:
            if session.remaining:
                self.engine.run(session, rng, initial)
        finally:
            session.sign = 1


class DrawLimitReached(Exception):
    pass


class LimitedRandom:
    """Bound native duplicate rejection loops; report a stall instead of hanging."""
    def __init__(self, limit):
        self.limit, self.calls = limit, 0

    def choices(self, *args, **kwargs):
        self.calls += 1
        if self.calls > self.limit:
            raise DrawLimitReached()
        return random.choices(*args, **kwargs)


def replacement_local_search(engine, session, rng, local_budget=20):
    """Same signature as FairAS/run_s.py local_search; keeps shared archive.

    Engine candidates stay within Hamming distance <=2 of the triggering seed,
    making this a local optimizer. Original neighborhood changes code +/-1/2;
    changing the neighborhood is an explicit experimental factor.
    """
    def local(test_info, feat_dict, temp_sample, score_target, direction,
              sampled_users_str, sampled_users, sampled_score):
        seed = np.asarray(temp_sample, int)
        parent_space = session.oracle.space
        class Neighborhood(ExactCellSpace):
            def sample(self, local_rng):
                h = seed.copy()
                for _ in range(min(2, self.dimension)):
                    h = super().mutate(h, local_rng)
                return h

            def repair(self, candidate, local_rng):
                h = super().repair(candidate, local_rng)
                changed = np.flatnonzero(h != seed)
                if len(changed) > 2:
                    reset = local_rng.choice(changed, len(changed) - 2, replace=False)
                    h[reset] = seed[reset]
                return h

            def mutate(self, g, local_rng):
                return self.repair(super().mutate(g, local_rng), local_rng)
        # Proxy forwards all scoring and logging to the global budget session.
        seen = {tuple(x) for x in sampled_users}
        class LocalOracle:
            space = Neighborhood(parent_space.cardinalities)
        class LocalSession:
            oracle = LocalOracle()
            def __init__(self):
                self.calls = 0
                self.best = -np.inf
                self.best_group = seed.copy()
            @property
            def remaining(self):
                return min(local_budget - self.calls, session.remaining)
            def evaluate(self, g):
                if not self.remaining:
                    raise BudgetExhausted()
                self.calls += 1
                score = session.evaluate(g)
                key = parent_space.validate(g)
                result = session.cache[key]
                token = "".join(map(str, key))
                if token not in sampled_users_str:
                    sampled_users_str.append(token)
                if result.valid and key not in seen:
                    sampled_users.append(list(key))
                    sampled_score.append(result.utility)
                    seen.add(key)
                value = result.utility * (1 if direction == "max" else -1) if result.valid else -np.inf
                if value > self.best:
                    self.best, self.best_group = value, np.array(key)
                return value
        local_session = LocalSession()
        # Local engines consume a single seed or a neighborhood population.
        pop_size = getattr(engine, "population", None)
        initial = ([seed] + [local_session.oracle.space.sample(rng) for _ in range(pop_size - 1)]
                   if pop_size else [seed])
        engine.run(local_session, rng, initial)
        if not np.isfinite(local_session.best):
            return temp_sample, score_target
        raw = local_session.best * (1 if direction == "max" else -1)
        return local_session.best_group.tolist(), raw
    return local


class NativeFairASSearch:
    """Preserve native adaptive sampler; optionally replace only local_search.

    Uses supplied frozen per-feature initial probabilities. The native support
    prefilter must be computed independently of fairness scores and passed in.
    A fixed query cap replaces native iteration stopping for budget comparisons.
    """
    def __init__(self, source, feature_names, local_engine=None, one_step_num=5,
                 local_budget=20, probabilities=None):
        self.source, self.feature_names = Path(source), tuple(feature_names)
        self.local_engine, self.one_step_num = local_engine, one_step_num
        self.local_budget, self.probabilities = local_budget, probabilities
        self.name = "fairas_native" if local_engine is None else "fairas_" + local_engine.name

    def run(self, session, rng, initial=None):
        if not isinstance(session.oracle.space, ExactCellSpace) or session.objective_field != "utility":
            raise ValueError("Native FairAS needs exact cells and original utility score")
        if initial is not None:
            raise ValueError("Native FairAS uses its adaptive initialization policy")
        space = session.oracle.space
        if len(self.feature_names) != space.dimension:
            raise ValueError("Feature names must match exact cell dimensions")
        ns = {"np": np, "random": LimitedRandom(max(1000, session.budget * space.dimension * 100)),
              "copy": copy, "Counter": Counter, "time": time,
              "algorithm_params": {"one_step_num": self.one_step_num},
              "feat_dict": dict(zip(self.feature_names, space.cardinalities)),
              "query_count": 0, "query_time": 0.0, "eval_time": 0.0, "unfairness_list": []}
        names = {"sample_one_direction", "count_sensitive_one_direction", "local_search", "sample_group",
                 "update_prob_one_direction", "update_prob", "one_step"}
        revision = source_definitions(self.source, names, ns)
        session.metadata["source_sha256"] = revision
        def run_score(info, g):
            session.evaluate(g)
            result = session.cache[space.validate(g)]
            return result.utility if result.valid else None
        ns["run_score"] = run_score
        if self.local_engine is not None:
            ns["local_search"] = replacement_local_search(self.local_engine, session, rng, self.local_budget)
        info = {"sensitive_features": list(self.feature_names)}
        p_min = copy.deepcopy(self.probabilities or {
            name: [1 / c] * c for name, c in zip(self.feature_names, space.cardinalities)})
        p_max = copy.deepcopy(p_min)
        users, scores, keys = [], [], []
        with seeded_legacy_rng(int(rng.integers(2**31 - 1))), contextlib.redirect_stdout(io.StringIO()):
            try:
                k = 0
                while session.remaining:
                    k += 1
                    p_min, p_max = ns["one_step"](info, p_min, p_max, self.one_step_num * k * 2,
                                                   users, scores, keys)
            except BudgetExhausted:
                pass
            except DrawLimitReached:
                session.metadata["termination"] = "native_duplicate_rejection_stalled"
