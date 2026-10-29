"""Conference controls; original training and benchmark outputs stay immutable.

This is a post hoc extension, not a preregistered independent replication.
Raw traces are compressed; every request, including duplicates/invalids, counts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from compare_fairrec_search import BudgetExhausted, MockDisparity, Space, load_native
from .empirical import atomic_csv, atomic_json, execute_search, feature_frame
from .recommender import NeuMF, load_movielens_1m, model_fingerprint, ranking_utilities
from .subgroup_objectives import CategoricalUtilityGap, CallableSubgroupObjective, json_default

ALGORITHMS = ("DPSO (native)", "GA", "ACO", "DE", "Random")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Ledger:
    """Compact exact request ledger with common caching and archive semantics."""
    def __init__(self, objective, budget, threshold, seed):
        self.objective, self.budget, self.threshold, self.seed = objective, budget, threshold, seed
        self.cache, self.logs, self.high = {}, [], set()
        self.best, self.best_group, self.best_query = -math.inf, None, None

    @property
    def remaining(self):
        return self.budget - len(self.logs)

    def evaluate(self, vector, phase="search", **context):
        if not self.remaining:
            raise BudgetExhausted
        key = self.objective.space.validate(vector)
        cached = key in self.cache
        if not cached:
            detail = self.objective.details(key)
            score = detail["disparity"]
            if detail["valid"] != (score is not None) or (score is not None and not np.isfinite(score)):
                raise ValueError("Inconsistent objective validity")
            self.cache[key] = detail
        detail = self.cache[key]
        score = detail["disparity"]
        if score is not None:
            if score > self.best:
                self.best, self.best_group, self.best_query = score, key, len(self.logs) + 1
            if score >= self.threshold:
                self.high.add(key)
        self.logs.append(dict(query=len(self.logs) + 1, candidate=key, score=score,
                              valid=detail["valid"], cached=cached,
                              best=None if self.best_group is None else self.best,
                              high_count=len(self.high), phase=phase))
        return score


def enumerate_cells(objective):
    return [objective.details(g) for g in itertools.product(
        *(range(c) for c in objective.space.cardinalities))]


def search(objective, algorithm, budget, population, threshold, seed, native_class):
    initial = Space(tuple(objective.space.cardinalities)).sample(
        np.random.default_rng(np.random.SeedSequence([seed, 0])), 2 * population).reshape(
            2, population, objective.space.dimension)
    ledger = Ledger(objective, budget, threshold, seed)
    start = perf_counter()
    if algorithm == "Random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        for end in range(2):
            stop = budget // 2 if end == 0 else budget
            for g in initial[end]:
                ledger.evaluate(g, "shared_initialization")
            while len(ledger.logs) < stop:
                ledger.evaluate(Space(tuple(objective.space.cardinalities)).sample(rng)[0], "uniform_random")
    else:
        execute_search(ledger, algorithm, SimpleNamespace(budget=budget, population=population), initial, native_class)
    elapsed = perf_counter() - start
    assert len(ledger.logs) == budget
    return ledger, elapsed


def save_search(directory, objective, algorithm, args, seed, native_class, common, exact=None):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        # The enclosing immutable protocol has already been checked.
        result = json.loads(result_path.read_text())
        if not (directory / "trace.jsonl.gz").exists():
            raise ValueError(f"Missing committed trace: {directory}")
        return result
    ledger, elapsed = search(objective, algorithm, args.budget, args.population,
                             args.threshold, seed, native_class)
    result = dict(**common, algorithm=algorithm, search_seed=seed,
        maximum_score=None if ledger.best_group is None else ledger.best,
        best_group=ledger.best_group, best_found_query=ledger.best_query,
        requests=len(ledger.logs), unique_groups=len(ledger.cache),
        invalid_requests=sum(not row["valid"] for row in ledger.logs),
        threshold_groups=len(ledger.high), search_seconds=elapsed,
        objective_sha256=objective.fingerprint)
    if exact is not None:
        valid = [r for r in exact if r["valid"]]
        optimum = max(r["disparity"] for r in valid)
        high = sum(r["disparity"] >= args.threshold for r in valid)
        result.update(exact_optimum=optimum, regret=optimum - ledger.best,
                      optimum_found=bool(np.isclose(optimum, ledger.best, atol=1e-12, rtol=0)),
                      valid_cells=len(valid), total_cells=len(exact), reference_high=high,
                      threshold_recall=len(ledger.high) / high if high else None)
        result["first_optimum_query"] = next((r["query"] for r in ledger.logs
            if r["best"] is not None and np.isclose(r["best"], optimum, atol=1e-12, rtol=0)), None)
    with gzip.open(directory / "trace.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in ledger.logs:
            stream.write(json.dumps(row, default=json_default, allow_nan=False) + "\n")
    atomic_json(result_path, result)
    return result


def inference_cache(args, seed, data, audit):
    source = args.baseline / f"seed_{seed}"
    checkpoint_path = source / "neumf.pt"
    cache_path = args.out / f"model_{seed}" / "utilities.npz"
    metadata_path = cache_path.with_suffix(".json")
    expected = dict(checkpoint_sha256=sha(checkpoint_path), data_sha256=data.source_sha256,
        audit_user_sha256=hashlib.sha256(data.user_ids[audit].tobytes()).hexdigest(),
        ranking_source_sha256=sha(Path(__file__).with_name("recommender.py")),
        metric="ndcg", k=10, protocol="original full-catalog train-mask; test also masks validation")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(k) != v for k, v in expected.items()) or metadata["cache_sha256"] != sha(cache_path):
            raise ValueError("Inference cache provenance differs")
        with np.load(cache_path) as values:
            return values["validation"], values["test"], metadata
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["data_sha256"] != data.source_sha256:
        raise ValueError("Checkpoint dataset hash mismatch")
    model = NeuMF(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    original = json.loads((source / "audit_protocol.json").read_text())
    if original["model_sha256"] != model_fingerprint(model):
        raise ValueError("Checkpoint no longer matches original model")
    start = perf_counter()
    validation = ranking_utilities(model, data, "validation", 10, audit)["ndcg"]
    test = ranking_utilities(model, data, "test", 10, audit)["ndcg"]
    metadata = dict(expected, model_sha256=model_fingerprint(model), inference_seconds=perf_counter() - start,
                    note="Test arrays are used for fixed-rule confirmation only, never fitness or tuning")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, validation=validation, test=test, user_ids=data.user_ids[audit])
    atomic_json(metadata_path, dict(metadata, cache_sha256=sha(cache_path)))
    return validation, test, metadata


def paired_bootstrap_gap(utilities, mask, seed=741, repetitions=2000):
    """User bootstrap keeps each member's contribution in the global mean.

    Conditional on a fixed selected rule/model, not simultaneous or selection-
    adjusted inference. This is exploratory same-user interaction confirmation.
    """
    utilities, mask = np.asarray(utilities), np.asarray(mask, bool)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        indices = rng.integers(len(utilities), size=len(utilities))
        selected = mask[indices]
        if selected.any():
            values = utilities[indices]
            samples.append(values.mean() - values[selected].mean())
    return np.quantile(samples, [0.025, 0.975]).tolist()


def empirical(args, native_class):
    config = json.loads((args.baseline / "config.json").read_text())
    data = load_movielens_1m(Path(config["data"]), config["positive_rating"], tuple(config["features"]))
    if data.source_sha256 != config["data_sha256"]:
        raise ValueError("Baseline dataset no longer matches")
    split = json.loads((args.baseline / "user_split.json").read_text())
    audit_ids = split["audit_user_ids"]
    indices = {int(u): i for i, u in enumerate(data.user_ids)}
    audit = np.array([indices[u] for u in audit_ids])
    frame = feature_frame(data, SimpleNamespace(feature_table=None))
    all_results, selection, confirmations = [], [], []
    for model_seed in config["seeds"]:
        print(f"Empirical checkpoint {model_seed}: cached full-catalog inference", flush=True)
        validation, test, metadata = inference_cache(args, model_seed, data, audit)
        report = json.loads((args.baseline / f"seed_{model_seed}/feature_selection.json").read_text())
        root = args.out / f"model_{model_seed}"
        source = f"{metadata['model_sha256']}:{data.source_sha256}:audit-users"
        base = CategoricalUtilityGap.from_frame(frame.iloc[audit], validation,
            args.min_support, source_id=source + ":validation")
        test_base = CategoricalUtilityGap(base.demographics, test, base.feature_names,
            base.categories, args.min_support, source_id=source + ":test")
        exact_by_grammar, objective_by_grammar = {}, {}
        for grammar, names in (("selected", report["selected_features"]), ("unfiltered", list(frame))):
            objective = base.project(names)
            objective_by_grammar[grammar] = objective
            start = perf_counter()
            exact = enumerate_cells(objective)
            elapsed = perf_counter() - start
            exact_by_grammar[grammar] = exact
            atomic_csv(root / f"{grammar}_exact.csv", exact)
            atomic_json(root / f"{grammar}_exact_metadata.json", dict(
                seconds=elapsed, requests=len(exact), objective_sha256=objective.fingerprint,
                stopping="space_exhaustion; separate reference, not forced to spend 4000 requests"))
            # Confirm one fixed exhaustive validation maximizer; never select using test utility.
            best = max((r for r in exact if r["valid"]), key=lambda r: r["disparity"])
            group = best["subgroup"]
            confirmation_objective = test_base.project(names)
            detail = confirmation_objective.details(group)
            mask = np.all(confirmation_objective.demographics == group, axis=1)
            lower, upper = paired_bootstrap_gap(test, mask, seed=model_seed)
            confirmations.append(dict(model_seed=model_seed, grammar=grammar,
                validation_gap=best["disparity"], **detail, test_ci_lower=lower, test_ci_upper=upper,
                interval_scope="exploratory pointwise fixed-rule bootstrap; same audit users; not selection-adjusted"))
            for search_seed in args.search_seeds:
                order = list(ALGORITHMS)
                np.random.default_rng(np.random.SeedSequence([model_seed, search_seed, 999])).shuffle(order)
                for algorithm in order:
                    directory = root / grammar / f"search_{search_seed}" / algorithm.replace(" ", "_")
                    all_results.append(save_search(directory, objective, algorithm, args, search_seed,
                        native_class, dict(model_seed=model_seed, grammar=grammar), exact))
                print(f"  {grammar}: search seed {search_seed} complete", flush=True)
        full = exact_by_grammar["unfiltered"]
        selected_obj = objective_by_grammar["selected"]
        selected_indices = [base.feature_names.index(n) for n in selected_obj.feature_names]
        high = [r for r in full if r["valid"] and r["disparity"] >= args.threshold]
        retained = [r for r in high if selected_obj.details(
            tuple(r["subgroup"][j] for j in selected_indices))["disparity"] >= args.threshold]
        selected_best = max(r["disparity"] for r in exact_by_grammar["selected"] if r["valid"])
        full_best = max(r["disparity"] for r in full if r["valid"])
        selection.append(dict(model_seed=model_seed, selected_features=list(selected_obj.feature_names),
            full_cells=len(full), supported_full_cells=sum(r["valid"] for r in full),
            selected_cells=len(exact_by_grammar["selected"]), full_maximum_gap=full_best,
            selected_maximum_gap=selected_best, maximum_gap_difference=full_best - selected_best,
            high_full_cells=len(high), high_cells_with_high_projected_parent=len(retained),
            parent_threshold_retention=len(retained)/len(high) if high else None,
            high_cell_users=sum(r["support"] for r in high),
            high_cell_users_with_high_parent=sum(r["support"] for r in retained),
            interpretation="threshold attenuation under projection; grammars differ, not optimizer regret"))
        atomic_csv(args.out / "empirical_results.csv", all_results)
        atomic_csv(args.out / "selection_ablation.csv", selection)
        atomic_csv(args.out / "test_confirmation.csv", confirmations)
    return all_results


def landscape(family, seed, permutations):
    space = Space((2, 4, 3) * 8)
    rng = np.random.default_rng(seed)
    nk = MockDisparity.create(space, seed)
    weights = [rng.random(c) for c in space.cardinalities]
    if family == "additive":
        def score(g):
            return float(np.mean([weights[j][v] for j, v in enumerate(g)]))
    elif family == "interaction":
        # Disjoint triples have modular targets and multiple satisfying assignments.
        # Unequal category cardinalities can leave marginal associations.
        targets = rng.integers(0, 3, size=8)
        def score(g):
            return float(np.mean([(sum(g[j:j+3]) % 3) == targets[j//3] for j in range(0, 24, 3)]))
    else:
        score = nk
    def evaluate(encoded):
        return score(tuple(int(permutations[j][v]) for j, v in enumerate(encoded)))
    # Relabeling maps new codes to the same semantic assignments.
    identity = json.dumps(dict(family=family, seed=seed, mapping=[p.tolist() for p in permutations]))
    return CallableSubgroupObjective(evaluate, space.cardinalities,
        tuple(f"x{j}" for j in range(24)), identity)


def synthetic(args, native_class):
    results = []
    for family in ("additive", "nk_trap", "interaction"):
        for landscape_seed in args.landscape_seeds:
            for permutation in range(args.permutations):
                rng = np.random.default_rng(np.random.SeedSequence([landscape_seed, permutation, 517]))
                mapping = [np.arange(c) if permutation == 0 else rng.permutation(c) for c in (2,4,3)*8]
                objective = landscape(family, landscape_seed, mapping)
                base = args.out / "synthetic" / family / f"landscape_{landscape_seed}" / f"encoding_{permutation}"
                base.mkdir(parents=True, exist_ok=True)
                atomic_json(base / "encoding.json", dict(new_code_to_semantic=mapping,
                    identity=permutation == 0, objective_sha256=objective.fingerprint))
                for search_seed in args.synthetic_search_seeds:
                    order = list(ALGORITHMS)
                    np.random.default_rng(np.random.SeedSequence([landscape_seed, permutation, search_seed, 999])).shuffle(order)
                    for algorithm in order:
                        results.append(save_search(base / f"search_{search_seed}" / algorithm.replace(" ", "_"),
                            objective, algorithm, args, search_seed, native_class,
                            dict(family=family, landscape_seed=landscape_seed, encoding=permutation)))
                print(f"Synthetic {family}, landscape {landscape_seed}, encoding {permutation} complete", flush=True)
            atomic_csv(args.out / "synthetic_results.csv", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/final_benchmark"))
    parser.add_argument("--out", type=Path, default=Path("outputs/conference_extension"))
    parser.add_argument("--search-seeds", type=int, nargs="+", default=list(range(1001, 1011)))
    parser.add_argument("--synthetic-search-seeds", type=int, nargs="+", default=list(range(1001, 1006)))
    parser.add_argument("--landscape-seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--synthetic-threshold", type=float, default=0.65)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mode", choices=["all", "empirical", "synthetic"], default="all")
    args = parser.parse_args()
    if args.budget < 2 * args.population or args.population < 4 or args.min_support < 1:
        parser.error("Need population >=4, budget >=2*population, support >=1")
    for seeds in (args.search_seeds, args.synthetic_search_seeds, args.landscape_seeds):
        if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
            parser.error("Seeds must be distinct uint32 values")
    if args.permutations < 1 or args.threads < 1 or not 0 <= args.synthetic_threshold <= 1 or args.threshold < 0:
        parser.error("Invalid encoding/thread/threshold settings")
    torch.set_num_threads(args.threads)
    baseline_config = json.loads((args.baseline / "config.json").read_text())
    native_class, revision = load_native(Path(baseline_config["dpso_source"]))
    if revision != baseline_config["dpso_source_sha256"]:
        raise ValueError("Native DPSO source differs from baseline")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "mode"}
    protocol.update(native_source_sha256=revision, extension_source_sha256=sha(__file__),
        empirical_source_sha256=sha(Path(__file__).with_name("empirical.py")),
        kernel_source_sha256=sha("compare_fairrec_search.py"),
        baseline_config_sha256=sha(args.baseline / "config.json"),
        user_split_sha256=sha(args.baseline / "user_split.json"),
        torch_version=torch.__version__, numpy_version=np.__version__,
        timing="cached utility scoring + search + in-memory trace; excludes inference and compressed output I/O",
        disclosure="Post hoc extension after examining original results; no test-based tuning; no new training",
        pairing="Independent search seeds crossed with each fixed checkpoint; within-grammar shared challenger starts; native initializer retained")
    path = args.out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Output protocol differs; choose a new output directory")
    atomic_json(path, protocol)
    if args.mode in {"all", "empirical"}:
        empirical(args, native_class)
    if args.mode in {"all", "synthetic"}:
        args.threshold = args.synthetic_threshold
        synthetic(args, native_class)
    atomic_json(args.out / f"{args.mode}_complete.json", dict(status="complete", protocol_sha256=sha(path)))


if __name__ == "__main__":
    main()
