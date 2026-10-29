"""End-to-end empirical runner with durable per-seed/per-algorithm artifacts.

The primary arm compares two-ended DPSO/GA/ACO/DE on the SAME selected grammar,
frozen NeuMF, audit user pool, disparity and total request budget. Optional
FairAS arms are separately labeled. Mathematical convergence is never assumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from compare_fairrec_search import ENGINES, Space, load_native, BudgetExhausted
from .categorical_selection import CategoricalUnionBackwardSelector
from .expanded_bias import ExpandedBiasEvaluator, ExpandedBiasObjective, movielens_genre_features
from .fairas_adapter import FairASSearchAdapter, SearchLedger
from .native import seeded_legacy_rng
from .recommender import (NeuMF, RecommenderDisparityObjective, download_movielens,
                          load_movielens_1m, ranking_utilities, train_neumf)
from .subgroup_objectives import CategoricalUtilityGap, json_default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path, rows):
    path = Path(path)
    if not rows:
        # Do not leave an earlier successful master in a failed fresh attempt.
        fields = ["algorithm", "run", "search_seed", "maximum_disparity",
                  "iterations_to_convergence", "subgroup_diversity", "execution_seconds"]
    else:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=json_default) if isinstance(value, (dict, tuple, list)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class StreamingLedger(SearchLedger):
    """Exact fitness requests, including repeats/invalid proposals, flushed to disk.

    Objective calls remain unchanged. Fitness caching is shared across both ends;
    cache hits still consume a request. Trace I/O is timed as part of execution.
    """
    def __init__(self, objective, budget, threshold, path, algorithm, seed, run, block_size):
        super().__init__(objective, budget, threshold)
        self.algorithm, self.seed, self.run, self.block_size = algorithm, seed, run, block_size
        self.best_query, self.previous = None, None
        self.file = Path(path).open("w", encoding="utf-8", buffering=1)

    def evaluate(self, vector, phase="search", **context):
        if not self.remaining:
            raise BudgetExhausted
        previous_best = self.best
        score = super().evaluate(vector, phase, **context)
        if self.best > previous_best:
            self.best_query = len(self.logs)
        row = self.logs[-1]
        candidate = np.asarray(row["candidate"])
        row.update(algorithm=self.algorithm, model_seed=self.seed, search_seed=self.seed, run=self.run,
            objective_sha256=self.objective.fingerprint, query_block=(len(self.logs) - 1) // self.block_size + 1,
            unique_groups=len(self.cache), subgroup_diversity=len(self.high),
            queries_since_best=None if self.best_query is None else len(self.logs) - self.best_query,
            proposal_hamming_from_previous=None if self.previous is None else int(np.count_nonzero(candidate != self.previous)))
        self.previous = candidate.copy()
        self.file.write(json.dumps(row, default=json_default, allow_nan=False) + "\n")
        return score

    def close(self):
        self.file.close()


class SearchEnd:
    def __init__(self, ledger, stop, sign):
        self.ledger, self.stop, self.sign = ledger, stop, sign
        # Existing random-key DE/GA/ACO use this sampling API, not wildcard rules.
        self.space = Space(tuple(ledger.objective.space.cardinalities))

    @property
    def remaining(self):
        return min(self.stop - len(self.ledger.logs), self.ledger.remaining)

    def evaluate(self, vector):
        if not self.remaining:
            raise BudgetExhausted
        score = self.ledger.evaluate(vector, "maximize_gap" if self.sign == 1 else "minimize_gap")
        return -np.inf if score is None else self.sign * score


def thermal_reference(ledger, rng, initial):
    """Independent thermal reference with nullable cells, never labeled native."""
    positions = initial.astype(float).copy()
    personal = positions.copy()
    values = np.full(positions.shape[:2], np.nan)
    for end in range(2):
        for i, group in enumerate(positions[end]):
            score = ledger.evaluate(group, "reference_initialization")
            values[end, i] = np.nan if score is None else score
    while ledger.remaining:
        center = personal.mean(axis=1)
        flat, raw = personal.reshape(-1, positions.shape[-1]), values.reshape(-1)
        valid = np.flatnonzero(np.isfinite(raw))
        guides = ([flat[valid[np.argmax(raw[valid])]].copy(), flat[valid[np.argmin(raw[valid])]].copy()]
                  if len(valid) else [personal[0, 0].copy(), personal[1, 0].copy()])
        for i in range(positions.shape[1]):
            phi = rng.normal()
            for end, sign in enumerate((1, -1)):
                if not ledger.remaining:
                    return
                x = positions[end, i]
                velocity = (0.09 * np.abs(center[end] - x) * phi
                            + 2 * rng.random() * (personal[end, i] - x)
                            + 2 * rng.random() * (guides[end] - x))
                positions[end, i] = np.clip(np.trunc(x + np.clip(velocity, -2, 2)),
                    0, np.asarray(ledger.objective.space.cardinalities) - 1)
                score = ledger.evaluate(positions[end, i], "reference_max" if sign == 1 else "reference_min")
                if score is not None and (not np.isfinite(values[end, i]) or sign * score > sign * values[end, i]):
                    personal[end, i], values[end, i] = positions[end, i].copy(), score


def execute_search(ledger, algorithm, args, initial, native_class=None):
    rng = np.random.default_rng(np.random.SeedSequence([ledger.seed, 1]))
    metadata = dict(stopping="request_budget")
    if algorithm.startswith("DPSO"):
        if native_class is None:
            thermal_reference(ledger, rng, initial)
            metadata["backend"] = "independent_thermal_reference"
        else:
            def fitness(instance, group):
                score = ledger.evaluate(group, "native_dpso")
                # Original FairRec sentinel; it already excludes this value from
                # the minimizing end. Finite raw disparity is never transformed.
                return -100 if score is None else score
            native_class.fitness = fitness
            cards = ledger.objective.space.cardinalities
            parameters = dict(dimension=len(cards), iter=args.budget, size=args.population,
                bound_low=[0] * len(cards), bound_up=[c - 1 for c in cards], v_low=-2, v_high=2,
                value_prob_list=[[list(range(c)), [1 / c] * c] for c in cards])
            legacy_seed = int(rng.integers(2**31 - 1))
            with seeded_legacy_rng(legacy_seed):
                try:
                    native_class(parameters, {}, {}).double_ended_dpso()
                except BudgetExhausted:
                    pass
            metadata.update(backend="native_fairrec_source", native_rng_seed=legacy_seed)
    elif algorithm.startswith("FairAS-"):
        adapter = FairASSearchAdapter(algorithm.split("-")[1], args.population,
            args.local_budget, args.radius, args.one_step_num, args.fairas_source)
        adapter.run(ledger.objective, args.budget, ledger.seed, args.threshold, ledger=ledger)
        metadata.update(adapter.metadata_)
    else:
        for end, sign in enumerate((1, -1)):
            view = SearchEnd(ledger, args.budget // 2 if end == 0 else args.budget, sign)
            ENGINES[algorithm](view, rng, initial[end])
        metadata["backend"] = "categorical_search_kernel"
    if not algorithm.startswith("FairAS-") and ledger.remaining:
        raise RuntimeError(f"{algorithm} returned before using its declared budget")
    return metadata


def block_trace(logs, population):
    rows = []
    for start in range(0, len(logs), population):
        block = logs[start:start + population]
        valid = [row["disparity"] for row in block if row["valid"]]
        rows.append(dict(block=start // population + 1, first_query=block[0]["query"],
            last_query=block[-1]["query"], maximum_in_block=max(valid) if valid else None,
            cumulative_maximum=block[-1]["best_disparity"],
            unique_groups=block[-1]["unique_groups"], subgroup_diversity=block[-1]["subgroup_diversity"],
            invalid_requests=sum(not row["valid"] for row in block)))
    return rows


def feature_frame(data, args):
    if args.feature_table:
        frame = pd.read_csv(args.feature_table, dtype=str)
        if "user_id" not in frame or frame.user_id.duplicated().any():
            raise ValueError("Feature CSV requires unique user_id and categorical attribute columns")
        frame["user_id"] = pd.to_numeric(frame.user_id, errors="raise").astype(int)
        frame = frame.set_index("user_id").loc[data.user_ids]
    else:
        frame = pd.DataFrame({name: [values[v] for v in data.demographics[:, j]]
                              for j, (name, values) in enumerate(zip(data.feature_names, data.categories))},
                             index=data.user_ids)
    if not frame.shape[1] or frame.isna().any().any():
        raise ValueError("Audit features must be present for every eligible user; impute upstream")
    return frame.astype(str)


def select_features(data, frame, development, args, seed):
    if args.selection_labels:
        labels = pd.read_csv(args.selection_labels)
        if not {"user_id", "target"} <= set(labels) or labels.user_id.duplicated().any():
            raise ValueError("Selection labels require unique user_id,target rows")
        y = labels.set_index("user_id").loc[data.user_ids[development], "target"].to_numpy()
        target = args.selection_labels_provenance
    else:
        activity = np.array([len(data.train_seen[u]) for u in development])
        cutoff = float(np.median(activity))
        y = (activity < cutoff).astype(int)
        target = f"training_activity_below_development_median={cutoff}; proxy, not fairness labels"
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or counts.min() < args.selection_cv:
        raise ValueError("Development selection labels need >=2 classes and enough rows per CV fold; supply independent selection labels")
    mandatory = tuple(frame.columns.get_loc(name) for name in args.mandatory_features)
    estimator = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                              DecisionTreeClassifier(max_depth=5, random_state=seed))
    selector = CategoricalUnionBackwardSelector(args.top_n, args.max_features,
        mandatory=mandatory, estimator=estimator, cv=args.selection_cv, seed=seed)
    start = perf_counter()
    selector.fit(frame.iloc[development], y)
    selected = tuple(selector.get_feature_names_out())
    report = dict(selected_features=selected, eliminated_features=[n for n in frame if n not in selected],
        filter_union=[frame.columns[j] for j in selector.filter_union_],
        mrmr_order=[frame.columns[j] for j in selector.mrmr_order_],
        chi_square_order=[frame.columns[j] for j in selector.chi_order_],
        wrapper_history=selector.history_, development_users=data.user_ids[development],
        label_source=target, selection_seconds=perf_counter() - start,
        warning="Activity labels select activity predictors; they do not establish preservation of fairness signals.")
    return report


def audit_objective(base, frame, audit, selected, evaluator):
    callback = None if base.cache_utilities else lambda members: base.score_users(audit[np.asarray(members, int)])
    source = hashlib.sha256(repr((base.fingerprint, audit.tolist(), "audit-user-reference-v1")).encode()).hexdigest()
    table = CategoricalUtilityGap.from_frame(frame.iloc[audit], base.utilities[audit],
        min_support=base.min_support, score_users=callback, source_id=source)
    return ExpandedBiasObjective(table.project(selected), evaluator.subset_users(audit))


def record_error(root, seed, stage, exception, algorithm=None):
    error = dict(seed=seed, algorithm=algorithm, stage=stage, type=type(exception).__name__,
                 message=str(exception), traceback=traceback.format_exc())
    with (root / "errors.jsonl").open("a", encoding="utf-8", buffering=1) as file:
        file.write(json.dumps(error) + "\n")
    print(f"Seed {seed}, {stage}{' / ' + algorithm if algorithm else ''}: {exception}", file=sys.stderr, flush=True)
    return error


def rebuild_master(root, include_traces=False):
    artifacts = sorted(root.glob("seed_*/search/*/result.json"))
    completed = []
    for path in artifacts:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "success":
            completed.append((path.parent, value))
    atomic_csv(root / "final_benchmark_results.csv", [row for _, row in completed])
    if include_traces:
        temporary = root / "trace.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as destination:
            for directory, _ in completed:
                with (directory / "trace.jsonl").open("r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, destination)
        temporary.replace(root / "trace.jsonl")
    return [row for _, row in completed]


def run_algorithm(objective, algorithm, args, seed, run, directory, initial, native_class, common):
    directory.mkdir(parents=True, exist_ok=True)
    ledger = StreamingLedger(objective, args.budget, args.threshold,
                            directory / "trace.jsonl", algorithm, seed, run, args.population)
    try:
        metadata = execute_search(ledger, algorithm, args, initial, native_class)
    finally:
        ledger.close()
    best = None if ledger.best_group is None else ledger.cache[ledger.best_group]
    last_window = ledger.logs[-min(100, len(ledger.logs)):]
    result = dict(status="success", algorithm=algorithm, run=run, model_seed=seed, search_seed=seed,
        maximum_disparity=None if best is None else ledger.best,
        iterations_to_convergence=None if ledger.best_query is None else math.ceil(ledger.best_query / args.population),
        best_found_query=ledger.best_query, subgroup_diversity=len(ledger.high),
        execution_seconds=perf_counter() - ledger.started, requests=len(ledger.logs),
        unique_groups=len(ledger.cache), invalid_requests=sum(not row["valid"] for row in ledger.logs),
        trailing_stagnation_queries=None if ledger.best_query is None else len(ledger.logs) - ledger.best_query,
        last_window_repeat_rate=float(np.mean([row["cached"] for row in last_window])),
        best_subgroup=ledger.best_group, selected_features=list(objective.feature_names),
        population=args.population, threshold=args.threshold, request_budget=args.budget,
        objective_sha256=objective.fingerprint, audit_sha256=objective.audit_fingerprint,
        stopping=metadata["stopping"], **common)
    from .expanded_bias import flatten_bias_report
    result.update(flatten_bias_report(None if best is None else best["expanded_bias"]))
    atomic_json(directory / "algorithm_metadata.json", metadata)
    atomic_csv(directory / "query_blocks.csv", block_trace(ledger.logs, args.population))
    atomic_json(directory / "best_subgroup.json", best)
    atomic_json(directory / "subgroups.json", sorted((detail for detail in ledger.cache.values() if detail["valid"]),
                key=lambda value: -value["disparity"]))
    atomic_csv(directory / "results.csv", [result])
    atomic_json(directory / "config.json", dict(population=args.population, threshold=args.threshold,
                live_inference=args.live_inference, objective_sha256=objective.fingerprint))
    # Commit marker written last. Resume ignores incomplete/failed algorithms.
    atomic_json(directory / "result.json", result)
    return result


def run_seed(data, frame, item_features, development, audit, args, seed, run, native_class, revision):
    directory = args.out / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    stage = "feature_selection"
    try:
        atomic_json(directory / "status.json", dict(status="running", seed=seed, stage=stage))
        feature_path = directory / "feature_selection.json"
        if args.resume and feature_path.exists():
            report = json.loads(feature_path.read_text(encoding="utf-8"))
        else:
            report = select_features(data, frame, development, args, seed)
            atomic_json(feature_path, report)
        selected = tuple(report["selected_features"])
        print(f"Seed {seed}: selected {len(selected)}/{frame.shape[1]} attributes: {selected}", flush=True)
        stage = "training"
        checkpoint_path = directory / "neumf.pt"
        start = perf_counter()
        if args.resume and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
            if checkpoint["data_sha256"] != data.source_sha256:
                raise ValueError("Checkpoint data fingerprint differs")
            model = NeuMF(**checkpoint["model_config"]).to(args.device)
            model.load_state_dict(checkpoint["state_dict"])
            history, training_summary = checkpoint["history"], checkpoint["training_summary"]
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            def validation(model, epoch):
                return ranking_utilities(model, data, k=args.k, user_indices=development)[args.metric].mean()
            with (directory / "training_progress.jsonl").open("w", encoding="utf-8", buffering=1) as progress:
                def epoch_finished(model, row):
                    progress.write(json.dumps(row) + "\n")
                    atomic_checkpoint(directory / "last_completed_epoch.pt", dict(state_dict=model.state_dict(),
                        model_config=model.config, epoch=row["epoch"], data_sha256=data.source_sha256,
                        note="Progress checkpoint only; interrupted training restarts from its seed, not optimizer resume"))
                model, history = train_neumf(data, args.epochs, args.negatives, args.batch_size,
                    args.learning_rate, seed, args.device, args.embedding_dim,
                    validation_callback=validation, patience=args.patience, min_delta=args.min_delta,
                    min_epochs=args.min_epochs, eval_every=args.eval_every, epoch_callback=epoch_finished)
            training_summary = model.training_summary
            atomic_checkpoint(checkpoint_path, dict(state_dict=model.state_dict(), model_config=model.config,
                data_sha256=data.source_sha256, history=history, training_summary=training_summary))
        train_seconds = perf_counter() - start
        atomic_json(directory / "training_summary.json", dict(training_summary, history=history,
                    training_or_loading_seconds=train_seconds))
        stage = "objective_setup"
        start = perf_counter()
        base = RecommenderDisparityObjective(model, data, args.metric, args.k, args.min_support,
                                             cache_utilities=not args.live_inference)
        evaluator = ExpandedBiasEvaluator.from_neumf(base, item_features)
        objective = audit_objective(base, frame, audit, selected, evaluator)
        atomic_json(directory / "audit_protocol.json", dict(feature_names=objective.feature_names,
            categories=objective.categories, global_utility=objective.global_utility,
            objective_sha256=objective.fingerprint, audit_sha256=objective.audit_fingerprint,
            audit_user_ids=data.user_ids[audit], model_sha256=base.model_sha256,
            selected_predicates_only=True, eliminated_attributes="unconstrained"))
        common = dict(model_sha256=base.model_sha256, data_sha256=data.source_sha256,
            training_epochs_completed=training_summary["epochs_completed"],
            best_training_epoch=training_summary["best_epoch"], training_stopping=training_summary["stopping"],
            training_seconds=train_seconds, objective_setup_seconds=perf_counter() - start,
            audit_users=len(audit), metric=args.metric, k=args.k,
            dpso_source_sha256=revision or "")
        initial_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        initial = Space(tuple(objective.space.cardinalities)).sample(initial_rng, 2 * args.population).reshape(
            2, args.population, objective.space.dimension)
        algorithms = [("DPSO (native)" if args.dpso_backend == "native" else "DPSO (thermal reference)")
                      if name == "DPSO" else name for name in args.algorithms]
        if args.include_fairas:
            algorithms.extend(["FairAS-GA", "FairAS-ACO"])
        order = list(algorithms)
        np.random.default_rng(np.random.SeedSequence([seed, 999])).shuffle(order)
        successes, errors = [], []
        stage = "search"
        for algorithm in order:
            arm_directory = directory / "search" / algorithm.replace(" ", "_")
            result_path = arm_directory / "result.json"
            if args.resume and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") == "success" and (arm_directory / "trace.jsonl").exists():
                    successes.append(result)
                    continue
            try:
                print(f"Seed {seed}: {algorithm}, budget={args.budget}", flush=True)
                result = run_algorithm(objective, algorithm, args, seed, run, arm_directory, initial, native_class, common)
                successes.append(result)
            except Exception as exception:
                error = record_error(args.out, seed, stage, exception, algorithm)
                errors.append(error)
                atomic_json(arm_directory / "error.json", error)
            rebuild_master(args.out)
        stage = "test_confirmation"
        # No final test inference occurs before ALL search arms for this seed end.
        test_base = RecommenderDisparityObjective(model, data, args.metric, args.k, args.min_support, split="test")
        test_evaluator = ExpandedBiasEvaluator.from_neumf(test_base, item_features)
        confirmation = audit_objective(test_base, frame, audit, selected, test_evaluator)
        confirmations = []
        for result in successes:
            group = result["best_subgroup"]
            if group is not None:
                detail = confirmation.details(group)
                confirmations.append(dict(algorithm=result["algorithm"], model_seed=seed, run=run, **detail))
        atomic_csv(directory / "test_confirmation.csv", confirmations)
        atomic_json(directory / "status.json", dict(status="complete" if not errors else "search_failures",
                    seed=seed, successful_algorithms=len(successes), failed_algorithms=len(errors)))
        return not errors
    except Exception as exception:
        error = record_error(args.out, seed, stage, exception)
        atomic_json(directory / "status.json", dict(status="failed", stage=stage, error=error))
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_benchmark(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.download and any(not (args.data / name).exists() for name in ("users.dat", "ratings.dat", "movies.dat")):
        download_movielens(args.data)
    data = load_movielens_1m(args.data, args.positive_rating, tuple(args.features))
    frame = feature_frame(data, args)
    item_features, genre_names = movielens_genre_features(data, args.data)
    indices = np.random.default_rng(args.split_seed).permutation(len(data.user_ids))
    n_development = int(len(indices) * args.development_fraction)
    if n_development < 4 or len(indices) - n_development < args.min_support:
        raise ValueError("Insufficient development/audit users for the declared split/support")
    development, audit = np.sort(indices[:n_development]), np.sort(indices[n_development:])
    native_class, revision = (load_native(args.dpso_source) if "DPSO" in args.algorithms and args.dpso_backend == "native" else (None, None))
    configuration = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
                     if key not in {"resume", "generate_assets"}}
    configuration.update(data_sha256=data.source_sha256, dpso_source_sha256=revision or "",
        feature_table_sha256=hashlib.sha256(args.feature_table.read_bytes()).hexdigest() if args.feature_table else "",
        selection_labels_sha256=hashlib.sha256(args.selection_labels.read_bytes()).hexdigest() if args.selection_labels else "",
        torch_version=str(torch.__version__), numpy_version=np.__version__)
    protocol_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "config.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise ValueError("Output already has a benchmark; use --resume or a new --out directory")
        if previous.get("protocol_sha256") != protocol_hash:
            raise ValueError("Resume configuration/data/source versions differ; use a new output directory")
    elif args.resume and list(args.out.glob("seed_*")):
        raise ValueError("Cannot resume seed artifacts without their protocol manifest")
    configuration.update(protocol_sha256=protocol_hash, dataset="movielens-1m",
        protocol="selection first; development-user early stopping; audit-user validation search; separate test confirmation",
        initialization="Shared categorical starts except native initializer, whose uniform category distribution is matched",
        query_policy="All requests count, including initialization, repeats and invalid cells; challengers split budget over both ends",
        convergence="Exact traces and observed stagnation only; no mathematical convergence claim",
        selected_grammar="Partial conjunction over selected features; eliminated attributes unconstrained")
    atomic_json(manifest_path, configuration)
    atomic_json(args.out / "user_split.json", dict(development_user_ids=data.user_ids[development],
                                                  audit_user_ids=data.user_ids[audit]))
    completed = []
    try:
        for run, seed in enumerate(args.seeds):
            status_path = args.out / f"seed_{seed}" / "status.json"
            if args.resume and status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                print(f"Seed {seed}: already complete; skipped", flush=True)
                completed.append(True)
                continue
            completed.append(run_seed(data, frame, item_features, development, audit, args, seed, run, native_class, revision))
            rebuild_master(args.out, include_traces=True)
    finally:
        results = rebuild_master(args.out, include_traces=True)
    summary = dict(protocol_sha256=protocol_hash, requested_seeds=args.seeds,
        complete_seeds=sum(completed), failed_seeds=len(completed) - sum(completed), successful_algorithm_runs=len(results))
    if args.generate_assets and all(completed) and completed:
        try:
            from .publication import ExperimentVisualizer
            ExperimentVisualizer([args.out / "final_benchmark_results.csv"], algorithms=("ALL",)).generate(args.out / "paper_assets")
            summary["paper_assets"] = "generated"
        except Exception as exception:
            record_error(args.out, None, "publication_assets", exception)
            summary["paper_assets"] = "failed; raw benchmark artifacts retained"
    atomic_json(args.out / "benchmark_summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["movielens-1m"], default="movielens-1m")
    parser.add_argument("--data", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2026])
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs, with development-only early stopping")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--positive-rating", type=int, default=4)
    parser.add_argument("--features", nargs="+", default=["gender", "age", "occupation"])
    parser.add_argument("--feature-table", type=Path, help="Optional user_id + 200+ real categorical attributes")
    parser.add_argument("--selection-labels", type=Path, help="Independent development user_id,target CSV")
    parser.add_argument("--selection-labels-provenance", default="")
    parser.add_argument("--development-fraction", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--selection-cv", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=2)
    parser.add_argument("--mandatory-features", nargs="*", default=[])
    parser.add_argument("--metric", choices=["ndcg", "recall"], default="ndcg")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--algorithms", nargs="+", choices=["DPSO", "GA", "ACO", "DE"], default=["DPSO", "GA", "ACO", "DE"])
    parser.add_argument("--dpso-backend", choices=["native", "reference"], default="native")
    default_source = Path("C:/Users/DELL/Documents/repo/FairRec/fairrec.py")
    parser.add_argument("--dpso-source", type=Path, default=default_source if default_source.exists() else None)
    parser.add_argument("--include-fairas", action="store_true")
    parser.add_argument("--fairas-source", type=Path, help="Optional native FairAS outer loop; otherwise explicitly portable")
    parser.add_argument("--local-budget", type=int, default=36)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--one-step-num", type=int, default=5)
    parser.add_argument("--live-inference", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-assets", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/full_empirical_benchmark"))
    return parser


def validate_args(args):
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("Seeds must be distinct integers in [0,2**32)")
    if len(set(args.algorithms)) != len(args.algorithms):
        raise ValueError("Algorithms must be distinct")
    if args.population < 4 or args.budget < 2 * args.population:
        raise ValueError("Need population>=4 and total budget>=2*population")
    if not 0 < args.development_fraction < 1 or args.selection_cv < 2:
        raise ValueError("Invalid development split or selection CV")
    if min(args.epochs, args.negatives, args.batch_size, args.threads, args.k, args.min_support,
           args.top_n, args.max_features, args.patience, args.min_epochs, args.eval_every, args.local_budget, args.one_step_num) < 1:
        raise ValueError("Positive training/selection/search settings required")
    if args.threshold < 0 or args.min_delta < 0 or args.learning_rate <= 0 or args.radius < 0:
        raise ValueError("Invalid threshold, temperature-free training, or neighborhood settings")
    if "DPSO" in args.algorithms and args.dpso_backend == "native" and (args.dpso_source is None or not args.dpso_source.is_file()):
        raise ValueError("Native DPSO requires --dpso-source FairRec/fairrec.py; choose --dpso-backend reference explicitly for portable runs")
    if args.selection_labels and not args.selection_labels_provenance:
        raise ValueError("Declare --selection-labels-provenance for external development labels")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        summary = run_benchmark(args)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if not summary["failed_seeds"] else 2
    except KeyboardInterrupt:
        print("Interrupted; completed artifacts retained. Use --resume with the same configuration.", file=sys.stderr)
        return 130
    except Exception as exception:
        print(f"Benchmark setup failed: {exception}", file=sys.stderr)
        return 1
