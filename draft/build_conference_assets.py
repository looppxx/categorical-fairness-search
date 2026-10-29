"""Summarize completed extension records; no model fitting or search is run."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from draft.build_publication_assets import table
from fairness_lab.empirical import atomic_csv, atomic_json
from fairness_lab.conference import landscape


def verify_traces(source, protocol):
    """Check every recorded request; compare all empirical scores with exact cells.

    Synthetic scores receive deterministic semantic spot checks, in addition to
    full request/cache/archive checks. This does not pretend to rerun all searches.
    """
    requests = runs = semantic_checks = 0
    for path in sorted(source.rglob("result.json")):
        result = json.loads(path.read_text())
        synthetic = "family" in result
        if synthetic:
            encoding = json.loads((path.parent.parent.parent / "encoding.json").read_text())
            objective = landscape(result["family"], result["landscape_seed"],
                                  [np.array(v) for v in encoding["new_code_to_semantic"]])
            threshold = protocol["synthetic_threshold"]
            exact = None
        else:
            exact_path = source / f"model_{result['model_seed']}" / f"{result['grammar']}_exact.csv"
            with exact_path.open(newline="", encoding="utf-8") as stream:
                exact = {tuple(json.loads(row["subgroup"])): None if not row["disparity"] else float(row["disparity"])
                         for row in csv.DictReader(stream)}
            threshold = protocol["threshold"]
        seen, high = {}, set()
        best, best_query, invalid, count = -math.inf, None, 0, 0
        with gzip.open(path.with_name("trace.jsonl.gz"), "rt", encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line)
                key, score = tuple(row["candidate"]), row["score"]
                assert row["query"] == count and row["cached"] == (key in seen), path
                assert row["valid"] == (score is not None), path
                if key in seen:
                    assert score == seen[key], path
                seen[key] = score
                if exact is not None:
                    truth = exact[key]
                    assert (truth is None and score is None) or (truth is not None and score is not None and abs(truth-score) < 1e-12), path
                    semantic_checks += 1
                elif count % 61 == 1:
                    assert score is not None and abs(objective(key)-score) < 1e-12, path
                    semantic_checks += 1
                if score is None:
                    invalid += 1
                else:
                    assert math.isfinite(score), path
                    if score > best:
                        best, best_query = score, count
                    if score >= threshold:
                        high.add(key)
                assert row["best"] == (None if best_query is None else best), path
                assert row["high_count"] == len(high), path
        assert count == protocol["budget"] == result["requests"], path
        assert len(seen) == result["unique_groups"] and len(high) == result["threshold_groups"], path
        assert invalid == result["invalid_requests"] and best_query == result["best_found_query"], path
        assert abs(best-result["maximum_score"]) < 1e-12, path
        runs += 1
        requests += count
    atomic_json(source / "trace_validation.json", dict(runs=runs, requests=requests,
        semantic_checks=semantic_checks, status="passed", scope="all request/cache/archive records; every empirical score; every 61st synthetic score"))
    print("Verified", requests, "requests across", runs, "compressed traces", flush=True)


def main():
    source = ROOT / "outputs/conference_extension"
    out = ROOT / "draft/assets"
    protocol = json.loads((source / "protocol.json").read_text())
    if not (source / "all_complete.json").exists():
        raise ValueError("Full extension has not completed")
    empirical = pd.read_csv(source / "empirical_results.csv")
    synthetic = pd.read_csv(source / "synthetic_results.csv")
    selection = pd.read_csv(source / "selection_ablation.csv")
    confirmation = pd.read_csv(source / "test_confirmation.csv")
    diagnostics = []
    for seed in (42, 100, 2026):
        exact = pd.read_csv(source / f"model_{seed}/unfiltered_exact.csv")
        with np.load(source / f"model_{seed}/utilities.npz") as cache:
            utilities = cache["validation"]
            diagnostics.append(dict(model_seed=seed, validation_zero_utility_fraction=float(np.mean(utilities == 0)),
                global_validation_utility=float(utilities.mean()),
                tied_full_maximizers=int(np.isclose(exact.disparity, exact.disparity.max(), atol=1e-12, rtol=0).sum())))
    atomic_csv(source / "empirical_diagnostics.csv", diagnostics)
    expected_empirical = 3 * 2 * 5 * len(protocol["search_seeds"])
    expected_synthetic = 3 * len(protocol["landscape_seeds"]) * protocol["permutations"] * len(protocol["synthetic_search_seeds"]) * 5
    assert len(empirical) == expected_empirical and len(synthetic) == expected_synthetic
    assert (empirical.requests == protocol["budget"]).all() and (synthetic.requests == protocol["budget"]).all()
    verify_traces(source, protocol)
    assert empirical.groupby(["model_seed", "grammar", "algorithm"]).size().eq(len(protocol["search_seeds"])).all()
    assert not empirical.duplicated(["model_seed", "grammar", "algorithm", "search_seed"]).any()
    assert not synthetic.duplicated(["family", "landscape_seed", "encoding", "algorithm", "search_seed"]).any()
    assert empirical.groupby(["model_seed", "grammar"]).objective_sha256.nunique().eq(1).all()
    assert synthetic.groupby(["family", "landscape_seed", "encoding"]).objective_sha256.nunique().eq(1).all()
    original = pd.read_csv(ROOT / "outputs/final_benchmark/final_benchmark_results.csv")
    for seed in original.model_seed.unique():
        old = original.loc[original.model_seed == seed, "maximum_disparity"].iloc[0]
        exact = empirical.loc[(empirical.model_seed == seed) & (empirical.grammar == "selected"), "exact_optimum"].iloc[0]
        assert np.isclose(old, exact, atol=1e-12, rtol=0), "Original selected objective changed"
    names = ["DPSO (native)", "GA", "ACO", "DE", "Random"]
    rows, empirical_summary = [], []
    for grammar in ("selected", "unfiltered"):
        for name in names:
            frame = empirical[(empirical.grammar == grammar) & (empirical.algorithm == name)]
            record = dict(grammar=grammar, algorithm=name, runs=len(frame),
                optimum_hit_rate=frame.optimum_found.mean(), mean_regret=frame.regret.mean(),
                mean_threshold_recall=frame.threshold_recall.mean(),
                median_first_optimum=frame.first_optimum_query.median(),
                mean_unique_cells=frame.unique_groups.mean(), mean_seconds=frame.search_seconds.mean())
            empirical_summary.append(record)
            rows.append(["Sel." if grammar == "selected" else "Full", name.replace(" (native)", ""),
                f"{record['optimum_hit_rate']*100:.0f}", f"{record['mean_regret']:.5f}",
                f"{record['mean_threshold_recall']*100:.1f}", f"{record['median_first_optimum']:.1f}"])
    table(out / "extended_neumf.tex",
        "Checkpoint controls: 30 searches per method and grammar (ten search seeds on each of three fixed models), 4,000 requests each. Hit is exact-optimum attainment (percent); recall is the mean fraction of supported cells above 0.02 recovered, excluding instances with no such cells. Query is median first optimum request among successes. Full uses all three demographics.",
        "tab:extended-neumf", ["Space", "Method", r"Hit\%", "Regret", r"Rec.\%", "Query"], rows, "llrrrr")
    atomic_csv(source / "empirical_summary.csv", empirical_summary)
    anytime = []
    for grammar in ("selected", "unfiltered"):
        for name in names:
            frame = empirical[(empirical.grammar == grammar) & (empirical.algorithm == name)]
            for budget in (10, 20, 40, 80, 160, 320, 640, 1280, 4000):
                anytime.append(dict(grammar=grammar, algorithm=name, requests=budget,
                    optimum_fraction=(frame.first_optimum_query.fillna(np.inf) <= budget).mean()))
    atomic_csv(source / "attainment_summary.csv", anytime)
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    colors = ["#64748b", "#007c91", "#bf6c24", "#4c70b5", "#8467a6"]
    for name, color in zip(names, colors):
        frame = empirical[(empirical.grammar == "unfiltered") & (empirical.algorithm == name)]
        hits = frame.first_optimum_query.fillna(np.inf).to_numpy()
        queries = sorted({1, 4000, *(int(q) for q in hits if np.isfinite(q))})
        ax.step(queries, [np.mean(hits <= q) for q in queries], where="post", color=color,
                label=name.replace(" (native)", ""), linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlim(1, 4000)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Objective requests (log scale)", fontsize=8)
    ax.set_ylabel("Exact-optimum attainment", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "optimizer_attainment.png", dpi=260)
    fig.savefig(out / "optimizer_attainment.svg")
    plt.close(fig)
    rows = []
    for row in selection.itertuples():
        rows.append([row.model_seed, int(row.selected_cells), int(row.supported_full_cells),
            f"{row.selected_maximum_gap:.5f}", f"{row.full_maximum_gap:.5f}",
            f"{row.high_cells_with_high_projected_parent}/{row.high_full_cells}"])
    table(out / "selection_ablation.tex",
        "Selection audit against exhaustive full conjunctions (294 possible cells per model). Valid counts supported full cells; retained counts high-gap full cells whose selected-feature parent also exceeds 0.02. This is threshold attenuation under aggregation, not a same-grammar optimizer comparison.",
        "tab:selection-ablation", ["Seed", "Sel.", "Valid", r"$\Delta_{sel}$", r"$\Delta_{full}$", "Retained"], rows, "rrrrrr")
    rows = []
    interval_rows = []
    for row in confirmation.itertuples():
        # For fixed memberships and bounded independent user outcomes, the gap
        # is (1-p)*(mean_complement-mean_group). Apply two Hoeffding bounds
        # with a union bound. This guards against zero bootstrap variance in
        # small groups with no observed hits. Selection dependence still prevents
        # interpreting these exploratory intervals as adjusted population claims.
        n = int(row.support)
        with np.load(source / f"model_{row.model_seed}" / "utilities.npz") as cache:
            total = len(cache["user_ids"])
        p = n / total
        radius = (1-p) * (np.sqrt(np.log(4/.05)/(2*n)) + np.sqrt(np.log(4/.05)/(2*(total-n))))
        lower, upper = max(-(1-p), row.disparity-radius), min(1-p, row.disparity+radius)
        interval_rows.append(dict(model_seed=row.model_seed, grammar=row.grammar,
            group_support=n, audit_users=total, test_gap=row.disparity,
            bootstrap_lower=row.test_ci_lower, bootstrap_upper=row.test_ci_upper,
            bounded_lower=lower, bounded_upper=upper,
            scope="fixed-rule exploratory; no adaptive-selection or multiplicity guarantee"))
        rows.append([row.model_seed, "Sel." if row.grammar == "selected" else "Full",
            int(row.support), f"{row.validation_gap:.4f}", f"{row.disparity:+.4f}",
            f"[{lower:+.3f},{upper:+.3f}]"])
    table(out / "extended_confirmation.tex",
        "Fixed exhaustive validation maximizers on test interactions of the same audit users. Conservative pointwise bounded-outcome intervals guard against sparse-cell bootstrap degeneracy. Neither these nor the separately recorded 2,000-resample bootstrap intervals adjust for adaptive selection or provide independent-user confirmation.",
        "tab:extended-confirmation", ["Seed", "Space", "N", "Val.", "Test", "Interval"], rows, "rlrrrl")
    atomic_csv(source / "confirmation_intervals.csv", interval_rows)
    rows, summaries = [], []
    family_names = {"additive": "Additive", "nk_trap": "NK/trap", "interaction": "Modular"}
    for family in family_names:
        for name in names:
            frame = synthetic[(synthetic.family == family) & (synthetic.algorithm == name)]
            landscape_means = frame.groupby("landscape_seed").maximum_score.mean()
            encoding_means = frame.groupby(["landscape_seed", "encoding"]).maximum_score.mean()
            spreads = encoding_means.groupby("landscape_seed").agg(lambda values: values.max()-values.min())
            record = dict(family=family, algorithm=name, runs=len(frame),
                mean_maximum=landscape_means.mean(), landscape_sd=landscape_means.std(),
                mean_threshold_groups=frame.threshold_groups.mean(), mean_encoding_spread=spreads.mean(),
                mean_seconds=frame.search_seconds.mean())
            summaries.append(record)
            rows.append([family_names[family], name.replace(" (native)", ""),
                f"{record['mean_maximum']:.4f}", f"{record['mean_threshold_groups']:.1f}",
                f"{record['mean_encoding_spread']:.4f}"])
    table(out / "synthetic_robustness.tex",
        "Synthetic robustness: 45 searches per family/method, crossing three landscapes, three encodings, and five search seeds. Score is mean maximum; breadth counts distinct candidates scoring at least 0.65. Spread is the mean across landscapes of the range of encoding-specific mean scores. These are descriptive summaries; score scales differ by family.",
        "tab:synthetic-robustness", ["Family", "Method", "Score", "Breadth", "Spread"], rows, "llrrr")
    atomic_csv(source / "synthetic_summary.csv", summaries)
    # Show instance-level variability, averaging repeated encodings/searches
    # within each instance, rather than treating encodings as independent data.
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7), constrained_layout=True)
    colors = ["#64748b", "#007c91", "#bf6c24", "#4c70b5", "#8467a6"]
    for ax, family in zip(axes, family_names):
        frame = synthetic[synthetic.family == family]
        for i, (name, color) in enumerate(zip(names, colors)):
            values = frame[frame.algorithm == name].groupby("landscape_seed").maximum_score.mean()
            ax.bar(i, values.mean(), color=color, width=.65, alpha=.85)
            ax.scatter(np.full(len(values), i), values, color="#111827", s=12, zorder=3)
        ax.set_title(family_names[family], fontsize=10)
        ax.set_xticks(range(5), ["DPSO", "GA", "ACO", "DE", "Rand."], rotation=30, fontsize=8)
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean best synthetic score", fontsize=9)
    fig.savefig(out / "synthetic_robustness.png", dpi=240)
    fig.savefig(out / "synthetic_robustness.svg")
    plt.close(fig)
    inputs = [p for p in source.rglob("*") if p.is_file() and p.name != "evidence_manifest.json"]
    inputs += [ROOT / p for p in ("fairness_lab/conference.py", "run_conference_extension.py",
        "tests/test_conference_extension.py", "draft/build_conference_assets.py")]
    evidence = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    atomic_json(out / "conference_evidence.json", dict(input_sha256=evidence,
        empirical_runs=len(empirical), synthetic_runs=len(synthetic),
        independent_checkpoints=3, synthetic_instances_per_family=3,
        post_hoc=True, inference_scope="fixed-rule pointwise bootstrap only"))
    print("Verified and summarized", len(empirical), "empirical and", len(synthetic), "synthetic runs")
    print(pd.DataFrame(empirical_summary).to_string(index=False))
    print(selection.to_string(index=False))
    print(confirmation[["model_seed", "grammar", "validation_gap", "disparity", "test_ci_lower", "test_ci_upper"]].to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
