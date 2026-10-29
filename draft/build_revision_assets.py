"""Build the three-dataset manuscript from preserved experiment records.

MovieLens, LFM360K, and Amazon form the manuscript scope. Other exploratory
records remain archived. Derived summaries and validation go to a separate
paper directory; the original multi-dataset records are never rewritten.
"""
from pathlib import Path
import argparse
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
from fairness_lab.conference import landscape
from fairness_lab.empirical import atomic_csv, atomic_json

ADDED = ROOT / "outputs/revision_extension"
EXTERNAL = ROOT / "outputs/revision_external_complete"
ORIGINAL = ROOT / "outputs/conference_extension"
OUT = ROOT / "draft/assets"
PAPER = ROOT / "outputs/manuscript_scope"
METHODS = ("DPSO (native)", "GA", "ACO", "DE", "ABC", "SA", "Random")
DATASETS = (("last-fm", "LFM360K"), ("amazon", "Amazon"))
DATASET_IDS = tuple(dataset for dataset, _ in DATASETS)


def manuscript_rows(frame):
    """Select complete dataset blocks without changing individual outcomes."""
    selected = frame.loc[frame.dataset.isin(DATASET_IDS)].copy()
    assert set(selected.dataset) == set(DATASET_IDS), "A manuscript dataset is missing"
    return selected


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table(name, caption, label, headings, rows, alignment, wide=False):
    env = "table*" if wide else "table"
    lines = [r"\begin{"+env+r"}[t]", r"\centering", r"\caption{"+caption+"}",
             r"\label{"+label+"}", r"\small", r"\begin{tabular}{"+alignment+"}",
             r"\toprule", " & ".join(headings)+r" \\", r"\midrule"]
    lines += [" & ".join(str(v) for v in row)+r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{"+env+"}", ""]
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")


def verify_new_traces():
    counts = dict(runs=0, requests=0, exact_score_checks=0, synthetic_score_checks=0)
    inherited = json.loads((EXTERNAL / "external_protocol.json").read_text()).get("inherited_records", {})
    for relative, digest in inherited.items():
        record_path = Path(relative.replace("\\", "/"))
        if record_path.parts[0] in DATASET_IDS:
            assert sha(EXTERNAL / record_path) == digest, "Inherited result record changed"
    for source in (ADDED, EXTERNAL):
        for path in sorted(source.rglob("result.json")):
            # Exclude the interrupted attempt, whose selected searches are now
            # validated in the completed external directory.
            if source == ADDED and path.relative_to(source).parts[0] not in ("movielens", "synthetic"):
                continue
            if source == EXTERNAL and path.relative_to(source).parts[0] not in DATASET_IDS:
                continue
            result = json.loads(path.read_text())
            synthetic = "family" in result
            if synthetic:
                instance, encoding = result["landscape_seed"], result["encoding"]
                rng = np.random.default_rng(np.random.SeedSequence([instance, encoding, 517]))
                mapping = [np.arange(c) if encoding == 0 else rng.permutation(c) for c in (2,4,3)*8]
                objective = landscape(result["family"], instance, mapping)
                truth, threshold = None, .65
            else:
                exact_path = ((EXTERNAL / result["dataset"]) if "dataset" in result else
                              (ORIGINAL / f"model_{result['model_seed']}")) / f"{result['grammar']}_exact.csv"
                with exact_path.open(newline="", encoding="utf-8") as stream:
                    truth = {tuple(json.loads(r["subgroup"])): None if not r["disparity"] else float(r["disparity"])
                             for r in csv.DictReader(stream)}
                threshold = .02
            seen, high, best, best_query, invalid = {}, set(), -math.inf, None, 0
            with gzip.open(path.with_name("trace.jsonl.gz"), "rt", encoding="utf-8") as stream:
                for count, line in enumerate(stream, 1):
                    row = json.loads(line)
                    group, score = tuple(row["candidate"]), row["score"]
                    assert row["query"] == count and row["cached"] == (group in seen), path
                    assert row["valid"] == (score is not None), path
                    if group in seen:
                        assert score == seen[group], path
                    seen[group] = score
                    if truth is not None:
                        target = truth[group]
                        assert (score is None and target is None) or (score is not None and target is not None and abs(score-target)<1e-12), path
                        counts["exact_score_checks"] += 1
                    elif count % 61 == 1:
                        assert abs(objective(group)-score)<1e-12, path
                        counts["synthetic_score_checks"] += 1
                    if score is None:
                        invalid += 1
                    else:
                        assert math.isfinite(score)
                        if score > best:
                            best, best_query = score, count
                        if score >= threshold:
                            high.add(group)
                    assert row["best"] == (None if best_query is None else best) and row["high_count"] == len(high), path
            assert count == result["requests"] == 4000, path
            assert len(seen) == result["unique_groups"] and len(high) == result["threshold_groups"], path
            assert invalid == result["invalid_requests"] and best_query == result["best_found_query"], path
            assert abs(best-result["maximum_score"]) < 1e-12, path
            counts["runs"] += 1
            counts["requests"] += count
    assert counts["runs"] == 670, counts
    atomic_json(PAPER / "trace_validation.json", dict(status="passed", datasets=DATASET_IDS, **counts))
    print("Verified new evidence:", counts, flush=True)


def main():
    global ADDED, EXTERNAL, PAPER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--added", type=Path, default=ADDED)
    parser.add_argument("--external", type=Path, default=EXTERNAL)
    parser.add_argument("--paper-out", type=Path, default=PAPER)
    args = parser.parse_args()
    ADDED, EXTERNAL, PAPER = args.added.resolve(), args.external.resolve(), args.paper_out.resolve()
    assert PAPER not in (ADDED, EXTERNAL), "Derived paper files must not overwrite source records"
    assert (ADDED / "existing_complete.json").exists()
    assert (EXTERNAL / "external_complete.json").exists()
    verify_new_traces()
    synthetic = pd.concat([pd.read_csv(ORIGINAL / "synthetic_results.csv"),
                           pd.read_csv(ADDED / "synthetic_added_results.csv")], ignore_index=True)
    movie = pd.concat([pd.read_csv(ORIGINAL / "empirical_results.csv"),
                       pd.read_csv(ADDED / "movielens_added_results.csv")], ignore_index=True)
    ext = manuscript_rows(pd.read_csv(EXTERNAL / "external_search_results.csv"))
    assert len(synthetic) == 945 and len(movie) == 420 and len(ext) == 280
    assert not synthetic.duplicated(["family","landscape_seed","encoding","search_seed","algorithm"]).any()
    assert not movie.duplicated(["model_seed","grammar","search_seed","algorithm"]).any()
    assert not ext.duplicated(["dataset","grammar","search_seed","algorithm"]).any()
    assert set(ext.algorithm) == set(METHODS)
    rows, summaries = [], []
    for method in METHODS:
        row = [method.replace(" (native)", "")]
        for family in ("additive", "nk_trap", "interaction"):
            f = synthetic[(synthetic.algorithm == method) & (synthetic.family == family)]
            assert len(f) == 45
            per_encoding = f.groupby(["landscape_seed","encoding"]).maximum_score.mean()
            spread = per_encoding.groupby(level=0).agg(lambda x: x.max()-x.min()).mean()
            score, breadth = f.maximum_score.mean(), f.threshold_groups.mean()
            row += [f"{score:.4f}", f"{breadth:.1f}", f"{spread:.4f}"]
            summaries.append(dict(family=family, algorithm=method, mean_score=score, mean_breadth=breadth, encoding_spread=spread))
        rows.append(row)
    table("revision_synthetic.tex", "Synthetic comparison: 945 searches, with 45 per family/method. Each family lists mean maximum score, mean distinct candidates above 0.65 (breadth), and mean within-instance encoding spread. These are descriptive search outcomes on designed landscapes, not population utility disparities.",
          "tab:synthetic-robustness", ["Method","Add. score","Breadth","Spread","NK score","Breadth","Spread","Mod. score","Breadth","Spread"], rows, "lrrrrrrrrr", True)
    atomic_csv(PAPER / "synthetic_summary.csv", summaries)
    rows = []
    for grammar in ("selected", "unfiltered"):
        for method in METHODS:
            f = movie[(movie.algorithm == method) & (movie.grammar == grammar)]
            assert len(f) == 30
            rows.append(["Sel." if grammar=="selected" else "Full", method.replace(" (native)",""),
                f"{100*f.optimum_found.mean():.0f}", f"{f.regret.mean():.5f}",
                f"{100*f.threshold_recall.mean():.1f}", f"{f.first_optimum_query.median():.1f}"])
    table("revision_neumf.tex", "MovieLens: 30 searches per method/grammar (ten seeds on three fixed models). Hit is exact-optimum attainment; recall is the fraction of supported above-threshold cells recovered, omitting cases with no such cells. Query is median first optimum request among successful searches. Timings from the earlier and indexed implementations are not pooled.",
          "tab:extended-neumf", ["Space","Method",r"Hit\%","Regret",r"Rec.\%","Query"], rows,"llrrrr")
    fig, ax = plt.subplots(figsize=(3.5,2.5), constrained_layout=True)
    for method in METHODS:
        f = movie[(movie.algorithm==method)&(movie.grammar=="unfiltered")]
        hits = f.first_optimum_query.fillna(np.inf).to_numpy()
        queries = sorted({1,4000,*(int(q) for q in hits if np.isfinite(q))})
        ax.step(queries,[np.mean(hits<=q) for q in queries],where="post",label=method.replace(" (native)",""),linewidth=1.2)
    ax.set(xscale="log", xlim=(1,4000), ylim=(0,1.03), xlabel="Objective requests (log scale)", ylabel="Exact-optimum attainment")
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5,loc="lower right",frameon=False,ncol=2)
    ax.grid(alpha=.2); ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"revision_attainment.png",dpi=260); plt.close(fig)
    controls = pd.read_csv(ROOT/"outputs/utility_support_extension/support_results.csv")
    rows = []
    for seed in (42,100,2026):
        f=controls[(controls.model_seed==seed)&(controls.min_support==20)].set_index("arm")
        rows.append([seed]+[f"{f.loc[a,'validation_gap']:.4f} ({int(f.loc[a,'high_cells_with_high_parent'])}/{int(f.loc[a,'high_full_cells'])})"
            for a in ("activity_cap2","activity_cap3","utility_cap2","utility_cap3","unfiltered")])
    table("revision_target_ablation.tex", "MovieLens target-by-cap ablation at support 20: exact maximum gap and, in parentheses, high full cells with an above-threshold parent divided by all high full cells. A/U denote activity/utility-deficit targets; 2/3 are maximum subset sizes. Each selection arm applies backward elimination from the complete audit schema.",
          "tab:target-ablation", ["Seed","A2","A3","U2","U3","Full"], rows,"rlllll",True)
    ablations=manuscript_rows(pd.read_csv(EXTERNAL/"external_ablation.csv"))
    assert len(ablations) == 24
    records={r["dataset"]:r for r in json.loads((EXTERNAL/"external_datasets.json").read_text())}
    datasets=[records[dataset] for dataset in DATASET_IDS]
    atomic_csv(PAPER/"external_search_results.csv", ext.to_dict("records"))
    atomic_csv(PAPER/"external_ablation.csv", ablations.to_dict("records"))
    atomic_json(PAPER/"external_datasets.json", datasets)
    rows=[]
    for dataset,label in DATASETS:
        record=records[dataset]
        assert record["dataset"] == dataset
        f=ablations[(ablations.dataset==dataset)&(ablations.min_support==20)].set_index("arm")
        full,sel=f.loc["unfiltered"],f.loc["utility_cap2"]
        assert full.valid_cells > 0 and sel.valid_cells > 0
        assert pd.notna(full.maximum_gap) and pd.notna(sel.maximum_gap) and full.high_full_cells > 0
        report=json.loads((EXTERNAL/dataset/"selection.json").read_text())
        names=report["arms"]["utility_cap2"]["selected_features"]
        short={"gender":"gender","country":"country","age":"age","his_level":"history","mean_price_level":"mean price","max_price_level":"max price"}
        rows.append([label,f"{record['development_users']:,}/{record['discovery_users']:,}/{record['confirmation_users']:,}",
            f"{int(full.total_cells):,}/{int(full.valid_cells):,}", ", ".join(short.get(n,n.replace("_"," ")) for n in names),
            f"{full.maximum_gap:.4f}", f"{sel.maximum_gap:.4f}",
            f"{int(sel.retained_full_cells)}/{int(full.high_full_cells)}"])
    table("revision_datasets.tex", "Additional frozen DeepFM audits at support 20. Users are development/discovery/confirmation; cells are total/supported full conjunctions; retained counts full high-gap cells whose selected parent remains above 0.02. Selected features use development-utility labels and a two-feature cap.",
          "tab:additional-datasets", ["Dataset","Users (D/A/C)","Full cells","Selected predicates",r"$\Delta_{full}$",r"$\Delta_{sel}$","Retained"],rows,"llrlrrr",True)
    rows=[]
    summary=[]
    for dataset,label in DATASETS:
        for grammar in ("selected","unfiltered"):
            row=[label,"Sel." if grammar=="selected" else "Full"]
            for method in METHODS:
                f=ext[(ext.dataset==dataset)&(ext.grammar==grammar)&(ext.algorithm==method)]
                assert len(f)==10
                assert f[["optimum_found", "threshold_recall", "regret"]].notna().all().all()
                row.append(f"{100*f.optimum_found.mean():.0f}/{100*f.threshold_recall.mean():.0f}")
                summary.append(dict(dataset=dataset,grammar=grammar,algorithm=method,runs=len(f),
                    hit_rate=f.optimum_found.mean(),mean_recall=f.threshold_recall.mean(),mean_regret=f.regret.mean()))
            rows.append(row)
    table("revision_external_search.tex", "Additional datasets: exact-optimum hit rate / mean above-threshold recall (percent), ten search seeds per method/grammar at 4,000 requests. Each dataset has one supplied checkpoint; all seven methods are evaluated in both selected and full grammars.",
          "tab:additional-search", ["Dataset","Space","DPSO","GA","ACO","DE","ABC","SA","Random"],rows,"llrrrrrrr",True)
    atomic_csv(PAPER/"external_summary.csv",summary)
    confirmation=manuscript_rows(pd.read_csv(EXTERNAL/"external_confirmation.csv"))
    assert len(confirmation) == 4
    assert confirmation[["discovery_gap", "confirmation_gap", "lower", "upper"]].notna().all().all()
    assert (confirmation.support > 0).all()
    atomic_csv(PAPER/"external_confirmation.csv", confirmation.to_dict("records"))
    rows=[]
    for r in confirmation.itertuples():
        label=dict(DATASETS)[r.dataset]
        rows.append([label,"Sel." if r.grammar=="selected" else "Full",
            f"{r.discovery_gap:.4f}", f"{r.confirmation_gap:.4f}",int(r.support),
            f"[{r.lower:.3f}, {r.upper:.3f}]"])
    table("revision_external_confirmation.tex", "One discovery-fixed rule per additional dataset/grammar, confirmed on disjoint users. Bounds are pointwise 95\\% bounded-outcome intervals, conditional on the supplied checkpoint and fixed membership proportion; no familywise guarantee is claimed.",
          "tab:additional-confirmation", ["Dataset","Space",r"$\Delta_D$",r"$\Delta_C$","Support","Interval"],rows,"llrrrl",True)
    paths = [ADDED/"existing_protocol.json",EXTERNAL/"external_protocol.json",PAPER/"trace_validation.json",
             ADDED/"movielens_added_results.csv",ADDED/"synthetic_added_results.csv"]
    paths += list(EXTERNAL.glob("*.csv"))+[EXTERNAL/"external_datasets.json"]
    paths += list(PAPER.glob("*.csv"))+[PAPER/"external_datasets.json"]
    paths += [ROOT/"outputs/external_utilities_checked"/dataset/"metadata.json" for dataset in DATASET_IDS]
    paths += list((ADDED/"source_snapshots").glob("*.py"))
    paths += [ROOT/"export_external_utilities.py",Path(__file__),ROOT/"fairness_lab/revision.py"]
    atomic_json(OUT/"revision_evidence.json",dict(status="validated",synthetic_searches=945,
        datasets=["MovieLens-1M", "LFM360K", "Amazon"], dataset_scope="post hoc; full-grammar support feasibility",
        movielens_searches=420,additional_dataset_searches=280,additional_ablation_cases=24,
        input_source_note="Original multi-dataset records preserved; paper tables use the explicit dataset subset",
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print("Revised manuscript assets generated",flush=True)


if __name__ == "__main__":
    main()
