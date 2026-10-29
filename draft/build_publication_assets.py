"""Build manuscript assets from recorded evidence; never launch new experiments.

Run from the project root with .venv/Scripts/python.exe draft/build_publication_assets.py.
All numerical table entries are read from the original CSV/JSONL artifacts.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from fairness_lab.publication import ExperimentVisualizer


def table(path, caption, label, headings, rows, specification):
    text = [r"\begin{table}[t]", r"\centering", "\\caption{" + caption + "}",
            "\\label{" + label + "}", r"\small", "\\begin{tabular}{" + specification + "}",
            r"\toprule", " & ".join(headings) + r" \\", r"\midrule"]
    text.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    text.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    synthetic = ROOT / "outputs/search_comparison_native"
    live = ROOT / "outputs/final_benchmark"
    out = ROOT / "outputs/paper_assets"
    out.mkdir(parents=True, exist_ok=True)
    ExperimentVisualizer(synthetic, algorithms=("DPSO", "GA", "ACO", "DE"),
        context="Synthetic categorical benchmark").generate(out)
    # Explicitly identify the synthetic score; this is not a measured population disparity.
    path = out / "results_table.tex"
    text = path.read_text(encoding="utf-8").replace("Max disparity", "Max synthetic score")
    text = text.replace("Iterations$^{*}$", "Blocks$^{*}$")
    text = text.replace(
        "Per-run endpoints: mean and sample standard deviation; single-run values have no error estimate.",
        "Mean and sample standard deviation across three runs per method. "
        "Subgroup diversity counts distinct candidates with synthetic score above 0.65.")
    text = text.replace(r"\label{tab:fairness-search}", r"\label{tab:synthetic}")
    path.write_text(text, encoding="utf-8")
    frame = pd.read_csv(live / "final_benchmark_results.csv")
    frame = frame.loc[frame.status == "success"]
    rows = []
    for name in ("DPSO (native)", "GA", "ACO", "DE"):
        group = frame.loc[frame.algorithm == name]
        rows.append(["DPSO" if name == "DPSO (native)" else name, f"{group.maximum_disparity.mean():.5f}", f"{group.unique_groups.mean():.2f}",
                     int(group.invalid_requests.sum()), f"{group.subgroup_diversity.mean():.2f}", f"{group.execution_seconds.mean():.4f}"])
    table(out / "neumf_results.tex", "MovieLens/NeuMF: means across three paired model/search seeds, 4,000 requests per method and seed. DPSO uses the native backend. Invalid is the total invalid-request count; time excludes training and objective setup.",
          "tab:neumf", ["Method", r"$\Delta_{\mathrm{NDCG}}$", "Unique", "Invalid", r"$D_{.02}$", "Time (s)"], rows, "lrrrrr")
    selected = frame.loc[frame.algorithm == "GA"]
    confirmations = pd.concat([pd.read_csv(live / f"seed_{seed}/test_confirmation.csv")
                               for seed in (42, 100, 2026)], ignore_index=True)
    confirmations = confirmations.loc[confirmations.algorithm == "GA"]
    confirmed = [json.loads(value) for value in confirmations.expanded_bias]
    names = {"calibration_js": "Calibration (JSD)", "exposure_opportunity_tv": "Exposure (TV)",
             "intra_list_diversity": "Intra-list diversity"}
    rows = []
    for name, title in names.items():
        for split in ("Val.", "Test"):
            metric = {key: (selected[f"{name}_{key}"].mean() if split == "Val." else
                           sum(report[name][key] for report in confirmed) / len(confirmed))
                      for key in ("group_mean", "global_mean", "group_minus_global")}
            rows.append([title if split == "Val." else "", split,
                         f"{metric['group_mean']:.5f}", f"{metric['global_mean']:.5f}",
                         f"{metric['group_minus_global']:+.5f}"])
    table(out / "bias_audit.tex", "Companion measurements for GA-selected subgroups, averaged equally across three model seeds. Gaps are group minus global; a negative diversity gap denotes lower diversity.",
          "tab:audit", ["Measurement", "Split", "Group", "Global", "Gap"], rows, "llrrr")
    inputs = [synthetic / "results.csv", synthetic / "config.json", synthetic / "trace.jsonl",
              live / "final_benchmark_results.csv", live / "config.json", live / "benchmark_summary.json"]
    inputs += [live / f"seed_{seed}" / name for seed in (42, 100, 2026)
               for name in ("test_confirmation.csv", "feature_selection.json", "training_summary.json", "audit_protocol.json")]
    inputs += list(live.glob("seed_*/search/*/trace.jsonl"))
    inputs.append(Path(__file__))
    inputs += [ROOT / path for path in (
        "outputs/hybrid_expanded/GA/summary.json", "outputs/hybrid_expanded/GA/demo_config.json",
        "outputs/hybrid_mock_200/GA/summary.json", "outputs/hybrid_mock_200/GA/demo_config.json",
        "outputs/hybrid_mock_200/GA/feature_selection.json", "compare_fairrec_search.py",
        "run_hybrid_fairness.py", "fairness_lab/recommender.py", "fairness_lab/expanded_bias.py")]
    evidence = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    config = json.loads((live / "config.json").read_text(encoding="utf-8"))
    (out / "manuscript_evidence.json").write_text(json.dumps(dict(input_sha256=evidence,
        synthetic_runs=3, live_runs_per_method=3, live_training_epoch_cap=config["epochs"], live_dpso_available=True,
        live_training_epochs=selected.training_epochs_completed.astype(int).tolist(),
        selected_features=selected.selected_features.tolist()), indent=2), encoding="utf-8")
    bundled = ROOT / "draft/assets"
    bundled.mkdir(parents=True, exist_ok=True)
    for name in ("results_table.tex", "neumf_results.tex", "bias_audit.tex", "convergence.png",
                 "subgroup_diversity.png", "manuscript_evidence.json"):
        shutil.copyfile(out / name, bundled / name)
    print("Publication assets:", out)


if __name__ == "__main__":
    main()
