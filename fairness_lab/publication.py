"""Trace-backed paper assets. Summary endpoints never stand in for trajectories.

Supports compare_fairrec_search.py, run_live_fairness.py, and hybrid exports.
Plots are descriptive mean +/- sample SD, not significance or superiority tests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fairness-lab-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["maximum_disparity", "iterations_to_convergence", "subgroup_diversity", "execution_seconds"]


def algorithm_family(name):
    label = str(name).upper()
    if "PSO" in label:
        return "DPSO"
    for family in ("GA", "ACO", "DE", "ABC", "SA"):
        if label == family or label.endswith("-" + family) or label.endswith("_" + family) or label.startswith(family + "_"):
            return family
    return label


def latex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def latex_results_table(frame, caption="Search results: per-run endpoints, mean and sample standard deviation."):
    """Return booktabs LaTeX; missing values are --, one-run SD is not invented."""
    def value(series, digits):
        x = pd.to_numeric(series, errors="raise").dropna().to_numpy(float)
        if not len(x):
            return "--"
        if len(x) == 1:
            return f"${x.mean():.{digits}f}$"
        return f"${x.mean():.{digits}f} \\pm {x.std(ddof=1):.{digits}f}$"
    lines = [r"% Requires \usepackage{booktabs}", r"\begin{table*}[t]", r"\centering",
             "\\caption{" + latex_escape(caption) + "}", r"\label{tab:fairness-search}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Algorithm & Max disparity & Iterations$^{*}$ & Subgroup diversity & Time (s) \\", r"\midrule"]
    names = sorted(frame.algorithm.unique(), key=lambda name: ({"DPSO": 0, "GA": 1, "ACO": 2}.get(algorithm_family(name), 3), name))
    for name in names:
        group = frame[frame.algorithm == name]
        cells = [latex_escape(name)] + [value(group[key], digits) for key, digits in zip(METRICS, (4, 1, 1, 4))]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\par\smallskip\footnotesize $^{*}$First attainment of the final best in common evaluation blocks; retrospective proxy, not proven convergence.",
                  r"\end{table*}"])
    return "\n".join(lines) + "\n"


class ExperimentVisualizer:
    """Load homogeneous experiment results and their observed per-query traces.

    Algorithms retain exact labels: native/reference DPSO and FairAS-GA/plain GA
    are never pooled. Replicates are matched by objective hash, seed and run.
    Different objectives may be averaged only when every algorithm has the same
    objective cohorts. Summary-only input still produces tables and diversity.
    """
    def __init__(self, results_paths, algorithms=("DPSO", "GA", "ACO"),
                 x_axis="iterations", block_size=None, context=None):
        if x_axis not in {"iterations", "queries"} or (block_size is not None and block_size < 1):
            raise ValueError("Choose queries/iterations and a positive common block size")
        self.x_axis, self.block_size, self.context = x_axis, block_size, context
        self.notes, self.inputs, self.configs = [], [], []
        if isinstance(results_paths, (str, Path)):
            results_paths = [results_paths]
        paths = []
        for entry in results_paths:
            path = Path(entry)
            if path.is_dir():
                paths.extend([path / "results.csv"] if (path / "results.csv").exists()
                             else sorted(path.glob("*/results.csv")))
            else:
                paths.append(path)
        paths = list(dict.fromkeys(p.resolve() for p in paths))
        if not paths:
            raise ValueError("No results.csv files found")
        frames = []
        for index, path in enumerate(paths):
            frame = pd.read_csv(path)
            required = {"algorithm", *METRICS}
            if not required <= set(frame):
                raise ValueError(f"Missing result columns in {path}: {required - set(frame)}")
            frame["algorithm"] = frame.algorithm.astype(str)
            config = {}
            for name in ("config.json", "summary.json"):
                if (path.parent / name).exists():
                    config.update(json.loads((path.parent / name).read_text(encoding="utf-8")))
            self.configs.append(config)
            for column in METRICS:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
                if np.isinf(frame[column]).any():
                    raise ValueError(f"Infinite endpoint in {column}")
            if (frame[["subgroup_diversity", "iterations_to_convergence", "execution_seconds"]] < 0).any().any():
                raise ValueError("Counts and execution time cannot be negative")
            if "run" not in frame:
                frame["run"] = 0
            seed_column = "search_seed" if "search_seed" in frame else "seed" if "seed" in frame else None
            frame["_seed"] = frame[seed_column] if seed_column else config.get("seed", config.get("search_seed", 0))
            frame["_source"] = index
            if "objective_sha256" not in frame:
                frame["objective_sha256"] = config.get("objective_sha256", "unverified")
                self.notes.append(f"Objective matching metadata incomplete: {path}")
            frame["_cohort"] = frame.objective_sha256.fillna("unverified").astype(str)
            frame["_replicate"] = [repr((h, str(s), str(r))) for h, s, r in zip(frame._cohort, frame._seed, frame.run)]
            frames.append(frame)
            self.inputs.append(dict(results=path, trace=path.parent / "trace.jsonl"))
        combined = pd.concat(frames, ignore_index=True)
        if algorithms and "ALL" not in algorithms:
            requested = {a.upper() for a in algorithms}
            combined = combined[combined.algorithm.map(algorithm_family).isin(requested)].copy()
            for missing in sorted(requested - set(combined.algorithm.map(algorithm_family))):
                self.notes.append(f"No {missing} results supplied; no curve or bar was invented.")
        if combined.empty:
            raise ValueError("No requested algorithms found")
        request_column = "requests" if "requests" in combined else "queries" if "queries" in combined else None
        if request_column and combined[request_column].nunique() > 1:
            self.notes.append("Endpoint request counts differ; summary endpoints are not a matched-budget comparison.")
        if combined.duplicated(["algorithm", "_replicate"]).any():
            raise ValueError("Duplicate algorithm/cohort/seed/run rows; supply independent runs once")
        cohorts = [set(g._cohort) for _, g in combined.groupby("algorithm")]
        if any(group != cohorts[0] for group in cohorts[1:]):
            raise ValueError("Algorithms have different objective cohorts; generate separate paper assets")
        thresholds = {float(c["threshold"]) for c in self.configs if c.get("threshold") is not None}
        if len(thresholds) > 1:
            raise ValueError("Diversity thresholds differ between result sources")
        modes = {bool(c["live_inference"]) for c in self.configs if "live_inference" in c}
        if len(modes) > 1:
            raise ValueError("Cached and repeated-inference timings must not be pooled")
        populations = {int(c["population"]) for c in self.configs if c.get("population") is not None}
        if "population" in combined:
            populations |= set(combined.population.dropna().astype(int))
        if self.x_axis == "iterations" and self.block_size is None:
            if len(populations) != 1:
                raise ValueError("Iteration plots need one known common block size; use --block-size or --x-axis queries")
            self.block_size = populations.pop()
        self.frame = combined
        if self.context is None:
            if any("landscape_seed" in config for config in self.configs):
                self.context = "Synthetic benchmark"
            elif any((item["results"].parent / "demo_config.json").exists() for item in self.inputs):
                self.context = "Mock hybrid benchmark"
            elif any("positive_rating" in config for config in self.configs):
                epochs = [len(c.get("training_history", [])) for c in self.configs]
                suffix = f" ({epochs[0]} training epoch{'s' if epochs[0] != 1 else ''})" if epochs and len(set(epochs)) == 1 and epochs[0] else ""
                self.context = "MovieLens / NeuMF" + suffix
            else:
                self.context = "Experiment outputs"

    def load_curves(self):
        curves = []
        for index, item in enumerate(self.inputs):
            selected = self.frame[self.frame._source == index]
            if selected.empty:
                continue
            if not item["trace"].exists():
                self.notes.append(f"No trace at {item['trace']}; convergence plot requires actual trajectories.")
                continue
            records = [json.loads(line) for line in item["trace"].read_text(encoding="utf-8").splitlines() if line.strip()]
            for _, result in selected.iterrows():
                rows = [record for record in records
                        if str(record.get("algorithm", result.algorithm)) == result.algorithm
                        and str(record.get("run", result.run)) == str(result.run)
                        and str(record.get("search_seed", record.get("seed", result._seed))) == str(result._seed)
                        and str(record.get("objective_sha256", result._cohort)) == str(result._cohort)]
                same_run = selected[(selected.algorithm == result.algorithm) & (selected.run == result.run)]
                if len(same_run) > 1 and any("search_seed" not in r and "seed" not in r for r in rows):
                    raise ValueError("Repeated run labels across seeds need explicit trace seed labels")
                # Missing labels are safe only for a single result replicate.
                if rows and (len(selected) > 1 and any("algorithm" not in r or "run" not in r for r in rows)):
                    raise ValueError("A multi-run trace needs explicit algorithm and run labels")
                if not rows:
                    continue
                expected = result.get("requests", result.get("queries"))
                if expected is not None and pd.notna(expected) and len(rows) != int(expected):
                    raise ValueError("Incomplete trace: request count differs from results.csv")
                queries, raw = [], []
                for record in rows:
                    query = record.get("query", record.get("proposal"))
                    if query is None:
                        raise ValueError("Trace lacks query/proposal index")
                    score = record.get("score", record.get("disparity"))
                    if score is None and "objective_field" in record:
                        score = record.get(record["objective_field"])
                    if score is None:
                        score = record.get("best_disparity", record.get("best", record.get("best_objective")))
                    if not record.get("valid", True):
                        score = None
                    queries.append(int(query))
                    raw.append(np.nan if score is None else float(score))
                if queries != list(range(1, len(queries) + 1)):
                    raise ValueError("Complete per-request trace indices must be 1..N without gaps")
                best = pd.Series(raw, dtype=float).cummax().ffill().to_numpy()
                if np.isfinite(result.maximum_disparity):
                    if not np.isfinite(best[-1]) or not np.isclose(best[-1], result.maximum_disparity):
                        raise ValueError("Trace endpoint disagrees with results.csv maximum")
                curves.append(dict(algorithm=result.algorithm, replicate=result._replicate,
                                   queries=np.asarray(queries), best=best))
        # No selective subset of replicate curves: missing traces would bias means.
        if len(curves) != len(self.frame):
            self.notes.append("Convergence omitted because at least one selected replicate has no complete trace.")
            return []
        return curves

    @staticmethod
    def _save(fig, directory, stem):
        fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    def generate(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                 "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                 "svg.fonttype": "none", "axes.grid": True, "grid.alpha": 0.18}
        labels = sorted(self.frame.algorithm.unique(), key=lambda name: ({"DPSO": 0, "GA": 1, "ACO": 2}.get(algorithm_family(name), 3), name))
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
        family_colors = dict(zip(("DPSO", "GA", "ACO", "DE", "ABC", "SA"), palette))
        colors = {name: family_colors.get(algorithm_family(name), palette[j % len(palette)]) for j, name in enumerate(labels)}
        styles = ["-", "--", "-.", ":"]
        line_styles, family_counts = {}, {}
        for name in labels:
            family = algorithm_family(name)
            offset = {"DPSO": 0, "GA": 1, "ACO": 2}.get(family, 0)
            line_styles[name] = styles[(offset + family_counts.get(family, 0)) % len(styles)]
            family_counts[family] = family_counts.get(family, 0) + 1
        curves = self.load_curves()
        if not curves and any((directory / name).exists() for name in
                              ("convergence.png", "convergence.svg", "convergence_data.csv", "observed_plateaus.csv")):
            raise ValueError("Destination contains earlier convergence assets; use a fresh directory for summary-only input")
        curve_rows, plateau_rows = [], []
        with plt.rc_context(style):
            if curves:
                common_end = min(int(curve["queries"][-1]) for curve in curves)
                if len({int(curve["queries"][-1]) for curve in curves}) > 1:
                    self.notes.append(f"Convergence curves truncated to common observed budget {common_end}; no tail extrapolation.")
                grid = np.unique(np.r_[np.linspace(1, common_end, min(common_end, 500), dtype=int), common_end])
                improvements = [curve["queries"][np.r_[True,
                    (curve["best"][1:] > curve["best"][:-1]) |
                    (np.isfinite(curve["best"][1:]) & ~np.isfinite(curve["best"][:-1]))]] for curve in curves]
                grid = np.unique(np.r_[grid, *[points[points <= common_end] for points in improvements]])
                divisor = self.block_size if self.x_axis == "iterations" else 1
                fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
                for name in labels:
                    group = [curve for curve in curves if curve["algorithm"] == name]
                    values = []
                    for curve in group:
                        ids = np.searchsorted(curve["queries"], grid, side="right") - 1
                        aligned = np.where(ids >= 0, curve["best"][np.maximum(ids, 0)], np.nan)
                        values.append(aligned)
                        finite = np.flatnonzero(np.isfinite(curve["best"]))
                        if len(finite):
                            last_best = curve["best"][-1]
                            first_final = int(curve["queries"][np.flatnonzero(curve["best"] == last_best)[0]])
                            plateau_rows.append(dict(algorithm=name, replicate=curve["replicate"],
                                final_best=last_best, first_final_best_query=first_final,
                                trailing_requests_without_improvement=int(curve["queries"][-1]) - first_final))
                    matrix = np.asarray(values)
                    # All replicates must have a defined incumbent at a point.
                    defined = np.isfinite(matrix).all(axis=0)
                    mean = np.full(len(grid), np.nan)
                    std = np.full(len(grid), np.nan)
                    mean[defined] = matrix[:, defined].mean(axis=0)
                    std[defined] = matrix[:, defined].std(axis=0, ddof=1) if len(group) > 1 else 0
                    ax.step(grid / divisor, mean, where="post", color=colors[name], linestyle=line_styles[name], label=f"{name} (n={len(group)})", linewidth=1.8)
                    if len(group) > 1:
                        ax.fill_between(grid / divisor, mean - std, mean + std, step="post", color=colors[name], alpha=0.13)
                    curve_rows.extend(dict(algorithm=name, query=int(query), x=float(query / divisor),
                        mean=None if not np.isfinite(m) else float(m), sample_sd=None if not np.isfinite(s) else float(s), runs=len(group))
                        for query, m, s in zip(grid, mean, std))
                ax.set(xlabel=f"Evaluation blocks ({divisor} requests/block)" if self.x_axis == "iterations" else "Objective requests",
                       ylabel="Maximum disparity found", title=f"{self.context}: observed search progress\nMean trajectories; shading = sample SD where n > 1")
                ax.legend(fontsize=8, frameon=False, loc="best")
                self._save(fig, directory, "convergence")
            fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
            means, deviations, counts = [], [], []
            for name in labels:
                values = self.frame.loc[self.frame.algorithm == name, "subgroup_diversity"].dropna()
                means.append(values.mean() if len(values) else np.nan)
                deviations.append(values.std(ddof=1) if len(values) > 1 else 0)
                counts.append(len(values))
            ax.bar(np.arange(len(labels)), means,
                   color=[colors[name] for name in labels], width=0.62, alpha=0.9)
            repeated = [j for j, count in enumerate(counts) if count > 1]
            if repeated:
                ax.errorbar(repeated, np.asarray(means)[repeated], yerr=np.asarray(deviations)[repeated],
                            fmt="none", ecolor="black", capsize=4, linewidth=1.2)
            ax.set_xticks(np.arange(len(labels)), [f"{name}\n(n={count})" for name, count in zip(labels, counts)])
            ax.set(ylabel="Unique high-disparity subgroups", title=f"{self.context}: subgroup discovery\nBars = mean; error bars = sample SD where n > 1")
            ax.set_ylim(bottom=0)
            ax.grid(axis="x", visible=False)
            self._save(fig, directory, "subgroup_diversity")
        caption = f"{self.context}. Per-run endpoints: mean and sample standard deviation; single-run values have no error estimate."
        table = latex_results_table(self.frame, caption)
        (directory / "results_table.tex").write_text(table, encoding="utf-8")
        aggregated = self.frame.groupby("algorithm")[METRICS].agg(["count", "mean", "std"])
        aggregated.to_csv(directory / "aggregated_results.csv")
        # Expanded metrics are measured at the utility-selected best subgroup.
        # Never label these companion measurements as per-metric search maxima.
        bias_columns = [column for column in self.frame
                        if column.endswith(("_group_mean", "_global_mean", "_group_minus_global",
                                            "_absolute_disparity", "_disparity"))
                        and column != "maximum_disparity"]
        if bias_columns:
            self.frame.groupby("algorithm")[bias_columns].agg(["count", "mean", "std"]).to_csv(
                directory / "expanded_bias_summary.csv")
        if curve_rows:
            pd.DataFrame(curve_rows).to_csv(directory / "convergence_data.csv", index=False)
            pd.DataFrame(plateau_rows).to_csv(directory / "observed_plateaus.csv", index=False)
        hashes = []
        for item in self.inputs:
            for kind, path in item.items():
                if path.exists():
                    hashes.append(dict(kind=kind, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest = dict(context=self.context, algorithms=labels, input_hashes=hashes,
            convergence="generated" if curves else "omitted_missing_trace", x_axis=self.x_axis,
            block_size=self.block_size, error_bands="sample SD across observed replicate trajectories; no significance test",
            notes=list(dict.fromkeys(self.notes)),
            interpretation="Observed plateaus do not prove premature convergence; no superiority claim is generated.")
        (directory / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
