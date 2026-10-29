import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    from fairness_lab.publication import ExperimentVisualizer, latex_results_table


@unittest.skipUnless(HAS_MATPLOTLIB, "Optional Matplotlib dependency unavailable")
class PublicationTests(unittest.TestCase):
    def fixture(self, path, trace=True):
        rows, logs = [], []
        for name in ("DPSO (native)", "GA", "ACO"):
            for run in (0, 1):
                scores = [0.1, 0.2, 0.15, 0.4 + 0.01 * run]
                rows.append(dict(algorithm=name, run=run, search_seed=42,
                    maximum_disparity=max(scores), iterations_to_convergence=2,
                    subgroup_diversity=3 + run, execution_seconds=0.01, requests=4,
                    objective_sha256="same-fixed-objective"))
                logs.extend(dict(algorithm=name, run=run, query=j + 1, score=value)
                            for j, value in enumerate(scores))
        pd.DataFrame(rows).to_csv(path / "results.csv", index=False)
        (path / "config.json").write_text(json.dumps(dict(population=2, threshold=0.1)), encoding="utf-8")
        if trace:
            (path / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in logs), encoding="utf-8")
        return rows, logs

    def test_actual_trajectories_figures_aggregation_and_latex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            visualizer = ExperimentVisualizer([root / "results.csv"])
            curves = visualizer.load_curves()
            self.assertEqual(curves[0]["best"].tolist(), [0.1, 0.2, 0.2, 0.4])
            manifest = visualizer.generate(root / "paper")
            self.assertEqual(manifest["convergence"], "generated")
            for name in ("convergence.png", "convergence.svg", "subgroup_diversity.png", "subgroup_diversity.svg", "results_table.tex"):
                self.assertGreater((root / "paper" / name).stat().st_size, 100)
            data = pd.read_csv(root / "paper/convergence_data.csv")
            self.assertAlmostEqual(data[data.algorithm == "GA"].iloc[-1]["mean"], 0.405)
            self.assertIn("\\pm", (root / "paper/results_table.tex").read_text())

    def test_summary_only_never_fabricates_a_convergence_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, trace=False)
            manifest = ExperimentVisualizer([root]).generate(root / "paper")
            self.assertEqual(manifest["convergence"], "omitted_missing_trace")
            self.assertFalse((root / "paper/convergence.png").exists())
            self.assertTrue((root / "paper/subgroup_diversity.png").exists())

    def test_seed_labels_disambiguate_repeated_run_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, logs = self.fixture(root)
            for row in rows:
                row["search_seed"] = 42 + row["run"]
                row["run"] = 0
                row["long_tail_rate_absolute_disparity"] = 0.25
            for record in logs:
                record["search_seed"] = 42 + record["run"]
                record["run"] = 0
            pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
            (root / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in logs), encoding="utf-8")
            visualizer = ExperimentVisualizer(root / "results.csv")
            self.assertEqual(len(visualizer.load_curves()), 6)
            visualizer.generate(root / "paper")
            summary = pd.read_csv(root / "paper/expanded_bias_summary.csv", header=[0, 1], index_col=0)
            self.assertAlmostEqual(summary.loc["GA", ("long_tail_rate_absolute_disparity", "mean")], 0.25)

    def test_mixed_objectives_and_incomplete_traces_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, logs = self.fixture(root)
            rows[0]["objective_sha256"] = "other-model"
            pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
            with self.assertRaisesRegex(ValueError, "objective cohorts"):
                ExperimentVisualizer([root])
            self.fixture(root)
            (root / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in logs[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete trace"):
                ExperimentVisualizer([root]).load_curves()

    def test_single_run_latex_has_no_invented_sd_and_escapes_names(self):
        frame = pd.DataFrame([dict(algorithm="GA_test&more", maximum_disparity=0.2,
                    iterations_to_convergence=4, subgroup_diversity=2, execution_seconds=1)])
        table = latex_results_table(frame)
        self.assertIn(r"GA\_test\&more", table)
        self.assertNotIn(r"\pm", table)


if __name__ == "__main__":
    unittest.main()
