import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from fairness_lab.core import FixedOracle, SubgroupSpace, EvaluationSession, BudgetExhausted
from fairness_lab.features import MultiFilter, FilterConfig, HybridFeatureSelector, TrainingBinner
from fairness_lab.metrics import audit_groups, exposure_opportunity_gap, counterfactual_rank_instability
from fairness_lab.native import (ExactCellSpace, NativeCellOracle, NativeFairRecSearch,
                                 NativeFairASSearch, DualEndedSearch, source_definitions, seeded_legacy_rng)
from fairness_lab.search import (GeneticSearch, AntColonySearch, DiscretePSOReference,
                                 SimulatedAnnealing, TabuSearch, HybridTestingFramework, RandomSearch, GreedyLocalSearch)
from fairness_lab.statistics import paired_comparison, holm_adjust


class ResearchControls(unittest.TestCase):
    def setUp(self):
        self.space = SubgroupSpace((2, 2), 2)
        X = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (30, 1))
        u = np.tile([0.2, 0.8, 0.8, 0.8], 30)
        self.oracle = FixedOracle(X, u, self.space, 10)

    def test_budget_repeats_and_json(self):
        session = EvaluationSession(self.oracle, 2)
        session.evaluate((0, 0))
        session.evaluate((0, 0))
        self.assertEqual(len(session.cache), 1)
        self.assertTrue(session.logs[-1]["cached"])
        with self.assertRaises(BudgetExhausted):
            session.evaluate((1, 1))
        with tempfile.TemporaryDirectory() as d:
            session.write_jsonl(Path(d) / "trace.jsonl")
            self.assertEqual(len([json.loads(s) for s in (Path(d) / "trace.jsonl").read_text().splitlines()]), 2)

    def test_unique_budget_has_separate_request_cap(self):
        session = EvaluationSession(self.oracle, 50, unique_budget=2)
        session.evaluate((0, 0))
        session.evaluate((0, 0))
        self.assertGreater(session.remaining, 0)
        session.evaluate((1, 1))
        self.assertEqual(session.remaining, 0)
        self.assertEqual(len(session.logs), 3)
        with self.assertRaises(BudgetExhausted):
            session.evaluate((0, 1))

    def test_all_engines_respect_budget_and_fixed_oracle(self):
        fingerprint = self.oracle.fingerprint
        for engine in [RandomSearch(), GeneticSearch(6), AntColonySearch(6), DiscretePSOReference(6),
                       SimulatedAnnealing(), TabuSearch(), HybridTestingFramework(global_search=GeneticSearch(6))]:
            session = EvaluationSession(self.oracle, 31)
            engine.run(session, np.random.default_rng(4))
            self.assertEqual(len(session.logs), 31, engine.name)
            self.assertEqual(self.oracle.fingerprint, fingerprint)
            curve = [r["best_objective"] for r in session.logs if r["best_objective"] is not None]
            self.assertTrue(np.all(np.diff(curve) >= 0))

    def test_infeasible_and_category_validation(self):
        with self.assertRaises(ValueError):
            self.space.validate([0.5, 1])
        sparse = FixedOracle([[0, 0], [1, 1]], [0.0, 1.0], self.space, 2)
        session = EvaluationSession(sparse, 1)
        self.assertEqual(session.evaluate([0, 0]), -np.inf)
        self.assertIsNone(session.logs[0]["best_objective"])

    def test_entropy_threshold_and_mandatory_rare_feature(self):
        X = np.column_stack([np.zeros(100), np.r_[1, np.zeros(99)], np.arange(100) % 2])
        y = np.arange(100) % 2
        filt = MultiFilter(FilterConfig(entropy_threshold=0.2, top_k=1, categorical=(0, 1, 2), mandatory=(1,))).fit(X, y)
        self.assertNotIn(0, filt.selected_)
        self.assertIn(1, filt.selected_)
        self.assertIn(2, filt.selected_)

    def test_binner_uses_training_edges(self):
        b = TrainingBinner(bins=4).fit(np.arange(100).reshape(-1, 1))
        edges = b.rules_[0][1].copy()
        b.transform([[1e9], [-1e9]])
        np.testing.assert_array_equal(edges, b.rules_[0][1])

    def test_wrapper_tiers_and_heldout_schema(self):
        for p in (10, 20, 200):
            X, y = make_classification(n_samples=100, n_features=p, n_informative=3, random_state=42)
            X[:, -1] = 0
            for direction in ("forward", "backward"):
                s = HybridFeatureSelector(DecisionTreeClassifier(max_depth=2, random_state=42),
                                          FilterConfig(top_k=2), direction, max_features=2, cv=2).fit(X[:80], y[:80])
                before = s.selected_.copy()
                self.assertLessEqual(s.transform(X[80:]).shape[1], 2)
                self.assertNotIn(p - 1, before)
                np.testing.assert_array_equal(before, s.selected_)

    def test_intersection_masking_and_rank_sensitivity(self):
        rows = audit_groups(self.oracle, [(0, 0)], threshold=0.45)
        self.assertTrue(rows[0]["masked_disadvantage"])
        self.assertGreater(rows[0]["intersection_amplification"], 0)
        np.testing.assert_allclose(counterfactual_rank_instability([[1, 2]], [[1, 2]]), [0])
        np.testing.assert_allclose(counterfactual_rank_instability([[1, 2]], [[3, 4]]), [1])
        result = exposure_opportunity_gap([[0, 1]], ["a", "b"], {"a": 0.5, "b": 0.5})
        self.assertAlmostEqual(sum(result.values()), 0)

    def test_statistics(self):
        r = paired_comparison([1, 2, 3], [1, 2, 3], bootstrap_samples=50)
        self.assertEqual(r["wilcoxon_p"], 1)
        self.assertEqual(r["mean_paired_difference"], 0)
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_hybrid_unique_budget_shared_by_stages(self):
        session = EvaluationSession(self.oracle, 1000, unique_budget=4)
        HybridTestingFramework(global_search=GeneticSearch(4)).run(session, np.random.default_rng(8))
        self.assertLessEqual(len(session.cache), 4)
        self.assertLessEqual(len(session.logs), 1000)
        self.assertEqual(session.unique_budget, 4)
        self.assertEqual(session.budget, 1000)


class NativeIntegration(unittest.TestCase):
    repo = Path("C:/Users/DELL/Documents/repo")

    @unittest.skipUnless((repo / "FairRec/fairrec.py").exists(), "Local native source unavailable")
    def test_native_dpso_query_sequence_preserved(self):
        import random
        space = ExactCellSpace((2, 6, 8))
        scorer = lambda g: (0.1 + sum(g) / 30, 50)
        oracle = NativeCellOracle(space, scorer, "test-native")
        session = EvaluationSession(oracle, 60, objective_field="utility")
        NativeFairRecSearch(self.repo / "FairRec/fairrec.py", population=4).run(session, np.random.default_rng(7))
        ns = {"np": np, "random": random}
        source_definitions(self.repo / "FairRec/fairrec.py", {"FairRec"}, ns)
        cls, sequence = ns["FairRec"], []
        def fitness(obj, g):
            if len(sequence) == 60:
                raise BudgetExhausted()
            sequence.append(list(map(int, g)))
            return scorer(g)[0]
        cls.fitness = fitness
        params = {"dimension": 3, "iter": 60, "size": 4, "bound_low": [0, 0, 0],
                  "bound_up": [1, 5, 7], "v_low": -2, "v_high": 2,
                  "value_prob_list": [[list(range(c)), [1/c]*c] for c in space.cardinalities]}
        with seeded_legacy_rng(int(np.random.default_rng(7).integers(2**31 - 1))):
            try:
                cls(params, {}, {}).double_ended_dpso()
            except BudgetExhausted:
                pass
        self.assertEqual(sequence, [r["candidate"] for r in session.logs])

    @unittest.skipUnless((repo / "FairAS/run_s.py").exists(), "Local native source unavailable")
    def test_native_fairas_and_local_replacements(self):
        space = ExactCellSpace((2, 6, 8))
        oracle = NativeCellOracle(space, lambda g: (0.3 + sum(g) / 30, 50), "test-fairas")
        for local in (None, GreedyLocalSearch(), SimulatedAnnealing(), TabuSearch(), GeneticSearch(4), AntColonySearch(4)):
            engine = NativeFairASSearch(self.repo / "FairAS/run_s.py", ("gender", "age", "country"), local)
            session = EvaluationSession(oracle, 60, objective_field="utility")
            engine.run(session, np.random.default_rng(3))
            self.assertEqual(len(session.logs), 60, engine.name)
            self.assertTrue(all(r["valid"] for r in session.logs))
            self.assertIn("source_sha256", session.logs[0])

    @unittest.skipUnless((repo / "FairRec/rec_metrics.py").exists(), "Local metrics unavailable")
    def test_native_prediction_scorer_and_cli(self):
        import pandas as pd
        import subprocess
        import sys
        from sklearn.metrics import roc_auc_score
        from fairness_lab.predictions import PredictionTableScorer
        rows = []
        for gender in range(2):
            for age in range(3):
                for country in range(3):
                    for user in range(4):
                        uid = len(rows)
                        for item in range(5):
                            label = int(item in (0, 2))
                            rows.append(dict(user_id=uid, item_id=item, label=label,
                                             predict=(0.8 if item == 0 else 0.2 + item / 10),
                                             gender=gender, age=age, country=country,
                                             cate=item % 2, popularity=0.5, user_pop=0.5))
        frame = pd.DataFrame(rows)
        scorer = PredictionTableScorer(frame, self.repo / "FairRec/rec_metrics.py",
                                       ("gender", "age", "country"), "auc", threshold_users=2)
        group = frame[(frame.gender == 0) & (frame.age == 0) & (frame.country == 0)]
        score, n = scorer((0, 0, 0))
        self.assertAlmostEqual(score, roc_auc_score(group.label, group.predict))
        self.assertEqual(n, 4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            frame.to_csv(path / "predictions.csv", index=False)
            for baseline in ("FairRec", "FairAS"):
                result = subprocess.run([sys.executable, "-m", "fairness_lab.run_predictions",
                    "--predictions", str(path / "predictions.csv"), "--repo", str(self.repo),
                    "--baseline", baseline, "--features", "gender", "age", "country",
                    "--cardinalities", "2", "3", "3", "--metric", "auc", "--support", "2",
                    "--seeds", "1", "--unique-budget", "10", "--request-cap", "100", "--population", "4",
                    "--out", str(path / baseline)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = pd.read_csv(path / baseline / "summary.csv")
                self.assertTrue((summary.unique_queries <= 10).all())
                self.assertTrue((summary.requests <= 100).all())


if __name__ == "__main__":
    unittest.main()
