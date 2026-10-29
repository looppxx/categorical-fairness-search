import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from fairness_lab.recommender import (MovieLensData, NeuMF, RecommenderDisparityObjective,
        load_movielens_1m, ranking_utilities, train_neumf)
    from run_live_fairness import benchmark


def write_movielens_fixture(path):
    users, ratings = [], []
    for u in range(12):
        users.append(f"{u+1}::{'M' if u%2 else 'F'}::{[18,25,35][(u//2)%3]}::{u//6}::01234")
        for i in range(6):
            ratings.append(f"{u+1}::{(u+i)%17+1}::5::{100+i}")
    (path / "users.dat").write_text("\n".join(users), encoding="latin-1")
    (path / "ratings.dat").write_text("\n".join(ratings), encoding="latin-1")
    (path / "movies.dat").write_text("\n".join(f"{i}::Movie {i}::{'Action|Comedy' if i%2 else 'Drama'}"
                                        for i in range(1, 18)), encoding="latin-1")


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch dependency not installed")
class LiveRecommenderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_ranks_signed_global_gap_onehot_support_and_live_equivalence(self):
        class IncreasingItemScore(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(1))
                self.config = {"model": "known-ranking"}
            def forward(self, users, items):
                return items.float() + self.bias
        data = MovieLensData(np.array([10, 20]), np.arange(6), np.array([[0], [1]]),
                ("gender",), (("F", "M"),), np.array([[0, 0], [0, 1], [1, 0]]),
                (np.array([0, 1]), np.array([0])),
                (np.array([0, 1, 4, 5]), np.array([0, 2, 3])),
                np.array([4, 2]), np.array([5, 3]), "fixture")
        model = IncreasingItemScore()
        scores = ranking_utilities(model, data, k=2)
        np.testing.assert_allclose(scores["ndcg"], [1 / np.log2(3), 0])
        np.testing.assert_array_equal(scores["recall"], [1, 0])
        slates = ranking_utilities(model, data, k=2, return_recommendations=True)["recommendations"]
        np.testing.assert_array_equal(slates, [[5, 4], [5, 4]])
        from fairness_lab.expanded_bias import ExpandedBiasEvaluator
        evaluator = ExpandedBiasEvaluator.from_neumf(
            RecommenderDisparityObjective(model, data, k=2, min_support=1), np.eye(6))
        np.testing.assert_array_equal(evaluator.per_user["normalized_popularity"], [0, 0])
        np.testing.assert_array_equal(ranking_utilities(model, data, split="test", k=2)["recall"], [1, 0])
        cached = RecommenderDisparityObjective(model, data, k=2, min_support=1)
        live = RecommenderDisparityObjective(model, data, k=2, min_support=1, cache_utilities=False)
        self.assertAlmostEqual(cached([1]), 0.5 / np.log2(3))
        self.assertLess(cached([0]), 0)
        for vector in ([0], [1]):
            self.assertAlmostEqual(cached(vector), live(vector))
        self.assertEqual(cached.decode([0, 1], one_hot=True), (1,))
        self.assertEqual(cached.describe([1]), {"gender": "M"})
        with self.assertRaises(ValueError):
            cached.decode([1, 1], one_hot=True)
        unsupported = RecommenderDisparityObjective(model, data, k=2, min_support=2)
        self.assertIsNone(unsupported([0]))
        rows, traces = benchmark(unsupported, budget=11, population=4, runs=1)
        self.assertTrue(all(r["maximum_disparity"] is None for r in rows))
        self.assertTrue(all(r["invalid_requests"] == 11 for r in rows))
        self.assertTrue(all(r["best"] is None for r in traces))
        # Caller-side training cannot alter the objective snapshot.
        self.assertFalse(any(p.requires_grad for p in cached.model.parameters()))
        self.assertTrue(model.bias.requires_grad)

    def test_training_splits_search_budget_and_checkpoint_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_movielens_fixture(root)
            data = load_movielens_1m(root)
            for u, train in enumerate(data.train_seen):
                self.assertNotIn(data.validation_items[u], train)
                self.assertNotIn(data.test_items[u], train)
                self.assertNotEqual(data.validation_items[u], data.test_items[u])
            torch.manual_seed(4)
            before = NeuMF(len(data.user_ids), len(data.item_ids), embedding_dim=4)
            model, history = train_neumf(data, epochs=1, negatives=1, seed=4, embedding_dim=4)
            self.assertTrue(np.isfinite(history[0]["mean_bce"]))
            self.assertFalse(torch.equal(before.output.weight, model.output.weight))
            objective = RecommenderDisparityObjective(model, data, k=3, min_support=1)
            projected = objective.project(["gender"])
            live_projected = RecommenderDisparityObjective(model, data, k=3, min_support=1,
                                      cache_utilities=False).project(["gender"])
            for value in range(len(data.categories[0])):
                members = data.demographics[:, 0] == value
                self.assertEqual(projected.details([value])["support"], int(members.sum()))
                self.assertAlmostEqual(projected([value]), objective.global_utility - objective.utilities[members].mean())
                self.assertAlmostEqual(projected([value]), live_projected([value]), places=7)
            from fairness_lab.fairas_adapter import FairASSearchAdapter
            live_adapter = FairASSearchAdapter("GA", population=4, local_budget=7)
            ledger = live_adapter.run(objective, budget=11)
            self.assertEqual(len(ledger.logs), 11)
            self.assertEqual(ledger.objective.feature_names, data.feature_names)
            rows, trace = benchmark(objective, budget=41, population=4, runs=1, threshold=0)
            self.assertEqual(len(rows), 3)
            for row in rows:
                logs = [r for r in trace if r["algorithm"] == row["algorithm"]]
                self.assertEqual(row["requests"], 41)
                self.assertEqual(row["maximum_disparity"], max(r["disparity"] for r in logs))
                for log in logs:
                    self.assertAlmostEqual(log["disparity"], log["global_utility"] - log["subgroup_utility"])
            # Exercise the published command, checkpoint round-trip and outputs.
            base = [sys.executable, "run_live_fairness.py", "--data", str(root), "--epochs", "1",
                    "--negatives", "1", "--population", "4", "--budget", "11", "--runs", "1",
                    "--min-support", "1", "--k", "3", "--threads", "1"]
            first = subprocess.run(base + ["--out", str(root / "first"), "--expanded-bias"], capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(base + ["--checkpoint", str(root / "first/neumf.pt"),
                    "--out", str(root / "second"), "--live-inference"], capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            config = json.loads((root / "second/config.json").read_text())
            self.assertEqual(config["data_sha256"], data.source_sha256)
            self.assertTrue((root / "second/test_confirmation.csv").exists())


if __name__ == "__main__":
    unittest.main()
