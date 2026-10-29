import unittest

import numpy as np

from fairness_lab.expanded_bias import ExpandedBiasEvaluator, ExpandedBiasObjective, flatten_bias_report
from fairness_lab.fairas_adapter import FairASSearchAdapter
from fairness_lab.subgroup_objectives import CategoricalUtilityGap


class ExpandedBiasTests(unittest.TestCase):
    def fixture(self):
        features = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        weights = 1 / np.log2(np.array([2, 3]))
        mixed = weights / weights.sum()
        evaluator = ExpandedBiasEvaluator([[0, 1], [0, 2]], features, [100, 50, 10, 0],
            [[0, 1], mixed], opportunity_profiles=[[0, 1], mixed])
        return evaluator

    def test_known_js_opportunity_and_ild_values(self):
        evaluator = self.fixture()
        np.testing.assert_allclose(evaluator.per_user["calibration_js"], [1, 0], atol=1e-12)
        np.testing.assert_allclose(evaluator.per_user["exposure_opportunity_tv"], [1, 0], atol=1e-12)
        np.testing.assert_allclose(evaluator.per_user["intra_list_diversity"], [0, 1], atol=1e-12)
        detail = evaluator.evaluate_members([0])
        self.assertAlmostEqual(detail["calibration_js"]["disparity"], 0.5)
        self.assertAlmostEqual(detail["intra_list_diversity"]["disparity"], 0.5)
        self.assertIsNone(detail["normalized_popularity"]["disparity"])
        self.assertGreater(detail["normalized_popularity"]["absolute_disparity"], 0)
        self.assertAlmostEqual(detail["long_tail_rate"]["absolute_disparity"], 0)
        for values in evaluator.per_user.values():
            self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_missing_history_short_slate_invalid_support_and_input_checks(self):
        evaluator = ExpandedBiasEvaluator([[0], [1]], np.eye(2), [0, 0], [[0, 0], [0, 1]])
        report = evaluator.evaluate_members([0])
        self.assertFalse(report["calibration_js"]["available"])
        self.assertIsNone(report["intra_list_diversity"]["disparity"])
        self.assertIsNone(evaluator.evaluate_members([0], min_support=2)["long_tail_rate"]["group_mean"])
        with self.assertRaises(ValueError):
            ExpandedBiasEvaluator([[0, 0]], np.eye(2), [1, 1], [[1, 0]])
        with self.assertRaises(ValueError):
            evaluator.evaluate_members([0, 0])

    def test_side_channels_leave_fitness_rng_and_query_sequence_unchanged(self):
        X = np.array([[0], [1]])
        base = CategoricalUtilityGap(X, [0.2, 0.8], ("group",), (("a", "b"),), 1)
        wrapped = ExpandedBiasObjective(base, self.fixture())
        self.assertEqual(base.fingerprint, wrapped.fingerprint)
        for vector in ([0], [1]):
            self.assertEqual(wrapped(vector), base(vector))
            self.assertIn("expanded_bias", wrapped.details(vector))
            self.assertEqual(wrapped.project(["group"])(vector), base(vector))
        for algorithm in ("GA", "ACO"):
            adapter = FairASSearchAdapter(algorithm, population=4, local_budget=9)
            plain = adapter.run(base, budget=11, seed=8)
            expanded = adapter.run(wrapped, budget=11, seed=8)
            self.assertEqual([r["candidate"] for r in plain.logs], [r["candidate"] for r in expanded.logs])
            self.assertEqual(plain.best, expanded.best)
            self.assertTrue(all("expanded_bias" in r for r in expanded.logs))

    def test_csv_report_missing_values_and_invalid_fractional_indices(self):
        evaluator = self.fixture()
        flat = flatten_bias_report(evaluator.evaluate_members([0]))
        self.assertAlmostEqual(flat["intra_list_diversity_absolute_disparity"], 0.5)
        self.assertTrue(flat["calibration_js_available"])
        self.assertTrue(all(value is None for value in flatten_bias_report().values()))
        self.assertFalse(evaluator.evaluate_members([])["calibration_js"]["available"])
        for members in ([0.9], [True], ["0"]):
            with self.assertRaises(ValueError):
                evaluator.evaluate_members(members)
        with self.assertRaises(ValueError):
            evaluator.evaluate_members([0], min_support=1.5)


if __name__ == "__main__":
    unittest.main()
