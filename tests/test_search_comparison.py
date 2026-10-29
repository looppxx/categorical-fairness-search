"""Checks for the standalone synthetic search comparison."""
import itertools
import unittest

import numpy as np

from compare_fairrec_search import (MockDisparity, Space, Tracker, compare,
                                    evaluate_subgroup_disparity)


class SearchComparisonTests(unittest.TestCase):
    def test_fixed_objective_and_domain(self):
        objective = MockDisparity.create(Space((2, 3, 4)))
        clone = MockDisparity.create(Space((2, 3, 4)))
        fingerprint = objective.fingerprint
        for g in itertools.product(range(2), range(3), range(4)):
            self.assertEqual(objective(g), clone(g))
            self.assertTrue(0 <= objective(g) <= 1)
        self.assertEqual(objective.fingerprint, fingerprint)
        for invalid in ([0.5, 0, 0], [-1, 0, 0], [0, 3, 0], [0, 0]):
            with self.assertRaises(ValueError):
                objective(invalid)
        self.assertEqual(evaluate_subgroup_disparity([0] * 24),
                         evaluate_subgroup_disparity([0] * 24))

    def test_repeats_are_charged_but_not_counted_twice_in_diversity(self):
        objective = MockDisparity.create(Space((2, 2)))
        tracker = Tracker(objective, 3, 0, 4)
        for _ in range(3):
            tracker.evaluate([0, 0])
        self.assertEqual(tracker.calls, 3)
        self.assertEqual(len(tracker.high), 1)
        self.assertEqual(len(tracker.cache), 1)
        self.assertTrue(tracker.trace[-1]["cached"])

    def test_singleton_categories_are_valid_without_mutation_failures(self):
        objective = MockDisparity.create(Space((1, 1)))
        rows, traces = compare(objective, budget=31, population=4, runs=1, threshold=0)
        self.assertTrue(all(r["queries"] == 31 for r in rows))
        self.assertTrue(all(r["subgroup_diversity"] == 1 for r in rows))
        self.assertTrue(all(tuple(r["subgroup"]) == (0, 0) for r in traces))

    def test_every_engine_equal_budget_archive_and_reproducibility(self):
        # Binary + nominal categories, an odd budget, and a final partial batch.
        objective = MockDisparity.create(Space((2, 3, 4)))
        known_max = max(objective(g) for g in itertools.product(range(2), range(3), range(4)))
        for budget in (8, 9, 101):
            rows, traces = compare(objective, budget, population=4, runs=2, threshold=0)
            again, _ = compare(objective, budget, population=4, runs=2, threshold=0)
            self.assertEqual(len(rows), 12)
            for a, b in zip(rows, again):
                self.assertEqual({k: v for k, v in a.items() if k != "execution_seconds"},
                                 {k: v for k, v in b.items() if k != "execution_seconds"})
                self.assertEqual(a["queries"], budget)
                self.assertLessEqual(a["maximum_disparity"], known_max)
                log = [r for r in traces if r["algorithm"] == a["algorithm"] and r["run"] == a["run"]]
                unique = {tuple(r["subgroup"]) for r in log}
                self.assertEqual(a["subgroup_diversity"], len(unique))
                self.assertEqual(a["maximum_disparity"], max(r["score"] for r in log))
                self.assertEqual(a["best_found_query"], next(r["query"] for r in log
                                 if r["score"] == a["maximum_disparity"]))
                self.assertTrue(np.all(np.diff([r["best"] for r in log]) >= 0))


if __name__ == "__main__":
    unittest.main()
