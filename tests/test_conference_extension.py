import itertools
import unittest

import numpy as np

from fairness_lab.conference import (Ledger, enumerate_cells, landscape,
                                     paired_bootstrap_gap, search)
from fairness_lab.subgroup_objectives import CategoricalUtilityGap


class ConferenceControlsTests(unittest.TestCase):
    def fixture(self):
        # Full cells include an unsupported combination. Projection hides the
        # worst intersection through aggregation, while keeping global users fixed.
        return CategoricalUtilityGap(
            [[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]],
            [0, 0, 1, 1, .5, .5, .5], ("a", "b"), ((0, 1), (0, 1)), min_support=2)

    def test_enumeration_and_projection_reveal_hidden_intersection(self):
        objective = self.fixture()
        full, projected = enumerate_cells(objective), enumerate_cells(objective.project(["a"]))
        self.assertEqual(len(full), 4)
        self.assertEqual(sum(r["valid"] for r in full), 3)
        self.assertAlmostEqual(max(r["disparity"] for r in full if r["valid"]), .5)
        self.assertAlmostEqual(max(r["disparity"] for r in projected if r["valid"]), 0)
        self.assertEqual(objective.global_utility, objective.project(["a"]).global_utility)

    def test_invalid_and_duplicate_requests_count_and_random_is_reproducible(self):
        objective = self.fixture()
        ledger = Ledger(objective, 3, .1, 1)
        self.assertIsNone(ledger.evaluate([1, 1]))
        self.assertIsNone(ledger.evaluate([1, 1]))
        self.assertTrue(ledger.logs[-1]["cached"])
        ledger.evaluate([0, 0])
        self.assertEqual(ledger.remaining, 0)
        a, _ = search(objective, "Random", 80, 4, .1, 101, None)
        b, _ = search(objective, "Random", 80, 4, .1, 101, None)
        self.assertEqual(a.logs, b.logs)
        self.assertEqual(len(a.logs), 80)
        self.assertEqual(len(a.cache), 4)
        self.assertAlmostEqual(a.best, .5)

    def test_encoding_preserves_semantic_scores_for_all_families(self):
        cards = (2, 4, 3) * 8
        identity = [np.arange(c) for c in cards]
        rng = np.random.default_rng(12)
        mapping = [rng.permutation(c) for c in cards]
        for family in ("additive", "nk_trap", "interaction"):
            original = landscape(family, 20, identity)
            permuted = landscape(family, 20, mapping)
            for encoded in rng.integers(0, cards, size=(20, 24)):
                semantic = [mapping[j][v] for j, v in enumerate(encoded)]
                self.assertEqual(original(semantic), permuted(encoded))

    def test_bootstrap_includes_subgroup_in_global_reference(self):
        # In an all-user subgroup the paired gap is always zero.
        self.assertEqual(paired_bootstrap_gap([0, .5, 1], [True]*3, repetitions=50), [0, 0])


if __name__ == "__main__":
    unittest.main()
