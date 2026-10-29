import itertools
import unittest

import numpy as np
import pandas as pd

from fairness_lab.conference import search
from fairness_lab.revision import IndexedUtilityGap, ablation_record
from fairness_lab.subgroup_objectives import CategoricalUtilityGap


class RevisionControlsTests(unittest.TestCase):
    def test_indexed_scores_match_user_weighted_reference_and_projection(self):
        frame = pd.DataFrame(dict(a=[0,0,0,1,1], b=[0,0,1,0,1]))
        utilities = [0,.5,1,.3,.8]
        ref = CategoricalUtilityGap.from_frame(frame, utilities, 2)
        fast = IndexedUtilityGap.from_frame(frame, utilities, 2)
        for names in (("a","b"), ("a",), ("b",)):
            r, f = ref.project(names), fast.project(names)
            for group in itertools.product(*(range(c) for c in f.space.cardinalities)):
                rd, fd = r.details(group), f.details(group)
                self.assertEqual(rd["support"], fd["support"])
                self.assertEqual(rd["valid"], fd["valid"])
                if rd["valid"]:
                    self.assertAlmostEqual(rd["disparity"], fd["disparity"])

    def test_added_heuristics_charge_invalid_and_repeated_requests(self):
        objective = IndexedUtilityGap([[0,0],[0,0],[1,1]], [0,.4,1],
            ("a","b"), ((0,1),(0,1)), 2)
        for method in ("ABC", "SA", "DE"):
            ledger, _ = search(objective, method, 100, 4, .1, 1001, None)
            self.assertEqual(len(ledger.logs), 100)
            self.assertTrue(any(not r["valid"] for r in ledger.logs))
            self.assertTrue(any(r["cached"] for r in ledger.logs))
            self.assertLessEqual(len(ledger.cache), 4)

    def test_projection_retention_counts_full_cells_not_distinct_parents(self):
        obj = IndexedUtilityGap([[0,0],[0,1],[1,0],[1,1]], [0,0,1,1],
                                ("a","b"), ((0,1),(0,1)), 1)
        row = ablation_record(obj, ["a"], .1)
        self.assertEqual(row["high_full_cells"], 2)
        self.assertEqual(row["retained_full_cells"], 2)
        self.assertEqual(row["parent_threshold_retention"], 1.)


if __name__ == "__main__":
    unittest.main()
