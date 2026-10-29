import unittest
import numpy as np
from fairness_lab.utility_support import utility_labels, bounded_radius
from fairness_lab.conference import enumerate_cells
from fairness_lab.subgroup_objectives import CategoricalUtilityGap


class UtilitySupportTests(unittest.TestCase):
    def test_outcome_labels_use_development_reference_and_margin(self):
        labels, gaps = utility_labels([0., .1, .4], .02)
        np.testing.assert_array_equal(labels, [1,1,0])
        np.testing.assert_allclose(gaps, [1/6,1/6-.1,1/6-.4])
        with self.assertRaises(ValueError):
            utility_labels([np.nan])

    def test_support_is_an_objective_constraint_and_does_not_expand_space(self):
        X = [[0]]*52 + [[1]]*150
        q = [0.]*52 + [.5]*150
        low = CategoricalUtilityGap(X,q,["age"],[(0,1)],50)
        high = CategoricalUtilityGap(X,q,["age"],[(0,1)],100)
        self.assertTrue(low.details([0])["valid"])
        self.assertIsNone(high.details([0])["disparity"])
        self.assertEqual(len(enumerate_cells(low)),len(enumerate_cells(high)))
        self.assertEqual(low.global_utility,high.global_utility)

    def test_bound_handles_global_subgroup_and_support_tradeoff(self):
        self.assertEqual(bounded_radius(200,200),0)
        self.assertLess(bounded_radius(100,4225),bounded_radius(25,4225))


if __name__ == "__main__":
    unittest.main()
