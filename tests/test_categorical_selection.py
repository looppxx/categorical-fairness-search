import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from fairness_lab.categorical_selection import CategoricalUnionBackwardSelector
from fairness_lab.features import mrmr_filter, chi_square_filter, union_top_features


class CategoricalSelectionTests(unittest.TestCase):
    def test_filters_independent_union_and_code_permutation(self):
        rng = np.random.default_rng(9)
        Z = rng.integers(0, 4, (300, 10))
        y = (Z[:, 2] == 1).astype(int)
        m, _ = mrmr_filter(Z, y, 3)
        c, _, _ = chi_square_filter(Z, y, 3)
        self.assertEqual(set(union_top_features(m, c, 3)), set(m) | set(c))
        self.assertEqual(m[0], 2)
        self.assertEqual(c[0], 2)
        relabeled = np.array([10, -5, 42, 100])[Z]
        self.assertEqual(m, mrmr_filter(relabeled, y, 3)[0])
        self.assertEqual(c, chi_square_filter(relabeled, y, 3)[0])

    def test_tiers_unknown_categories_schema_and_mandatory_feature(self):
        for p in (10, 20, 200):
            rng = np.random.default_rng(4)
            Z = rng.integers(0, 3, (120, p))
            y = (Z[:, 0] == 0).astype(int)
            X = pd.DataFrame(Z.astype(str), columns=[f"f{j}" for j in range(p)])
            estimator = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                       DecisionTreeClassifier(max_depth=3, random_state=42))
            selector = CategoricalUnionBackwardSelector(top_n=2, max_features=2,
                      mandatory=(1,), cv=2, estimator=estimator).fit(X, y)
            self.assertIn(1, selector.selected_)
            self.assertLessEqual(len(selector.selected_), 2)
            self.assertTrue(set(selector.selected_) <= set(selector.filter_union_))
            before = selector.selected_.copy()
            heldout = X.iloc[:3].copy()
            heldout.iloc[0, 0] = "unseen"
            self.assertEqual(selector.transform(heldout).shape, (3, len(before)))
            np.testing.assert_array_equal(before, selector.selected_)
            with self.assertRaises(ValueError):
                selector.transform(heldout[heldout.columns[::-1]])


if __name__ == "__main__":
    unittest.main()
