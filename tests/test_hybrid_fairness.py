import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from fairness_lab.categorical_selection import CategoricalUnionBackwardSelector
from fairness_lab.fairas_adapter import FairASLocalSearchAdapter, FairASSearchAdapter, SearchLedger
from fairness_lab.hybrid import HybridFairnessTester
from fairness_lab.subgroup_objectives import CategoricalUtilityGap, CallableSubgroupObjective
from run_hybrid_fairness import mock_users


class HybridTests(unittest.TestCase):
    def objective(self, cardinalities=(3, 3, 3, 3)):
        X = np.array(list(itertools.product(*(range(c) for c in cardinalities))))
        utility = 0.8 - 0.4 * (X[:, 0] == 0) - 0.1 * (X[:, 1] == 1)
        return CategoricalUtilityGap(X, utility, tuple(f"f{i}" for i in range(X.shape[1])),
                                    tuple(tuple(range(c)) for c in cardinalities), min_support=1)

    def test_projection_aggregates_dropped_attributes_and_preserves_reference(self):
        full = self.objective((2, 3))
        projected = full.project(["f0"])
        detail = projected.details([0])
        self.assertEqual(detail["support"], 3)
        self.assertEqual(projected.global_utility, full.global_utility)
        self.assertAlmostEqual(projected([0]), full.global_utility - full.utilities[:3].mean())
        self.assertNotAlmostEqual(projected([0]), full([0, 1]))
        self.assertEqual(projected.evaluate_subgroup_disparity([1, 0], one_hot=True), projected([0]))
        with self.assertRaises(ValueError):
            full.project(["missing"])

    def test_callable_contract_converts_one_hot_and_preserves_invalidity(self):
        received = []
        def objective(bits):
            received.append(bits.copy())
            return None if bits[0] else 0.2
        wrapped = CallableSubgroupObjective(objective, (2, 3), ("gender", "age"), "fixed-test-v1", "one_hot")
        self.assertIsNone(wrapped([0, 2]))
        self.assertEqual(wrapped([1, 0]), 0.2)
        np.testing.assert_array_equal(received[0], [1, 0, 0, 0, 1])

    def test_live_provider_is_not_hidden_by_result_caching(self):
        base = self.objective()
        forward_passes = []
        def score_users(members):
            forward_passes.append(members.copy())
            return base.utilities[members]
        live = CategoricalUtilityGap(base.demographics, base.utilities, base.feature_names,
                    base.categories, min_support=1, score_users=score_users)
        projected = live.project(["f0", "f1"])
        ledger = FairASSearchAdapter("GA", population=4, local_budget=20).run(projected, budget=11)
        self.assertEqual(len(forward_passes), len(ledger.logs))
        self.assertTrue(all(not row["cached"] for row in ledger.logs))

    def test_hook_signature_directions_shared_cap_radius_and_frozen_attribute(self):
        objective = self.objective()
        info = {"sensitive_features": list(objective.feature_names)}
        dimensions = dict(zip(objective.feature_names, objective.space.cardinalities))
        seed = [1, 0, 0, 0]
        for algorithm in ("GA", "ACO"):
            for direction in ("min", "max"):
                ledger = SearchLedger(objective, budget=17)
                incumbent = ledger.evaluate(seed)
                adapter = FairASLocalSearchAdapter(objective, algorithm, population=6,
                    local_budget=30, radius=1, ledger=ledger, frozen_features=("f2",), key_format="tuple")
                users, scores, keys = [seed.copy()], [incumbent], [tuple(seed)]
                group, score = adapter.local_search(info, dimensions, seed, incumbent, direction, keys, users, scores)
                self.assertEqual(len(ledger.logs), 17)
                self.assertEqual(len({tuple(g) for g in users}), len(users))
                self.assertEqual(len(users), len(scores))
                self.assertTrue(all(r["hamming_distance"] <= 1 for r in ledger.logs[1:]))
                self.assertTrue(all(r["candidate"][2] == seed[2] for r in ledger.logs))
                self.assertEqual(group[2], seed[2])
                if direction == "max":
                    self.assertGreaterEqual(score, incumbent)
                else:
                    self.assertLessEqual(score, incumbent)
                # Local archive scores retain their original sign/semantics.
                for candidate, value in zip(users, scores):
                    self.assertAlmostEqual(value, objective(candidate))

    def test_all_invalid_and_final_partial_local_budget(self):
        base = self.objective()
        objective = CategoricalUtilityGap(base.demographics, base.utilities,
                    base.feature_names, base.categories, min_support=10000)
        for algorithm in ("GA", "ACO"):
            adapter = FairASSearchAdapter(algorithm, population=6, local_budget=7)
            ledger = adapter.run(objective, budget=13)
            self.assertEqual(len(ledger.logs), 13)
            self.assertIsNone(ledger.best_group)
            self.assertFalse(ledger.high)
            self.assertTrue(all(not r["valid"] for r in ledger.logs))
        objective = self.objective()
        adapter = FairASSearchAdapter("GA", population=20, local_budget=20)
        ledger = adapter.run(objective, budget=2)
        self.assertEqual(len(ledger.logs), 2)
        self.assertEqual(adapter.metadata_["local_invocations"], 1)

    def test_portable_reproducibility_and_collision_safe_exhaustion(self):
        for algorithm in ("GA", "ACO"):
            adapter = FairASSearchAdapter(algorithm, population=4, local_budget=9, radius=1)
            a = adapter.run(self.objective(), budget=41, seed=8)
            b = adapter.run(self.objective(), budget=41, seed=8)
            self.assertEqual(len(a.logs), 41)
            self.assertEqual([r["candidate"] for r in a.logs], [r["candidate"] for r in b.logs])
        adapter = FairASSearchAdapter("GA", population=2, local_budget=2, radius=0)
        ledger = adapter.run(self.objective((12, 12)), budget=1000)
        self.assertEqual(len(ledger.cache), 144)
        self.assertIn((1, 11), ledger.cache)
        self.assertIn((11, 1), ledger.cache)
        self.assertEqual(adapter.metadata_["stopping"], "categorical_space_exhausted")

    def test_pipeline_factory_elimination_and_export(self):
        development, dev_u = mock_users(180, 10, 4)
        audit, audit_u = mock_users(300, 10, 5)
        full_objective = CategoricalUtilityGap.from_frame(audit, audit_u, min_support=2)
        selector = CategoricalUnionBackwardSelector(top_n=2, max_features=2, cv=2,
            estimator=make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                    DecisionTreeClassifier(max_depth=4, random_state=42)))
        tester = HybridFairnessTester(selector, budget=45, population=4, local_budget=9)
        seen = []
        def factory(names):
            seen.append(names)
            return full_objective.project(names)
        with tempfile.TemporaryDirectory() as directory:
            groups = tester.run(development, (dev_u < 0.7).astype(int), objective_factory=factory, out=directory)
            self.assertEqual(seen, [tester.selected_features_])
            self.assertTrue(groups)
            self.assertFalse(set(tester.selected_features_) & set(tester.eliminated_features_))
            self.assertEqual(len(tester.selected_features_) + len(tester.eliminated_features_), 10)
            self.assertLessEqual(len(tester.selected_features_), 2)
            for name in tester.eliminated_features_:
                self.assertEqual(groups[0]["full_rule"][name], "*")
            root = Path(directory)
            self.assertEqual(json.loads((root / "feature_selection.json").read_text())["selected_features"],
                             list(tester.selected_features_))
            trace = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(len(trace), len(tester.ledger_.logs))
            self.assertTrue((root / "subgroups.csv").exists())


class NativeFairASHybridTests(unittest.TestCase):
    source = Path("C:/Users/DELL/Documents/repo/FairAS/run_s.py")

    @unittest.skipUnless(source.exists(), "Optional native FairAS source unavailable")
    def test_native_outer_loop_calls_replaced_hook_with_shared_objective(self):
        objective = HybridTests().objective()
        for algorithm in ("GA", "ACO"):
            adapter = FairASSearchAdapter(algorithm, population=4, local_budget=9, radius=1,
                                         native_source=self.source)
            ledger = adapter.run(objective, budget=43)
            self.assertEqual(len(ledger.logs), 43)
            self.assertGreater(adapter.metadata_["local_invocations"], 0)
            self.assertEqual(adapter.metadata_["backend"], "native")
            self.assertIn("source_sha256", adapter.metadata_)
            for row in ledger.logs:
                self.assertAlmostEqual(row["disparity"], objective(row["candidate"]))
                if row["phase"] == "local_search":
                    self.assertLessEqual(row["hamming_distance"], 1)


if __name__ == "__main__":
    unittest.main()
