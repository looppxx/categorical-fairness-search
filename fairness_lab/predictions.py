"""Bridge existing DRS predictions to untouched native RecMetrics methods.

Optional pandas dependency is required only for CSV input. This separates
expensive TensorFlow inference from the optimizer study without averaging AUC.
"""
from pathlib import Path
import hashlib
import numpy as np
from sklearn.metrics import roc_auc_score

from .native import NativeCellOracle, ExactCellSpace, source_definitions


class PredictionTableScorer:
    def __init__(self, frame, metrics_source, sensitive_features, metric="mrr", threshold_users=20,
                 threshold_rule="strict_greater", k=5):
        self.frame = frame.copy(deep=True)
        self.features = tuple(sensitive_features)
        self.metric, self.threshold_users, self.threshold_rule, self.k = metric, threshold_users, threshold_rule, k
        if threshold_rule not in {"strict_greater", "greater_equal"}:
            raise ValueError("Match the native support comparator explicitly")
        ns = {"np": np, "roc_auc_score": roc_auc_score}
        self.source_hash = source_definitions(metrics_source, {"RecMetrics"}, ns)
        self.RecMetrics = ns["RecMetrics"]
        self.method = {"mrr": "mrr_score", "auc": "auc_score", "ndcg": "ndcg_score",
                       "urd": "cate_diversity_score", "urp": "popularity_score"}[metric]
        required = set(self.features) | {"user_id", "label", "predict", "item_id", "cate", "popularity", "user_pop"}
        if not required <= set(self.frame.columns):
            raise ValueError(f"Missing prediction columns: {required - set(self.frame.columns)}")

    def __call__(self, g):
        mask = np.all(self.frame[list(self.features)].to_numpy() == np.asarray(g), axis=1)
        frame = self.frame.loc[mask]
        n = frame["user_id"].nunique()
        valid = n > self.threshold_users if self.threshold_rule == "strict_greater" else n >= self.threshold_users
        if not valid:
            return None, n
        if self.metric == "auc" and frame["label"].nunique() < 2:
            # Explicit undefined-AUC policy, applied to every engine.
            return None, n
        value = getattr(self.RecMetrics(frame, k=self.k), self.method)()
        return float(value), n

    def oracle(self, cardinalities, prediction_file_hash):
        tag = repr((self.source_hash, prediction_file_hash, self.features, self.metric,
                    self.threshold_users, self.threshold_rule, self.k, "undefined-auc-invalid"))
        return NativeCellOracle(ExactCellSpace(tuple(cardinalities)), self, tag)


def from_csv(path, metrics_source, sensitive_features, cardinalities, **kwargs):
    import pandas as pd
    path = Path(path)
    frame = pd.read_csv(path)
    scorer = PredictionTableScorer(frame, metrics_source, sensitive_features, **kwargs)
    return scorer.oracle(cardinalities, hashlib.sha256(path.read_bytes()).hexdigest())
