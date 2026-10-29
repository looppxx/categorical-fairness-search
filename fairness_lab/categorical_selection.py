"""Categorical union(mRMR, chi-square) -> greedy backward elimination.

The two filters rank the same inputs independently; neither prefilters the
other. 'Parallel' is the logical filter branches, without nested CPU pools.
Union does not guarantee retention of XOR/rare or other joint fairness harms.
Labels must come from training/development evidence, never final test scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .features import FilterConfig, HybridFeatureSelector


class CategoricalUnionBackwardSelector(TransformerMixin, BaseEstimator):
    def __init__(self, top_n=10, max_features=5, min_features=1, mandatory=(),
                 estimator=None, cv=3, scoring="balanced_accuracy", tolerance=0.001, seed=42):
        self.top_n, self.max_features, self.min_features = top_n, max_features, min_features
        self.mandatory, self.estimator, self.cv = mandatory, estimator, cv
        self.scoring, self.tolerance, self.seed = scoring, tolerance, seed

    def _matrix(self, X):
        if isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_") and tuple(X.columns) != tuple(self.feature_names_in_):
                raise ValueError("Categorical feature names/order changed")
            X = X.to_numpy()
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or not X.shape[0] or pd.isna(X).any():
            raise ValueError("Nonempty categorical table required; impute missing values in training folds")
        return X.astype(str)

    def fit(self, X, y, groups=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        raw = self._matrix(X)
        # Codes are labels only. The estimator one-hot encodes inside every CV
        # training fold, so category values do not acquire ordinal geometry.
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(raw)
        encoded = self.encoder_.transform(raw)
        estimator = self.estimator if self.estimator is not None else make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(max_iter=1000, random_state=self.seed))
        self.selector_ = HybridFeatureSelector(
            estimator, FilterConfig(entropy_threshold=0.0, top_k=self.top_n, fusion="union",
                                    categorical=tuple(range(raw.shape[1])), mandatory=tuple(self.mandatory)),
            direction="backward", max_features=self.max_features, min_features=self.min_features,
            tolerance=self.tolerance, cv=self.cv, scoring=self.scoring, seed=self.seed).fit(encoded, y, groups)
        self.n_features_in_ = raw.shape[1]
        self.selected_ = self.selector_.selected_.copy()
        self.filter_union_ = self.selector_.filter_.selected_.copy()
        self.mrmr_order_ = self.selector_.filter_.mrmr_order_
        self.chi_order_ = self.selector_.filter_.chi_order_
        self.history_ = self.selector_.history_
        return self

    def transform(self, X):
        encoded = self.encoder_.transform(self._matrix(X))
        return self.selector_.transform(encoded)

    def get_support(self, indices=False):
        return self.selector_.get_support(indices)

    def get_feature_names_out(self, input_features=None):
        names = (np.asarray(input_features) if input_features is not None else
                 getattr(self, "feature_names_in_", np.array([f"x{i}" for i in range(self.n_features_in_)])))
        return names[self.selected_]
