"""Torch-free categorical utility objectives and explicit callable contracts."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .native import ExactCellSpace


class CategoricalUtilityGap:
    """User-weighted global-minus-group utility; projected predicates are partial.

    One row per audit user, fixed utilities/model, fixed global reference.
    An optional score_users(indices) callback runs actual frozen inference.
    Projection changes the rule grammar, not the users, metric or global mean.
    """
    def __init__(self, demographics, utilities, feature_names, categories,
                 min_support=20, score_users=None, source_id="user-utility-table"):
        self.feature_names = tuple(feature_names)
        self.categories = tuple(tuple(v) for v in categories)
        if len(set(self.feature_names)) != len(self.feature_names) or len(self.categories) != len(self.feature_names):
            raise ValueError("Unique feature names and matching category vocabularies required")
        self.space = ExactCellSpace(tuple(len(v) for v in self.categories))
        X, u = np.asarray(demographics), np.asarray(utilities, dtype=float)
        if X.shape != (len(u), self.space.dimension) or not len(u) or not np.isfinite(u).all():
            raise ValueError("One finite utility per audit user required")
        if min_support < 1:
            raise ValueError("min_support must be positive")
        for row in X:
            self.space.validate(row)
        self.demographics, self.utilities = X.astype(int).copy(), u.copy()
        self.demographics.flags.writeable = self.utilities.flags.writeable = False
        self.global_utility = float(u.mean())
        self.min_support, self.score_users, self.source_id = min_support, score_users, source_id
        digest = hashlib.sha256(repr((self.feature_names, self.categories, min_support,
                                      source_id, "global-minus-group-v1")).encode())
        digest.update(self.demographics.tobytes())
        digest.update(self.utilities.tobytes())
        self.fingerprint = digest.hexdigest()

    @classmethod
    def from_frame(cls, frame, utilities, min_support=20, **kwargs):
        if not isinstance(frame, pd.DataFrame) or frame.isna().any().any():
            raise ValueError("DataFrame without missing categorical attributes required")
        # Preserve semantic categories: no arithmetic on codes is meaningful.
        categories = tuple(tuple(pd.unique(frame[name])) for name in frame.columns)
        codes = np.column_stack([pd.Categorical(frame[name], categories=values).codes
                                 for name, values in zip(frame.columns, categories)])
        return cls(codes, utilities, tuple(frame.columns), categories, min_support, **kwargs)

    def project(self, feature_names):
        names = tuple(feature_names)
        if not names or len(set(names)) != len(names) or not set(names) <= set(self.feature_names):
            raise ValueError("Project onto a nonempty, distinct subset of named features")
        indices = [self.feature_names.index(name) for name in names]
        return CategoricalUtilityGap(self.demographics[:, indices], self.utilities, names,
                tuple(self.categories[j] for j in indices), self.min_support,
                self.score_users, self.source_id)

    def decode(self, vector, one_hot=False):
        if not one_hot:
            return self.space.validate(vector)
        bits = np.asarray(vector)
        if bits.shape != (sum(self.space.cardinalities),) or not np.isin(bits, (0, 1)).all():
            raise ValueError("Expected concatenated one-hot categorical blocks")
        codes, start = [], 0
        for width in self.space.cardinalities:
            block = bits[start:start + width]
            if block.sum() != 1:
                raise ValueError("Each categorical block needs exactly one active bit")
            codes.append(int(block.argmax()))
            start += width
        return tuple(codes)

    def details(self, vector, one_hot=False):
        group = self.decode(vector, one_hot)
        members = np.flatnonzero(np.all(self.demographics == group, axis=1))
        valid = len(members) >= self.min_support
        utility = None
        if valid:
            values = self.utilities[members] if self.score_users is None else np.asarray(self.score_users(members), float)
            if values.shape != (len(members),) or not np.isfinite(values).all():
                raise ValueError("score_users must return one finite utility per member")
            utility = float(values.mean())
        return dict(subgroup=group, demographic_values={name: values[v] for name, values, v
                    in zip(self.feature_names, self.categories, group)}, support=len(members),
                    valid=valid, subgroup_utility=utility, global_utility=self.global_utility,
                    disparity=None if utility is None else self.global_utility - utility)

    def evaluate_subgroup_disparity(self, vector, one_hot=False):
        return self.details(vector, one_hot)["disparity"]

    def __call__(self, vector):
        return self.evaluate_subgroup_disparity(vector)


class CallableSubgroupObjective:
    """Wrap a fixed callable returning finite signed disparity or None.

    Bare functions cannot safely project eliminated features: use an explicit
    objective_factory(selected_names), or a projectable utility/live objective.
    """
    def __init__(self, evaluate_subgroup_disparity, cardinalities, feature_names,
                 objective_id, input_encoding="categorical"):
        if input_encoding not in {"categorical", "one_hot"} or not objective_id:
            raise ValueError("Explicit objective identifier and input encoding required")
        self.function = evaluate_subgroup_disparity
        self.space = ExactCellSpace(tuple(cardinalities))
        self.feature_names = tuple(feature_names)
        if len(self.feature_names) != self.space.dimension or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("One distinct name per feature required")
        self.input_encoding = input_encoding
        self.fingerprint = hashlib.sha256(repr((objective_id, cardinalities, feature_names, input_encoding)).encode()).hexdigest()

    def details(self, vector):
        group = self.space.validate(vector)
        argument = (np.concatenate([np.eye(c, dtype=int)[v] for c, v in zip(self.space.cardinalities, group)])
                    if self.input_encoding == "one_hot" else np.asarray(group, dtype=int))
        value = self.function(argument)
        if value is not None and not np.isfinite(value):
            raise ValueError("Objective returns finite disparity or None for invalid cells")
        return dict(subgroup=group, demographic_values=dict(zip(self.feature_names, group)),
                    support=None, valid=value is not None,
                    subgroup_utility=None, global_utility=None,
                    disparity=None if value is None else float(value))

    def __call__(self, vector):
        return self.details(vector)["disparity"]


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")
