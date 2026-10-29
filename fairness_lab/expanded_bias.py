"""Descriptive slate audits alongside an UNCHANGED utility search objective.

FairRec already studies popularity and diversity. Calibration (preference/slate
Jensen-Shannon divergence) and exposure versus eligible opportunity are separate
extensions here, not claims that these metrics are new to the literature.
Higher divergence alone does not establish causal or statistically significant bias.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _profiles(array, width, name, allow_empty=False):
    values = np.asarray(array, dtype=float)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} must be a finite nonnegative row-by-category matrix")
    totals = values.sum(axis=1, keepdims=True)
    if not allow_empty and np.any(totals == 0):
        raise ValueError(f"{name} contains an empty category profile")
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


class ExpandedBiasEvaluator:
    """Precompute per-user slate measurements using frozen, aligned user rows.

    recommendations: integer user x K item codes, no duplicates within a slate.
    item_features: nonnegative item x category metadata (e.g. multi-hot genres).
    history_profiles: TRAINING-ONLY user x category preferences. Empty histories
      yield unavailable calibration (never a fabricated zero divergence).
    opportunity_profiles: prespecified user x category eligible-item targets;
      default is catalog category share, an explicit descriptive reference.
    training_item_counts: item popularity calculated without held-out interactions.
    """
    DEFINITIONS = {
        "calibration_js": "Base-2 Jensen-Shannon divergence of training preference and discounted slate category profiles; higher is worse calibration.",
        "exposure_opportunity_tv": "Total variation between discounted category exposure and prespecified eligible-item opportunity; higher is greater mismatch.",
        "normalized_popularity": "Mean log1p(training item count), divided by catalog maximum log count; direction is descriptive.",
        "long_tail_rate": "Fraction of recommended items outside the prespecified most-popular head; direction is descriptive.",
        "intra_list_diversity": "Mean pairwise cosine distance over nonnegative item category vectors; lower is less diverse.",
    }

    def __init__(self, recommendations, item_features, training_item_counts,
                 history_profiles, opportunity_profiles=None, head_fraction=0.2,
                 user_ids=None, source_id="fixed-slates-v1"):
        R, features = np.asarray(recommendations), np.asarray(item_features, dtype=float)
        if R.ndim != 2 or not R.size or not np.issubdtype(R.dtype, np.integer):
            raise ValueError("Nonempty integer user x K recommendation slates required")
        if features.ndim != 2 or not features.size or not np.isfinite(features).all() or np.any(features < 0):
            raise ValueError("Finite nonnegative item-category features required")
        n_items, n_categories = features.shape
        if np.any(features.sum(axis=1) == 0) or np.any(R < 0) or np.any(R >= n_items):
            raise ValueError("Every catalog item needs category metadata; unknown item code")
        if any(len(set(row.tolist())) != len(row) for row in R):
            raise ValueError("Recommendation slates must not contain repeated item codes")
        counts = np.asarray(training_item_counts, dtype=float)
        if counts.shape != (n_items,) or not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("One finite nonnegative training popularity count per item required")
        if not 0 < head_fraction < 1:
            raise ValueError("head_fraction must lie strictly between zero and one")
        n_users, k = R.shape
        preferences = _profiles(history_profiles, n_categories, "history_profiles", allow_empty=True)
        if preferences.shape[0] != n_users:
            raise ValueError("Training preference rows must align with recommendation users")
        category_items = _profiles(features, n_categories, "item_features")
        if opportunity_profiles is None:
            opportunity_profiles = np.broadcast_to(category_items.mean(axis=0), (n_users, n_categories))
        opportunity = _profiles(opportunity_profiles, n_categories, "opportunity_profiles")
        if opportunity.shape[0] != n_users:
            raise ValueError("Opportunity targets must align with recommendation users")
        self.n_users, self.k = n_users, k
        self.user_ids = None if user_ids is None else np.array(user_ids, copy=True)
        if self.user_ids is not None and (self.user_ids.shape != (n_users,) or len(np.unique(self.user_ids)) != n_users):
            raise ValueError("Distinct aligned user IDs required")
        weights = 1 / np.log2(np.arange(k) + 2)
        exposure = (category_items[R] * weights[None, :, None]).sum(axis=1) / weights.sum()
        middle = (preferences + exposure) / 2
        def kl(p, q):
            term = np.zeros_like(p)
            mask = p > 0
            term[mask] = p[mask] * np.log2(p[mask] / q[mask])
            return term.sum(axis=1)
        calibration = 0.5 * (kl(preferences, middle) + kl(exposure, middle))
        calibration[preferences.sum(axis=1) == 0] = np.nan
        popularity = np.log1p(counts)
        scale = popularity.max()
        popularity = popularity / scale if scale > 0 else np.zeros_like(popularity)
        order = np.argsort(-counts, kind="stable")
        head = np.zeros(n_items, dtype=bool)
        head[order[:max(1, int(np.ceil(head_fraction * n_items)))]] = True
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        ild = np.full(n_users, np.nan)
        if k > 1:
            # ||sum v_i||^2 - K sums all off-diagonal cosine similarities.
            summed = normalized[R].sum(axis=1)
            ild = np.clip(1 - ((summed * summed).sum(axis=1) - k) / (k * (k - 1)), 0, 1)
        self.per_user = dict(calibration_js=calibration,
            exposure_opportunity_tv=0.5 * np.abs(exposure - opportunity).sum(axis=1),
            normalized_popularity=popularity[R].mean(axis=1),
            long_tail_rate=(~head[R]).mean(axis=1), intra_list_diversity=ild)
        for values in self.per_user.values():
            values.flags.writeable = False
        self.global_means = {name: float(values[np.isfinite(values)].mean())
                             if np.isfinite(values).any() else None for name, values in self.per_user.items()}
        digest = hashlib.sha256(repr((head_fraction, source_id, self.DEFINITIONS)).encode())
        for array in (R, features, counts, preferences, opportunity):
            digest.update(repr(array.shape).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        if self.user_ids is not None:
            digest.update(repr(self.user_ids.tolist()).encode())
        self.fingerprint = digest.hexdigest()

    @classmethod
    def from_neumf(cls, objective, item_features, opportunity_profiles=None, head_fraction=0.2):
        """Capture actual frozen NeuMF slates and TRAINING-ONLY preference/counts.

        Item metadata rows MUST match objective.data.item_ids, not raw movie IDs.
        Default opportunity is the category share of each user's eligible catalog;
        this is a chosen exposure reference, not an assumed ethical requirement.
        """
        from .recommender import ranking_utilities
        data, features = objective.data, np.asarray(item_features, float)
        if features.ndim != 2 or features.shape[0] != len(data.item_ids):
            raise ValueError("Metadata rows must match data.item_ids in exactly that order")
        category_items = _profiles(features, features.shape[1], "item_features")
        rankings = ranking_utilities(objective.model, data, objective.split,
                                     objective.k, return_recommendations=True)["recommendations"]
        histories = np.array([category_items[items].sum(axis=0) for items in data.train_seen])
        if opportunity_profiles is None:
            total = category_items.sum(axis=0)
            opportunities = []
            for u, items in enumerate(data.train_seen):
                excluded = items if objective.split == "validation" else np.r_[items, data.validation_items[u]]
                opportunities.append(np.maximum(0, total - category_items[excluded].sum(axis=0)))
            opportunity_profiles = np.asarray(opportunities)
        counts = np.bincount(data.train_pairs[:, 1], minlength=len(data.item_ids))
        return cls(rankings, features, counts, histories, opportunity_profiles, head_fraction,
                   user_ids=data.user_ids, source_id=objective.fingerprint)

    def subset_users(self, indices):
        """Restrict to audit users and recompute every global audit reference."""
        raw = np.asarray(indices)
        if (raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer)
                or np.any(raw < 0) or np.any(raw >= self.n_users) or len(np.unique(raw)) != len(raw)):
            raise ValueError("Distinct nonempty integer user-row subset required")
        subset = object.__new__(type(self))
        subset.n_users, subset.k = len(raw), self.k
        subset.user_ids = None if self.user_ids is None else self.user_ids[raw].copy()
        subset.per_user = {name: values[raw].copy() for name, values in self.per_user.items()}
        for values in subset.per_user.values():
            values.flags.writeable = False
        subset.global_means = {name: float(values[np.isfinite(values)].mean())
            if np.isfinite(values).any() else None for name, values in subset.per_user.items()}
        subset.fingerprint = hashlib.sha256(repr((self.fingerprint, raw.tolist(), "audit-subset-v1")).encode()).hexdigest()
        return subset

    def evaluate_members(self, member_indices, min_support=1):
        """Return signed, absolute and direction-aware gaps against audit users.

        Absolute gaps measure separation without declaring either popularity
        extreme harmful. Unavailable measurements remain None in every field.
        """
        members = np.asarray(member_indices)
        if (not isinstance(min_support, (int, np.integer)) or isinstance(min_support, (bool, np.bool_))
                or min_support < 1 or members.ndim != 1
                or (members.size and not np.issubdtype(members.dtype, np.integer))
                or np.any(members < 0) or np.any(members >= self.n_users)
                or len(np.unique(members)) != len(members)):
            raise ValueError("Distinct valid member-row indices required")
        members = members.astype(int)
        report = {}
        for name, all_values in self.per_user.items():
            values = all_values[members]
            values = values[np.isfinite(values)]
            mean = float(values.mean()) if len(values) >= min_support else None
            reference = self.global_means[name]
            difference = None if mean is None or reference is None else mean - reference
            # Positive directional gaps: excess mismatch or a deficit of ILD.
            directional = None if difference is None else (-difference if name == "intra_list_diversity" else difference)
            if name in {"normalized_popularity", "long_tail_rate"}:
                directional = None  # no imposed normative direction
            report[name] = dict(group_mean=mean, global_mean=reference,
                group_minus_global=difference,
                absolute_disparity=None if difference is None else abs(difference),
                disparity=directional, valid_users=len(values),
                available=mean is not None and reference is not None)
        return report


def flatten_bias_report(report=None, metric_names=None):
    """Stable CSV columns for audits of the utility-selected best subgroup.

    These are companion measurements at that subgroup, not separately searched
    maxima for each bias. Passing None emits explicit missing measurements.
    """
    report = {} if report is None else report
    names = ExpandedBiasEvaluator.DEFINITIONS if metric_names is None else metric_names
    fields = ("group_mean", "global_mean", "group_minus_global", "absolute_disparity",
              "disparity", "valid_users", "available")
    return {f"{name}_{field}": report.get(name, {}).get(field)
            for name in names for field in fields}


class ExpandedBiasObjective:
    """Decorator: audit side channels never alter the primary search fitness."""
    def __init__(self, objective, evaluator):
        if not hasattr(objective, "demographics") or len(objective.demographics) != evaluator.n_users:
            raise ValueError("Objective and slate evaluator need the same ordered audit users")
        actual_ids = getattr(getattr(objective, "data", None), "user_ids", getattr(objective, "user_ids", None))
        if actual_ids is not None and evaluator.user_ids is not None and not np.array_equal(actual_ids, evaluator.user_ids):
            raise ValueError("Objective and recommendations have different user row order")
        self.base, self.evaluator = objective, evaluator
        self.space, self.feature_names = objective.space, objective.feature_names
        self.categories = getattr(objective, "categories", ())
        self.demographics, self.min_support = objective.demographics, objective.min_support
        self.fingerprint = objective.fingerprint  # primary objective identity stays fixed
        self.audit_fingerprint = evaluator.fingerprint
        self.cache_utilities = getattr(objective, "cache_utilities", getattr(objective, "score_users", None) is None)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def project(self, feature_names):
        return ExpandedBiasObjective(self.base.project(feature_names), self.evaluator)

    def details(self, vector, one_hot=False):
        detail = dict(self.base.details(vector, one_hot=one_hot) if one_hot else self.base.details(vector))
        group = detail["subgroup"]
        members = np.flatnonzero(np.all(self.demographics == group, axis=1))
        detail["expanded_bias"] = self.evaluator.evaluate_members(members, self.min_support)
        detail["audit_sha256"] = self.audit_fingerprint
        return detail

    def evaluate_subgroup_disparity(self, vector, one_hot=False):
        return self.details(vector, one_hot)["disparity"]

    def __call__(self, vector):
        return self.evaluate_subgroup_disparity(vector)


def movielens_genre_features(data, directory):
    """Read static genres and align rows to internal item codes, not raw IDs."""
    from pathlib import Path
    import pandas as pd
    movies = pd.read_csv(Path(directory) / "movies.dat", sep="::", engine="python",
                         names=["item_id", "title", "genres"], encoding="latin-1")
    if movies.item_id.duplicated().any():
        raise ValueError("Duplicate movie metadata IDs")
    aligned = movies.set_index("item_id").loc[data.item_ids]
    if aligned.genres.isna().any():
        raise ValueError("Every ranked item requires genre metadata")
    genres = [str(value).split("|") for value in aligned.genres]
    names = sorted({name for row in genres for name in row})
    index = {name: j for j, name in enumerate(names)}
    features = np.zeros((len(data.item_ids), len(names)))
    for i, row in enumerate(genres):
        for name in row:
            features[i, index[name]] = 1
    return features, names
