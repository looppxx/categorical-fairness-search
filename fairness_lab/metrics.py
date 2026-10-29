"""Explicit descriptive fairness metrics; causal claims require extra assumptions."""
from __future__ import annotations
import numpy as np


def audit_groups(oracle, candidates, threshold=0.1):
    """Re-evaluate frozen discovered rules on an independent user set."""
    rows = []
    for g in sorted(set(tuple(g) for g in candidates)):
        r = oracle(g)
        if not r.valid:
            rows.append({"group": list(g), "support": r.support, "valid": False})
            continue
        mask = oracle.space.mask(oracle.X, g)
        # Mean of bottom 10% of per-user utility: severity in the subgroup tail.
        utility = np.sort(oracle.utility[mask])
        tail = float(utility[:max(1, int(np.ceil(0.1 * len(utility))))].mean())
        parents = []
        if np.count_nonzero(np.asarray(g) >= 0) >= 2:
            for j in np.flatnonzero(np.asarray(g) >= 0):
                p = list(g)
                p[j] = -1
                result = oracle(p)
                if result.valid:
                    parents.append(result.gap)
        # All parents required for a strict masking claim.
        complete = len(parents) == np.count_nonzero(np.asarray(g) >= 0)
        masking = (max(0.0, r.gap - max(parents)) if complete else None)
        rows.append({"group": list(g), "support": r.support, "valid": True,
                     "gap": r.gap, "disadvantaged": r.gap > threshold,
                     "bottom_decile_utility": tail,
                     "intersection_amplification": masking,
                     "masked_disadvantage": bool(complete and r.gap > threshold and max(parents) <= threshold)})
    return rows


def coverage_metrics(discovery_oracle, found, reference_groups, threshold=0.1, jaccard_limit=0.8):
    """Recall requires a predeclared reference universe; exhaustive only when small.

    Greedy Jaccard diversity is order-dependent, so sort by decreasing gap first.
    """
    oracle = discovery_oracle
    target = {tuple(g) for g in reference_groups if oracle(g).valid and oracle(g).gap > threshold}
    hit = {tuple(g) for g in found} & target
    ranked = sorted(hit, key=lambda g: (-oracle(g).gap, g))
    masks, diverse = [], []
    covered = np.zeros(len(oracle.X), bool)
    for g in ranked:
        mask = oracle.space.mask(oracle.X, g)
        covered |= mask
        if all(np.sum(mask & m) / max(1, np.sum(mask | m)) < jaccard_limit for m in masks):
            masks.append(mask)
            diverse.append(g)
    return {"reference_disadvantaged_groups": len(target), "discovered_reference_groups": len(hit),
            "reference_recall": len(hit) / len(target) if target else None,
            "diverse_disadvantaged_groups": len(diverse),
            "affected_user_fraction": float(covered.mean())}


def temporal_gap_change(early, late, candidates):
    """Matched group-rule gap change; time windows need comparable utility scales."""
    out = []
    for g in candidates:
        a, b = early(g), late(g)
        if a.valid and b.valid:
            out.append({"group": list(g), "early_gap": a.gap, "late_gap": b.gap,
                        "gap_amplification": b.gap - a.gap})
    return out


def exposure_opportunity_gap(recommendations, provider_group, opportunity_shares):
    """Discounted provider exposure minus externally specified eligible-item share.

    recommendations: n_users x k integer item IDs. provider_group[item_id] gives
    provider category. Opportunity shares must cover exactly those categories;
    equal exposure is not assumed to be the correct target.
    """
    R, providers = np.asarray(recommendations), np.asarray(provider_group)
    if R.ndim != 2 or not R.size or not np.issubdtype(R.dtype, np.integer):
        raise ValueError("Nonempty integer recommendation matrix required")
    if np.any(R < 0) or np.any(R >= len(providers)):
        raise ValueError("Unknown recommended item")
    categories = set(providers.tolist())
    if set(opportunity_shares) != categories or any(v < 0 for v in opportunity_shares.values()):
        raise ValueError("Supply a nonnegative opportunity share for every provider category")
    if not np.isclose(sum(opportunity_shares.values()), 1):
        raise ValueError("Opportunity shares must sum to one")
    weights = np.broadcast_to(1 / np.log2(np.arange(R.shape[1]) + 2), R.shape)
    return {str(g): float(weights[providers[R] == g].sum() / weights.sum() - opportunity_shares[g])
            for g in sorted(categories, key=str)}


def counterfactual_rank_instability(rankings_a, rankings_b):
    """1 - Jaccard@k on paired, schema-valid protected-attribute interventions.

    Caller must generate both recommendation matrices from the same frozen DRS.
    This sensitivity metric alone does not establish unlawful or causal bias.
    """
    a, b = np.asarray(rankings_a), np.asarray(rankings_b)
    if a.shape != b.shape or a.ndim != 2 or not a.size:
        raise ValueError("Matched nonempty user x k rankings required")
    return np.array([1 - len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)])
