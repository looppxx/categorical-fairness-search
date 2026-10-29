"""MovieLens-1M implicit NeuMF and a frozen, real ranking-utility objective.

User vectors contain one category code per attribute, NOT arbitrary binary
integers. An optional concatenated one-hot representation is decoded strictly.
The objective is signed: global mean utility minus subgroup mean utility.
Positive values identify disadvantaged groups. Unsupported cells return None.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import ssl
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import io

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compare_fairrec_search import Space


def download_movielens(destination):
    """Download official archive and write only the three named data files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    # Use the operating-system trust store when available (including Windows
    # enterprise roots). Certificate and hostname verification stay enabled.
    try:
        import truststore
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        context = ssl.create_default_context()
    with urlopen(url, timeout=120, context=context) as response:
        content = response.read()
    with ZipFile(io.BytesIO(content)) as archive:
        for name in ("users.dat", "ratings.dat", "movies.dat"):
            (destination / name).write_bytes(archive.read("ml-1m/" + name))
    return destination


@dataclass
class MovieLensData:
    user_ids: np.ndarray
    item_ids: np.ndarray
    demographics: np.ndarray
    feature_names: tuple[str, ...]
    categories: tuple[tuple, ...]
    train_pairs: np.ndarray
    train_seen: tuple[np.ndarray, ...]
    all_positive: tuple[np.ndarray, ...]
    validation_items: np.ndarray
    test_items: np.ndarray
    source_sha256: str

    @property
    def space(self):
        return Space(tuple(len(c) for c in self.categories))


def load_movielens_1m(directory, positive_rating=4, features=("gender", "age", "occupation")):
    """Positive-only implicit task; chronological leave-last-two-out per user.

    Users need >=3 positives. Item IDs cover the observed rating catalog.
    All known positives (including held-out positives) are excluded from sampled
    training negatives, a declared static implicit-feedback sampling protocol.
    Ties in timestamps are broken by item ID. No demographic enters NeuMF.
    """
    directory = Path(directory)
    if not 1 <= positive_rating <= 5:
        raise ValueError("positive_rating must be between 1 and 5")
    if not features or len(set(features)) != len(features) or not set(features) <= {"gender", "age", "occupation", "zip"}:
        raise ValueError("Choose distinct MovieLens demographic feature names")
    users = pd.read_csv(directory / "users.dat", sep="::", engine="python",
                        names=["user_id", "gender", "age", "occupation", "zip"],
                        dtype={"zip": str}, encoding="latin-1")
    ratings = pd.read_csv(directory / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "item_id", "rating", "timestamp"])
    if users.user_id.duplicated().any() or not set(ratings.user_id) <= set(users.user_id):
        raise ValueError("Duplicate users or ratings without demographics")
    ratings = ratings.sort_values(["timestamp", "item_id"], kind="stable").drop_duplicates(
        ["user_id", "item_id"], keep="last")
    positive = ratings[ratings.rating >= positive_rating]
    counts = positive.groupby("user_id").size()
    user_ids = np.sort(counts[counts >= 3].index.to_numpy())
    item_ids = np.sort(ratings.item_id.unique())
    if not len(user_ids) or len(item_ids) < 2:
        raise ValueError("Need eligible users and at least two items")
    vocabularies = {name: tuple(sorted(users[name].unique().tolist())) for name in features}
    users = users.set_index("user_id").loc[user_ids]
    demographics, categories = [], []
    for name in features:
        # MovieLens publishes these attribute vocabularies in users.dat. Keep
        # the full provided vocabulary, including categories absent in a split.
        vocabulary = vocabularies[name]
        lookup = {value: i for i, value in enumerate(vocabulary)}
        demographics.append(users[name].map(lookup).to_numpy())
        categories.append(vocabulary)
    item_map = {value: i for i, value in enumerate(item_ids)}
    grouped = positive.groupby("user_id", sort=False)
    train, seen, all_positive, validation, test = [], [], [], [], []
    for u, uid in enumerate(user_ids):
        frame = grouped.get_group(uid).sort_values(["timestamp", "item_id"], kind="stable")
        items = np.array([item_map[v] for v in frame.item_id], dtype=np.int64)
        seen.append(items[:-2].copy())
        all_positive.append(items.copy())
        train.extend((u, int(i)) for i in items[:-2])
        validation.append(items[-2])
        test.append(items[-1])
        if len(items) == len(item_ids):
            raise ValueError(f"User {uid} has no unobserved items for negative sampling")
    digest = hashlib.sha256()
    for name in ("users.dat", "ratings.dat"):
        digest.update((directory / name).read_bytes())
    digest.update(repr((positive_rating, features, "chronological-positive-last-two-v1")).encode())
    return MovieLensData(user_ids, item_ids, np.column_stack(demographics).astype(np.int64),
                         tuple(features), tuple(categories), np.asarray(train, dtype=np.int64),
                         tuple(seen), tuple(all_positive), np.asarray(validation), np.asarray(test),
                         digest.hexdigest())


class NeuMF(nn.Module):
    """Jointly trained GMF + MLP baseline, without separate pretraining."""
    def __init__(self, n_users, n_items, embedding_dim=16, hidden=(64, 32)):
        super().__init__()
        if min(n_users, n_items, embedding_dim) < 1 or not hidden or min(hidden) < 1:
            raise ValueError("Positive model dimensions required")
        self.config = dict(n_users=n_users, n_items=n_items, embedding_dim=embedding_dim, hidden=tuple(hidden))
        self.user_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_gmf = nn.Embedding(n_items, embedding_dim)
        self.user_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_mlp = nn.Embedding(n_items, embedding_dim)
        layers, width = [], 2 * embedding_dim
        for output in hidden:
            layers.extend([nn.Linear(width, output), nn.ReLU()])
            width = output
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(embedding_dim + width, 1)
        for embedding in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(embedding.weight, std=0.01)

    def forward(self, users, items):
        gmf = self.user_gmf(users) * self.item_gmf(items)
        mlp = self.mlp(torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1))
        return self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


def train_neumf(data, epochs=5, negatives=4, batch_size=1024, learning_rate=0.001,
                seed=42, device="cpu", embedding_dim=16, validation_callback=None,
                patience=5, min_delta=1e-4, min_epochs=5, eval_every=1, epoch_callback=None):
    if min(epochs, negatives, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("Positive training settings required")
    if min(patience, min_epochs, eval_every) < 1 or min_delta < 0:
        raise ValueError("Invalid validation early-stopping settings")
    torch.manual_seed(seed)
    # CPU is the reproducible default. GPU kernels may require additional
    # platform-specific deterministic settings for bitwise reproduction.
    model = NeuMF(len(data.user_ids), len(data.item_ids), embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    catalog = np.arange(len(data.item_ids))
    pools = [np.setdiff1d(catalog, items, assume_unique=False) for items in data.all_positive]
    history = []
    best_state, best_score, anchor = None, -np.inf, -np.inf
    stale, best_epoch, stopped = 0, None, False
    for epoch in range(epochs):
        chunks = []
        for u, pool in enumerate(pools):
            n = len(data.train_seen[u]) * negatives
            chunks.append(np.column_stack([np.full(n, u), rng.choice(pool, n)]))
        negative_pairs = np.concatenate(chunks)
        pairs = np.concatenate([data.train_pairs, negative_pairs])
        labels = np.r_[np.ones(len(data.train_pairs)), np.zeros(len(negative_pairs))].astype(np.float32)
        loader = DataLoader(TensorDataset(torch.from_numpy(pairs[:, 0].copy()).long(),
                                         torch.from_numpy(pairs[:, 1].copy()).long(),
                                         torch.from_numpy(labels)),
                            batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        model.train()
        total = 0.0
        for users, items, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(users.to(device), items.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(users)
        history.append(dict(epoch=epoch + 1, mean_bce=total / len(pairs)))
        if validation_callback is not None and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
            score = float(validation_callback(model.eval(), epoch + 1))
            if not np.isfinite(score):
                raise ValueError("Training validation callback returned a non-finite score")
            history[-1]["validation_score"] = score
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if score > anchor + min_delta:
                anchor, stale = score, 0
            else:
                stale += 1
            stopped = stale >= patience and epoch + 1 >= min_epochs
        if epoch_callback is not None:
            epoch_callback(model, dict(history[-1]))
        print(f"Epoch {epoch + 1}/{epochs}: BCE={history[-1]['mean_bce']:.5f}", flush=True)
        if stopped:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_summary = dict(epochs_completed=len(history), best_epoch=best_epoch,
        best_validation_score=None if best_state is None else best_score,
        early_stopping_triggered=stopped,
        stopping="validation_patience" if stopped else "epoch_cap",
        criterion="development-user held-out utility" if validation_callback is not None else "fixed epochs")
    return model.eval(), history


def model_fingerprint(model):
    digest = hashlib.sha256(json.dumps(model.config, sort_keys=True).encode())
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def ranking_utilities(model, data, split="validation", k=10, user_indices=None,
                      user_batch_size=32, inference_batch_size=32768, return_recommendations=False):
    """Exact full-catalog one-relevant-item NDCG/Recall, no sampled evaluation.

    Validation excludes train positives. Test excludes train+validation positives.
    The target is always unmasked. Ties use ascending item code. User utility
    is 1/log2(rank+1) for an in-top-k hit (NDCG), or 1 for Recall; otherwise 0.
    Model sigmoid is unnecessary because logits have the same ranking order.
    """
    if split not in {"validation", "test"} or min(k, user_batch_size, inference_batch_size) < 1:
        raise ValueError("Invalid ranking protocol")
    indices = np.arange(len(data.user_ids)) if user_indices is None else np.asarray(user_indices, dtype=int)
    targets = data.validation_items if split == "validation" else data.test_items
    n_items = len(data.item_ids)
    device = next(model.parameters()).device
    ndcg, recall = np.zeros(len(indices)), np.zeros(len(indices))
    slates = np.empty((len(indices), k), dtype=np.int64) if return_recommendations else None
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), user_batch_size):
            batch = indices[start:start + user_batch_size]
            users = np.repeat(batch, n_items)
            items = np.tile(np.arange(n_items), len(batch))
            flat = np.empty(len(users), dtype=np.float32)
            for offset in range(0, len(users), inference_batch_size):
                end = offset + inference_batch_size
                values = model(torch.as_tensor(users[offset:end], device=device),
                               torch.as_tensor(items[offset:end], device=device))
                flat[offset:end] = values.cpu().numpy()
            scores = flat.reshape(len(batch), n_items)
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite recommender predictions")
            for row, u in enumerate(batch):
                scores[row, data.train_seen[u]] = -np.inf
                if split == "test":
                    scores[row, data.validation_items[u]] = -np.inf
                target = int(targets[u])
                target_score = scores[row, target]
                if not np.isfinite(target_score):
                    raise ValueError("Held-out target overlaps training history")
                rank = (1 + np.count_nonzero(scores[row] > target_score)
                        + np.count_nonzero((scores[row] == target_score) & (np.arange(n_items) < target)))
                if rank <= k:
                    ndcg[start + row] = 1 / float(np.log2(rank + 1))
                    recall[start + row] = 1.0
                if return_recommendations:
                    if np.isfinite(scores[row]).sum() < k:
                        raise ValueError("Eligible catalog is smaller than requested slate length")
                    slates[start + row] = np.argsort(-scores[row], kind="stable")[:k]
    result = {"ndcg": ndcg, "recall": recall}
    if return_recommendations:
        result["recommendations"] = slates
    return result


class RecommenderDisparityObjective:
    """Drop-in callable using one frozen model, users and ranking protocol.

    The fixed global reference weights each eligible user once. Cached per-user
    inference is EXACTLY equivalent to rescoring the frozen deterministic model
    for each group. Set cache_utilities=False for actual repeated forward passes.
    Input is categorical by default; one_hot=True requires exactly one active bit
    per feature block. There are no wildcard bits or silent categorical repairs.
    """
    def __init__(self, model, data, metric="ndcg", k=10, min_support=20,
                 split="validation", cache_utilities=True):
        if metric not in {"ndcg", "recall"} or min_support < 1 or k < 1:
            raise ValueError("Invalid utility or support settings")
        self.data, self.space = data, data.space
        self.feature_names, self.categories = data.feature_names, data.categories
        self.metric, self.k, self.min_support, self.split = metric, k, min_support, split
        self.cache_utilities = cache_utilities
        # Own a frozen snapshot so external training cannot change the oracle.
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.model_sha256 = model_fingerprint(self.model)
        self.utilities = ranking_utilities(self.model, data, split, k)[metric]
        self.utilities.flags.writeable = False
        self.global_utility = float(self.utilities.mean())
        self.demographics = data.demographics.copy()
        self.demographics.flags.writeable = False
        settings = (self.model_sha256, data.source_sha256, data.feature_names,
                    data.categories, metric, k, min_support, split, "global-minus-group",
                    "full-catalog-train-mask-val-also-masked-on-test-v1")
        self.fingerprint = hashlib.sha256(repr(settings).encode()).hexdigest()

    def decode(self, vector, one_hot=False):
        if not one_hot:
            return self.space.validate(vector)
        bits = np.asarray(vector)
        if bits.shape != (sum(self.space.cardinalities),) or not np.isin(bits, [0, 1]).all():
            raise ValueError("Expected concatenated one-hot feature blocks")
        values, offset = [], 0
        for c in self.space.cardinalities:
            block = bits[offset:offset + c]
            if block.sum() != 1:
                raise ValueError("Each feature block needs exactly one active bit")
            values.append(int(np.argmax(block)))
            offset += c
        return tuple(values)

    def describe(self, vector, one_hot=False):
        group = self.decode(vector, one_hot)
        return {name: categories[v] for name, categories, v in
                zip(self.data.feature_names, self.data.categories, group)}

    def score_users(self, member_indices):
        """Score the same frozen model for arbitrary user sets, including projections."""
        members = np.asarray(member_indices, dtype=int)
        return (self.utilities[members] if self.cache_utilities else
                ranking_utilities(self.model, self.data, self.split, self.k, members)[self.metric])

    def project(self, feature_names):
        """Drop predicates, leaving eliminated attributes unconstrained."""
        from .subgroup_objectives import CategoricalUtilityGap
        view = CategoricalUtilityGap(self.demographics, self.utilities,
                self.data.feature_names, self.data.categories, self.min_support,
                score_users=None if self.cache_utilities else self.score_users,
                source_id=self.fingerprint)
        return view.project(feature_names)

    def details(self, vector, one_hot=False):
        group = self.decode(vector, one_hot)
        members = np.flatnonzero(np.all(self.demographics == group, axis=1))
        n = len(members)
        valid = n >= self.min_support
        value = None
        if valid:
            utility = self.score_users(members)
            value = float(utility.mean())
        return dict(subgroup=group, demographic_values=self.describe(group), support=n,
                    valid=valid, subgroup_utility=value, global_utility=self.global_utility,
                    disparity=None if value is None else self.global_utility - value)

    def evaluate_subgroup_disparity(self, subgroup_vector, one_hot=False):
        return self.details(subgroup_vector, one_hot)["disparity"]

    def __call__(self, subgroup_vector):
        return self.evaluate_subgroup_disparity(subgroup_vector)
