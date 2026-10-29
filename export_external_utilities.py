"""Stream frozen legacy DeepFM scores into a one-row-per-user utility table.

Run with the existing Python 3.8/TensorFlow 2.3 environment. No training or
upstream writes. Candidate sets are the supplied processed test rows, not the
full catalog. Exactly one positive per user is required. Ties rank the smaller
item id first; duplicate candidate rows remain part of the supplied protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dataset", choices=["last-fm", "black-fri", "amazon"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=8192)
    args = p.parse_args()
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.source.resolve()))
    import utilis
    source = args.source / "data" / args.dataset / "processed_data/test_data.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "user_utilities.csv"
    if target.exists() or (args.out / "metadata.json").exists():
        raise ValueError("Choose an unused output directory")
    params = dict(model_name="deepfm", embedding_dim=4, batch_size=args.batch_size,
                  model_dir=str((args.source / "model").resolve()) + os.sep)
    data_params = dict(dataset=args.dataset)
    checkpoint, features = utilis.get_saved_model_details(params, data_params)
    from deepctr.feature_column import SparseFeat
    from deepctr.models import DeepFM
    columns = [SparseFeat(n, utilis.get_feat_dict(args.dataset)[n], embedding_dim=4) for n in features]
    model = DeepFM(columns, columns, task="binary")
    restore = model.load_weights(checkpoint)
    restore.assert_existing_objects_matched()
    restore.expect_partial()  # supplied optimizer slots are irrelevant to frozen inference
    attrs = {"last-fm": ["gender", "age", "country"],
             "black-fri": ["gender", "age", "occupation", "city", "stay_years", "martial_status"],
             "amazon": ["his_level", "mean_price_level", "max_price_level"]}[args.dataset]
    positive_parts = []
    rows = 0
    for chunk in pd.read_csv(source, chunksize=args.chunk_size):
        if not chunk.label.isin([0, 1]).all():
            raise ValueError("Binary relevance labels required")
        rows += len(chunk)
        positive_parts.append(chunk.loc[chunk.label == 1, ["user_id", "item_id"] + attrs])
    users = pd.concat(positive_parts, ignore_index=True).sort_values("user_id").reset_index(drop=True)
    if users.user_id.duplicated().any() or users.empty:
        raise ValueError("Exactly one positive per user required")
    ids = pd.Index(users.user_id)
    pos_scores = np.empty(len(users), dtype=float)
    # Fetch feature rows for positives, then predict in their original ID order.
    parts = []
    for chunk in pd.read_csv(source, chunksize=args.chunk_size):
        parts.append(chunk.loc[chunk.label == 1, features])
    positive_features = pd.concat(parts, ignore_index=True).sort_values("user_id")
    pos_scores[:] = model.predict({n: positive_features[n].values for n in features},
                                 batch_size=args.batch_size, verbose=0).reshape(-1)
    del parts, positive_parts, positive_features
    if not np.isfinite(pos_scores).all():
        raise ValueError("Nonfinite positive scores")
    outranking = np.zeros(len(users), dtype=np.int64)
    counts = np.ones(len(users), dtype=np.int64)
    pos_items = users.item_id.to_numpy()
    negatives = 0
    for number, chunk in enumerate(pd.read_csv(source, chunksize=args.chunk_size)):
        indices = ids.get_indexer(chunk.user_id)
        if (indices < 0).any():
            raise ValueError("Candidate user without positive")
        if not np.array_equal(chunk[attrs].to_numpy(), users[attrs].to_numpy()[indices]):
            raise ValueError("Inconsistent per-user demographics")
        negative = chunk.loc[chunk.label == 0]
        if negative.empty:
            continue
        indices = ids.get_indexer(negative.user_id)
        scores = model.predict({n: negative[n].values for n in features},
                               batch_size=args.batch_size, verbose=0).reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite negative scores")
        before = (scores > pos_scores[indices]) | ((scores == pos_scores[indices]) &
                  (negative.item_id.to_numpy() <= pos_items[indices]))
        np.add.at(outranking, indices, before.astype(np.int64))
        np.add.at(counts, indices, 1)
        negatives += len(negative)
        print("%s chunk %d: %d negative scores" % (args.dataset, number, negatives), flush=True)
    if rows != len(users) + negatives or (counts < 2).any():
        raise ValueError("Incomplete candidate accounting")
    users["rank"] = outranking + 1
    users["candidates"] = counts
    users["ndcg10"] = np.where(outranking < 10, 1 / np.log2(outranking + 2), 0.)
    users.to_csv(target, index=False)
    checkpoints = sorted(Path(checkpoint).parent.glob(Path(checkpoint).name + ".*"))
    metadata = dict(dataset=args.dataset, rows=rows, users=len(users), features=features,
        audit_attributes=attrs, source_csv=str(source.resolve()), source_sha256=sha(source),
        checkpoint_sha256={str(x.resolve()): sha(x) for x in checkpoints},
        native_source_sha256=sha(args.source / "utilis.py"), exporter_sha256=sha(Path(__file__)),
        utility_csv_sha256=sha(target), python=sys.version, tensorflow=tf.__version__,
        model="supplied frozen DeepFM; no retraining; original training seeds/settings not verified",
        restoration="assert_existing_objects_matched passed for all model variables; unused saved optimizer slots allowed",
        embedding_dim=4, prediction_batch_size=args.batch_size, threads=[4, 1],
        candidate_count_min=int(counts.min()), candidate_count_max=int(counts.max()),
        utility="NDCG@10; one positive; supplied sampled candidates; score descending then item id ascending",
        duplicate_policy="candidate rows retained; negative tie with identical item id ranks before positive",
        disclosure="Post hoc external-dataset audit. Original train/test disjointness cannot be independently reconstructed.")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(dict(dataset=args.dataset, users=len(users), mean_ndcg10=float(users.ndcg10.mean()))), flush=True)


if __name__ == "__main__":
    main()
