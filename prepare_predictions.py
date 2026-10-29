"""Export predictions using the existing repo's legacy Python/TensorFlow env.

Run this script directly; it deliberately does not import fairness_lab, which
targets a newer Python. This does not train or save a recommender checkpoint.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="FairRec or FairAS source folder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", choices=["fairrec", "sampling"], required=True)
    parser.add_argument("--model-dir", help="Override missing or relative model directory")
    parser.add_argument("--out", required=True, help="Absolute output CSV in the result workspace")
    args = parser.parse_args()
    source, config, output = Path(args.source).resolve(), Path(args.config).resolve(), Path(args.out).resolve()
    params = json.loads(config.read_text(encoding="utf-8"))[args.section]
    model_params, data_params = params["model_params"].copy(), params["data_params"].copy()
    data_root = Path(data_params["data_dir"])
    if not data_root.is_absolute():
        data_root = source / data_root
    data_params["data_dir"] = str(data_root.resolve()) + os.sep
    model_root = Path(args.model_dir or model_params.get("model_dir", str(source / "model")))
    if not model_root.is_absolute():
        model_root = source / model_root
    model_params["model_dir"] = str(model_root.resolve()) + os.sep
    test_csv = data_root / data_params["dataset"] / "processed_data/test_data.csv"
    model_name = model_params["model_name"] + "-" + data_params["dataset"]
    checkpoint = model_root / model_name / (model_name + ".ckpt")
    if not test_csv.is_file() or not Path(str(checkpoint) + ".index").is_file():
        raise FileNotFoundError("Existing test CSV and TensorFlow checkpoint required; no folders will be created upstream")
    if not output.parent.is_dir():
        raise FileNotFoundError("Create the output folder in the result workspace first")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    native = importlib.import_module("utilis")
    frame = native.get_test_data(model_params, data_params)
    _, train_features = native.get_saved_model_details(model_params, data_params)
    model = native.get_model(model_params, data_params)
    frame = frame.copy()
    frame["predict"] = model.predict({name: frame[name].values for name in train_features}).reshape(-1)
    frame.to_csv(output, index=False)
    metadata = {"source": str(source), "config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "test_csv_sha256": hashlib.sha256(test_csv.read_bytes()).hexdigest(),
                "checkpoint_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in checkpoint.parent.glob(checkpoint.name + ".*")},
                "rows": len(frame), "training_features": train_features,
                "note": "Prediction export only; create independent discovery and confirmation splits before searching"}
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
