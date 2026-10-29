"""Restore byte-identical original JSONL traces for the older asset builder."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = json.loads((ROOT / "compressed_original_traces.json").read_text())
    for entry in entries:
        destination = (ROOT / entry["original_path"]).resolve()
        source = (ROOT / entry["compressed_path"]).resolve()
        if not destination.is_relative_to(ROOT / "outputs") or not source.is_relative_to(ROOT / "outputs"):
            raise ValueError("Trace path must stay inside the repository's outputs directory")
        if destination.suffix != ".jsonl":
            raise ValueError("Expected an original JSONL trace")
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != entry["original_sha256"]:
                raise ValueError(f"Refusing to overwrite a different existing file: {destination}")
            continue
        with gzip.open(source, "rb") as stream:
            content = stream.read()
        if hashlib.sha256(content).hexdigest() != entry["original_sha256"]:
            raise ValueError(f"Archive does not match original trace: {source}")
        destination.write_bytes(content)
    print(f"Verified/restored {len(entries)} original traces")


if __name__ == "__main__":
    main()
