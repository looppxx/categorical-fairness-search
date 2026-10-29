# Release contents and external inputs

This repository packages the manuscript, source code, configuration references, and recorded search artifacts for the MovieLens-1M/LFM360K/Amazon study. The original workspace also contains exploratory records outside the final manuscript scope. Archived aggregate tables preserve that history; the current manuscript builder selects only the documented dataset blocks.

Included:

- Modern search, scoring, selection, statistics, and experiment-runner code; the legacy frozen-score exporter; research-control tests.
- Current LaTeX source, bibliography, figures/tables, and compiled PDF.
- Original synthetic and MovieLens aggregate results, conference-extension ledgers/exact references, ABC/SA additions, and supplementary external-dataset search records.
- MovieLens selection-target/cap/support summaries and request ledgers.
- Original protocol records, source snapshots, input fingerprints, and external frozen-export provenance.
- `release_input_manifest.json`: SHA-256 checksums for the copied source/artifact files at packaging time. Locally generated summaries may change their hashes when rebuilt.

Not bundled:

- Raw or processed row-level datasets, saved model weights, one-row-per-user utility tables, and user-partition ID lists.
- Historical feature-selection reports that embed development-user ID arrays. Their selected features are available in the aggregate selection summaries.
- Original environment directories, local caches, and original upstream FairRec/FairAS source trees.
- Duplicate records from the interrupted external attempt, except the source/protocol provenance retained with the completed run.

Obtain external inputs separately using [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The recorded-evidence rebuild (`python draft/build_revision_assets.py`) works with the bundled ledgers and aggregate records. Rerunning model inference/training or the earliest publication builder additionally requires the external inputs and original reports listed in its input manifest. The original absolute paths are provenance, not a promise that those paths exist on another machine.

The largest original uncompressed JSONL files are distributed as losslessly compressed `.jsonl.gz` files. `compressed_original_traces.json` records their original paths and SHA-256 hashes. If those original JSONL paths are needed for the earlier asset builder, run:

```bash
python scripts/unpack_original_traces.py
```

The script verifies the original checksums before writing files. The current revision-asset builder reads its compressed traces directly and does not need this extraction step.

The repository records data/checkpoint provenance but does not establish their original training histories, grant rights to third-party data, or independently validate historical train/test separation. See the paper's limitations and reproduction guide.
