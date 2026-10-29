# Categorical Fairness Search

Code and recorded search artifacts for **An Empirical Analysis of Categorical Search and Feature Selection in Recommender Fairness Testing**.

The study investigates how categorical subgroup search and feature projection affect the utility disparities an audit can detect. It evaluates established search algorithms; it does not propose a new fairness-testing algorithm.

## Study scope

- Methods: categorical GA, ACO, random-key DE, ABC, SA, native FairRec DPSO, and uniform random sampling. Exact enumeration supplies real-data references.
- Synthetic experiments: 945 searches across three landscape families, three instances, three category encodings, and five search seeds.
- MovieLens-1M/NeuMF: 420 searches across three frozen checkpoints and selected/full audit grammars.
- LFM360K and Amazon/frozen DeepFM: 280 searches, 24 selection/support cases, and four fixed-rule confirmations.
- Feature selection: backward elimination from the complete three-attribute audit schema, using development-only labels. Inactive legacy filter settings are disclosed in the reproduction notes; no mRMR/Chi-square fusion benefit is claimed.

Amazon uses behavioral history/price profiles rather than protected demographic attributes. All extensions are post hoc. Search repeats are not independent datasets or training replications. Dataset-scope decisions and the original exploratory records are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Getting started

Python 3.10 or newer is required for the modern code. The original legacy checkpoint exporter runs separately under Python 3.8/TensorFlow 2.3/DeepCTR 0.9.3.

```bash
python -m venv .venv
# Windows: .venv/Scripts/activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-recommender.txt -r requirements-publication.txt
```

Run a small portable synthetic demonstration without downloading data or loading a model:

```bash
python compare_fairrec_search.py --runs 2 --budget 200 --population 10 --out outputs/demo_release
python -m unittest discover -s tests -p test_search_comparison.py -v
python -m unittest discover -s tests -p test_revision_extension.py -v
```

The standalone demonstration includes an independent thermal DPSO reference. The manuscript's native DPSO results use the upstream FairRec implementation, whose source hash is recorded. Do not interpret the portable demonstration as a native reproduction or as a real recommender experiment.

## Rebuild and inspect the recorded evidence

The repository includes compressed request ledgers, exact subgroup references, configurations, and aggregate results needed by the current revision-asset builder:

```bash
python draft/build_revision_assets.py
```

This validates 670 added searches and 2,680,000 requests within the current manuscript scope, checks every real-data score against exact references, and regenerates the revised tables. It performs no training or model inference. Derived summaries go to `outputs/manuscript_scope`; the source records remain unchanged. The existing earlier tables and figure assets are also included with the manuscript.

Archived source hashes refer to the versions that produced the recorded runs. Historical source snapshots and original protocol files are retained. Some provenance records contain original machine-specific paths; these identify historical inputs and are not portable runtime paths. A new run needs local input paths and a fresh output directory, as explained in the reproduction guide. Existing evidence manifests may mention external per-user inputs that are intentionally not bundled; [RELEASE_CONTENTS.md](RELEASE_CONTENTS.md) identifies the release boundary.

## Run new experiments

The [configuration reference](configs/manuscript_protocol.json) lists the algorithm, seed, budget, ranking, and selector settings. It documents the protocol; the runners accept command-line arguments rather than a `--config` flag. [Runtime path examples](configs/runtime_paths.example.json) show the required local input layout.

The full real-model experiments require data, saved checkpoints, the relevant user partitions, and the upstream native source. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for export and execution commands. The external runner defaults to LFM360K and Amazon:

```bash
python run_revision_extension.py --scope external --datasets last-fm amazon --exports outputs/reproduced_exports --out outputs/reproduced_external
```

Raw datasets, saved model weights, and per-user utility tables are not included. Obtain the original processed datasets and checkpoints through the [upstream FairRec project](https://github.com/Grey-z/FairRec), or supply compatible inputs and disclose the changed protocol. The original Amazon category/release and checkpoint training histories could not be independently reconstructed.

## Paper

- [Compiled manuscript](draft/CATEGORICAL_FAIRNESS_TESTING.pdf)
- [LaTeX source](draft/CATEGORICAL_FAIRNESS_TESTING.tex)
- [Detailed reproduction guide](REPRODUCIBILITY.md)

Build with an installed TeX distribution, from `draft/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex
bibtex CATEGORICAL_FAIRNESS_TESTING
pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex
pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex
```

## Interpretation and attribution

Exact and random baselines limit claims for adaptive optimization in the tractable MovieLens grammar. Projection can reduce observed disparity signals, but these results concern the evaluated wrappers, labels, schemas, and ranking protocols. Pointwise confirmation intervals do not establish population discrimination or causal harm. Search timing excludes model training and inference, and timings from different scoring implementations are not pooled.

FairRec and FairAS are upstream dependencies with separate attribution and input requirements. They are not new algorithms introduced by this project. Third-party data and weights must be obtained under their respective terms; they are not relicensed by this repository. Citation metadata are provided in [CITATION.cff](CITATION.cff).
