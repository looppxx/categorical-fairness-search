# Revised empirical study: reproduction and evidence

The manuscript is `draft/CATEGORICAL_FAIRNESS_TESTING.tex`. Its contribution is a controlled study of subgroup search and the effect of audit feature projection. It proposes no new optimization algorithm. Its three datasets are MovieLens-1M, LFM360K, and Amazon. All extensions are post hoc, and no additional recommender training was performed.

## Recorded evidence

| Evidence | Location | Scope |
| --- | --- | --- |
| Original model training and search | `outputs/final_benchmark` | Three NeuMF checkpoints, seeds 42/100/2026 |
| Earlier controlled searches | `outputs/conference_extension` | 675 synthetic and 300 MovieLens searches, five methods |
| Additional heuristics | `outputs/revision_extension` | 270 synthetic and 120 MovieLens searches, ABC/SA |
| MovieLens selection controls | `outputs/utility_support_extension` | Two targets × two caps plus unfiltered × four supports × three checkpoints = 60 cases |
| Additional-dataset scores | `outputs/external_utilities_checked` | One supplied frozen DeepFM checkpoint per dataset; one utility row per eligible user |
| Archived additional-dataset audit | `outputs/revision_external_complete` | Original multi-dataset records, preserved unchanged |
| Current manuscript dataset subset | `outputs/manuscript_scope` | 280 LFM360K/Amazon searches, 24 selection/support cases, four fixed-rule confirmations |
| Revised tables and evidence manifest | `draft/assets/revision_*.tex`, `revision_evidence.json` | Generated from recorded outputs; no training/search |

Dataset scope was finalized post hoc. BlackFriday was explored but excluded from the manuscript because its full six-attribute grammar has no supported discovery cell at the fixed primary support of 20. Its recorded searches and support-sensitivity outputs remain in the original archive. No support threshold or retained-dataset outcome was changed. The manuscript builder selects whole LFM360K/Amazon dataset blocks and writes derived summaries to a separate directory; it never rewrites the original multi-dataset tables. This is an editorial scope decision, not independent validation of dataset selection.

Historical run inheritance and source snapshots remain available in `outputs/revision_extension` and the archived external protocol. New-trace validation for the manuscript scope covers 670 searches and 2,680,000 requests: every real-data score is checked against its exact table and every 61st synthetic score is independently recomputed, alongside full request order, caching, invalidity, archive, and maximum checks. The current output is `outputs/manuscript_scope/trace_validation.json`; the earlier all-dataset validation remains archived. These checks validate recorded search behavior, not the historical training provenance of supplied DeepFM checkpoints.

## Environments and inputs

Modern experiments use the local `.venv`; versions are in `requirements-tested.txt` and `requirements-recommender.txt`, and the original protocol records include Python-library versions. Frozen legacy inference uses Python 3.8.0, TensorFlow 2.3.0, and DeepCTR 0.9.3 from the existing FairRec environment. The two environments remain separate. CPU: Intel i7-11370H; four intra-operation inference threads, one TensorFlow inter-operation thread; prediction batch 8,192.

The study repository is [categorical-fairness-search](https://github.com/looppxx/categorical-fairness-search). It provides the search implementations, configuration references, reproduction instructions, manuscript sources, and recorded search artifacts. Raw datasets, per-user utility tables, and saved model weights are external inputs. The upstream baseline and processed-data entry point is [FairRec](https://github.com/Grey-z/FairRec). The Amazon extract's original category/release cannot be established from the supplied files.

Required external inputs are `data/<dataset>/processed_data/test_data.csv` and `model/deepfm-<dataset>/deepfm-<dataset>.ckpt.*` under a local FairRec checkout. Manuscript dataset identifiers are `last-fm` and `amazon`. Checkpoint/data/source hashes are in each export's `metadata.json`. MovieLens requires the saved baseline checkpoints, user split, the recorded native `fairrec.py`, and raw MovieLens files referenced by `outputs/final_benchmark/config.json`. Absolute paths in the original configuration must be resolved to the corresponding local inputs on another machine. Hashes identify files; they do not distribute them.

The model restore asserts that every current model variable matches a checkpoint entry, while allowing unused optimizer slots. TensorFlow reports legacy object-graph alias warnings; no unmatched current model variable is allowed. Original checkpoint training seeds, epochs, preprocessing, and historical training/evaluation disjointness are not independently reconstructed.

## Frozen-score export

Use unused output directories. For example, from the project root in PowerShell:

```powershell
$legacyPython = 'C:/Users/DELL/Documents/repo/FairRec/fairrec-env/Scripts/python.exe'
$fairrecSource = 'C:/Users/DELL/Documents/repo/FairRec'
foreach ($datasetName in @('last-fm', 'amazon')) {
    & $legacyPython export_external_utilities.py --source $fairrecSource --dataset $datasetName --out "outputs/reproduced_exports/$datasetName"
    if ($LASTEXITCODE -ne 0) { throw "Export failed: $datasetName" }
}
```

The exporter verifies exactly one relevant item per user, 50 supplied candidate rows per user in these extracts, binary labels, finite scores, and consistent per-user audit attributes. It uses score descending, then smaller item ID first for ties. Candidate rows, including duplicates, remain in the supplied protocol; an equal-score negative with the same item ID is counted before the positive. NDCG@10 is `1/log2(rank+1)` for rank ≤10 and zero otherwise. These sampled-candidate utilities are not pooled with MovieLens full-catalog utilities.

LFM360K: 267,034 users, 13,351,700 rows; gender/age/country (2×111×239). Amazon: 175,878 users, 8,793,900 rows; history/mean-price/max-price levels (10×10×10). The Amazon audit is behavioral, not protected demographic fairness, and history construction may affect interpretation.

## Search and feature-selection settings

| Component | Setting |
| --- | --- |
| Budget | 4,000 requests; initialization, duplicates, rejected proposals, and unsupported cells count |
| Population / two ends | 20 initial candidates per end; 2,000 requests per end for challengers |
| Seeds | Model 42/100/2026 in MovieLens; search 1001–1010; synthetic search 1001–1005 |
| Shared initialization | NumPy SeedSequence `[search_seed, 0]`; challenger RNG `[search_seed, 1]`; native initializer retained |
| Native DPSO | Thermal 0.09; cognitive/social 2.0; velocity bounds −2/+2; integer conversion; bound clipping; native double-ended loop |
| GA | Tournament 3; uniform crossover 0.5/coordinate; mutation 1/d to a different category; one cached elite |
| ACO | Trails initialized 1; evaporation 0.2; best 6 valid ants, deposit 1/(rank+1); bounds 0.05–20; uniform mixture 0.1; constant heuristic |
| DE | DE/rand/1/bin random keys; F=0.7; CR=0.8; one forced donor key; clipping 0–1; category argmax decoder; nonworsening replacement |
| ABC | 20 employed + 20 rank-weighted onlooker attempts/cycle; one-coordinate donor proposals; strict improvement; scout after 10 failures |
| SA | Best initialized candidate per end; one-coordinate proposals; geometric temperature 0.15→0.001; Metropolis acceptance |
| Random | Shared starts followed by independent uniform category assignments; repeats count |
| Exact | Each grammar cell once; stop at space exhaustion; separate reference rather than a forced 4,000-request run |
| Primary real-data support / threshold | 20 users / signed global-minus-group NDCG gap ≥0.02 |
| Synthetic protocol | Three families × three instances (2026–2028) × three encodings × five seeds × seven methods =945 searches; threshold 0.65 |
| Evaluated real-data selector | Backward elimination from the complete audit schema; minimum one attribute; tolerance 0.001; three stratified shuffled CV folds; balanced accuracy; one-hot encoding + depth-five decision tree |

Archived implementation detail: the recorded code contains independent top-ten mRMR and Chi-square hooks followed by union. Each real-data schema retained in the manuscript has three attributes, so both hooks retain the entire schema and backward elimination performs every reduction. The manuscript consequently omits these identity operations from its real-data method formulation and records them only for reproduction of saved runs. This is a description of our archived experimental implementation, not a claim to reproduce an original FairRec feature-selection pipeline. No mRMR, Chi-square, or fusion effect is estimated. The scope revision changes dataset selection and adds an explicit dataset argument to the runner; it does not change any retained dataset's scoring or search operators.

MovieLens ablations use activity-below-development-median labels versus development-utility-deficit ≥0.02 labels, caps two/three, and supports 20/50/100/200. Each checkpoint seeds its selector. Additional datasets use development-utility-deficit labels, caps two/full-size, and supports 5/20/50/100. Caps are upper bounds, not fixed output sizes. Additional selectors/splits use seed 314159, with 30% development, 35% discovery, and the remainder confirmation; the first two counts are floored. The metadata vocabulary uses supplied attributes without utility labels. Development users alone choose features. No confirmation utility chooses labels, support, parameters, predicates, or maximizers.

There are 420 MovieLens searches. Additional datasets contribute 140 LFM360K and 140 Amazon searches. All seven methods have ten searches in each dataset's selected and full grammars. The two additional datasets contribute 24 target-cap/support cases and four fixed-rule confirmations. Confirmation fixes one exact discovery maximizer per grammar, with enumeration-order tie-breaking, before evaluating separate users. Bounds are pointwise conditional bounded-outcome intervals; they do not supply a familywise multiplicity guarantee or establish causal discrimination.

## Reproduce runs and assets

From the project root, using unused result directories:

```powershell
./.venv/Scripts/python.exe run_revision_extension.py --scope existing --out outputs/reproduced_revision
./.venv/Scripts/python.exe run_revision_extension.py --scope external --datasets last-fm amazon --exports outputs/reproduced_exports --out outputs/reproduced_external
./.venv/Scripts/python.exe draft/build_revision_assets.py --added outputs/reproduced_revision --external outputs/reproduced_external
```

The external runner defaults to the two additional manuscript datasets; its dataset list is recorded in the protocol. Earlier controls can be reproduced with `run_conference_extension.py` and `run_utility_support_extension.py` from saved baseline checkpoints, using a fresh `--out` if a source/configuration protocol differs. Building the revised tables requires the earlier conference and selection-control records too. Running older asset builders alone will not update the revision tables. The asset builder writes the scoped summaries to `outputs/manuscript_scope` by default; `--paper-out` selects a different derived-output directory.

To regenerate from the currently recorded outputs without executing searches:

```powershell
./.venv/Scripts/python.exe draft/build_revision_assets.py
```

To compile the paper:

```powershell
Set-Location draft
pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex
bibtex CATEGORICAL_FAIRNESS_TESTING
pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex
pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex
```

`tests/test_revision_extension.py` checks indexed-score equivalence, projection membership/retention, and charged invalid/repeated requests for the added heuristics. Existing search, conference, and utility-support tests cover the reused kernels and controls. Numerical equality of recorded objective scores does not make timings comparable across the original user-scanning and new indexed implementations; the manuscript does not pool those timings.
