Conference evidence extension
=============================

This is an explicitly post hoc extension of outputs/final_benchmark. It reuses
the three recorded NeuMF checkpoints and the original development/audit split;
it does not retrain models or overwrite the original benchmark.

Run from the project root:
  .venv/Scripts/python.exe run_conference_extension.py
  .venv/Scripts/python.exe draft/build_conference_assets.py

protocol.json locks code, baseline configuration, native-source hashes, seeds,
budgets, and timing scope. Completed runs resume only under that same protocol.
All requested searches must finish before all_complete.json is written.

Empirical controls: 3 checkpoints x 2 grammars x 10 search seeds x 5 methods
= 300 runs, each with 4,000 requests. Methods are native DPSO, GA, ACO, DE, and
uniform random sampling. Exact reference scores every possible cell once and
stops at space exhaustion; its request count is not inflated to the search cap.
Selected grammar has 7/14 cells; full gender/age/occupation grammar has 294 cells.
The support threshold is 20 and the validation utility-gap threshold is 0.02.

Synthetic controls: 3 families x 3 landscape seeds x 3 encodings x 5 search seeds
x 5 methods = 675 runs, each with 4,000 requests. Families are additive category
scores, NK/trap interactions, and disjoint modular triples. All cells are valid.
New category codes map to the same semantic score within each instance.
Encoded initial candidates, not semantically identical starts, are shared across
encodings. Encoding spread cannot isolate category arithmetic from initialization.
Synthetic threshold 0.65 is interpreted separately within each score family.

Outputs:
  empirical_results.csv / synthetic_results.csv: per-run endpoints.
  model_*/utilities.npz + JSON: verified cached per-audit-user validation/test
    NDCG@10 and inference/checkpoint/split provenance.
  model_*/*_exact.csv: all cells, including invalid cells and semantic rules.
  model_*/.../result.json + trace.jsonl.gz: committed result and every request.
  synthetic/.../encoding.json: new-code-to-semantic mappings.
  selection_ablation.csv: threshold attenuation after projecting full cells.
  test_confirmation.csv: one fixed exhaustive validation maximizer per grammar;
    enumeration order breaks ties, and test outcomes never select the rule.
  empirical_summary.csv / synthetic_summary.csv / confirmation_intervals.csv:
    generated only by the asset builder from completed records.

Timings describe cached objective scoring/search/in-memory recording, excluding
model inference and compressed output. They are not pooled with earlier runs
that include companion behavioral measurements. Runtime hardware: Intel Core
i7-11370H, 4 physical/8 logical cores; CPU inference uses 4 PyTorch threads.

Interpretation boundaries:
  One dataset and model architecture; three checkpoints are not three datasets.
  Encodings/repeated searches are not independent landscape instances.
  Selection labels encode activity, not measured unfairness.
  Projected-parent retention measures aggregation, not same-grammar optimizer regret.
  Different held-out interactions share the same audit users and histories.
  Bootstrap intervals are exploratory fixed-rule summaries and can be unreliable
    for sparse zero-hit cells. Conservative bounded-outcome intervals accompany
    them; neither adjusts for adaptive selection or multiplicity.
  No causal discrimination, mitigation effectiveness, or population significance
    follows from descriptive validation gaps alone.
