ICSCA 2027 manuscript source
===========================

Main document: CATEGORICAL_FAIRNESS_TESTING.tex
Current PDF: CATEGORICAL_FAIRNESS_TESTING.pdf
Current length: 11 pages including references, using ACM sigconf formatting
from the LaTeX template linked by ICSCA 2027. See ICSCA_SUBMISSION_NOTES.txt.
The former PUBLICATION_DRAFT.tex/PDF names are retired; open the files above.
Bibliography: references.bib
Layout class: acmart (installed with the TeX distribution)
Manuscript body: maintained directly in the main document
Included figures, tables, and provenance: assets/*

Build from this directory:
  pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex
  bibtex CATEGORICAL_FAIRNESS_TESTING
  pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex
  pdflatex -interaction=nonstopmode -halt-on-error CATEGORICAL_FAIRNESS_TESTING.tex

The source archive includes manuscript assets and the bibliography. It does not require
the original experiment folders to compile. A TeX installation with acmart,
ACM-Reference-Format.bst, Libertine fonts, amsmath, graphicx, booktabs,
textcomp, and the dependencies loaded by acmart is required.

To regenerate assets from the project workspace, run from the project root:
  .venv/Scripts/python.exe draft/build_publication_assets.py
  .venv/Scripts/python.exe draft/build_conference_assets.py
  .venv/Scripts/python.exe draft/build_revision_assets.py

Run the revision builder last: the current manuscript inputs revision_*.tex
tables and the seven-method attainment figure. REPRODUCIBILITY.md in the project
root gives exact experiment/export commands, environment details, provenance,
source snapshots, and input availability limits.

Do not run convert_publication_draft.py on this manuscript: that earlier
converter exports the older Markdown planning outline as PUBLICATION_DRAFT.tex;
it does not generate the current manuscript. The main document and section files
are maintained directly.

Evidence scope
--------------
Current revision: 945 synthetic searches (seven methods); 420 MovieLens searches;
280 supplementary frozen DeepFM searches on LFM360K and Amazon.
The five heuristic challengers are GA, ACO, DE, ABC, and SA, alongside native
DPSO and uniform random sampling. Exhaustive references are separate controls.
MovieLens additionally includes 60 target/cap/support cases; the new datasets
include 24 selection/support cases. All seven methods are evaluated in both
grammars of each retained dataset at primary support 20. Amazon uses behavioral
history/price profiles rather than protected demographics; the supplied extract
does not establish an original category/release. New checkpoints are supplied
frozen models, not fresh training replications. All candidate tables have one
positive and 49 negative rows per eligible user; these are sampled-candidate
audits, separate from MovieLens full-catalog ranking.

New records: outputs/revision_extension (ABC/SA additions),
outputs/revision_external_complete (archived additional dataset audits),
outputs/manuscript_scope (the current LFM360K/Amazon subset and validation), and
outputs/external_utilities_checked (frozen-score exports with data/checkpoint
hashes and verified model-variable restoration). revision_evidence.json hashes
the scoped records. Validation checks 2,680,000 requests across 670 added
searches, including every real-data score against exact references. Historical
exploratory records and explicit inheritance provenance remain available;
REPRODUCIBILITY.md records the post hoc dataset-scope decision.
Section III.B formulates the evaluated selector as backward elimination from
the complete audit schema. The archived top-ten mRMR and Chi-square hooks retain
all attributes and are disclosed only for reproduction of the saved runs; they
are not presented as active method components or attributed to a FairRec
feature-selection pipeline. The principal contribution is empirical evidence
about categorical search and the effect of feature selection on expressible
utility disparities. No new optimizer or universal superiority is claimed.

Earlier evidence, preserved unchanged:
The paper reports three searches per method on one synthetic landscape and a
MovieLens-1M/NeuMF comparison of native DPSO, GA, ACO, and DE across three paired
model/search seeds (42, 100, 2026), with 4,000 requests per method and seed.
Training completed 25, 18, and 13 epochs, retaining validation-selected checkpoints
from epochs 20, 13, and 8, respectively. The cap was 30 epochs with patience five.
The real-model record contains 12 successful algorithm runs and no failed seeds.
The conference extension is recorded separately in outputs/conference_extension:
300 real-model searches cross ten independent search seeds, three fixed
checkpoints, two grammars (selected and unfiltered), and five methods including
uniform random sampling. Exact enumeration scores the selected 7/14-cell spaces
and the full 294-cell demographic space (61 supported cells). GA/ACO/DE/random
find full-grammar optima in all 30 runs each; native DPSO succeeds in 28/30.
The selection ablation finds 14/18/15 supported above-threshold full intersections
but only 0/0/1 remain above threshold after projection to selected predicates.
The synthetic extension has 675 searches: three families, three seeded instances
per family, three encodings, five search seeds, and five methods. Each search
retains a 4,000-request budget. No new recommender training is performed.
These controls were designed after the original results and are explicitly
post hoc. Fixed-rule test confirmation is inconsistent; exploratory intervals
do not adjust for selection or multiplicity and do not replace independent-user
confirmation. conference_evidence.json hashes the extension records and code.
It does not establish real-model superiority
over DPSO, training convergence, or statistically significant population harms.
All displayed numerical results are traceable to the local recorded outputs.
Earlier reference metadata were verified through Crossref DOI records and arXiv
citation metadata. Crossref records are retained in assets/reference_metadata.json
and assets/additional_reference_metadata.json. The Li et al. entry also includes
the page range supplied by the author. The key zoralioglu2024dynamic retains
the requested identifier but records the verified publication year 2026.

The selected venue is ICSCA 2027. The manuscript retains the four-author list
and includes the supplied funding acknowledgement. The public submission page does not specify an
anonymity policy or separate artifact requirements. Any declarations in the
authenticated portal still need to be reviewed. No ethics approval or acceptance
is assumed. The manuscript has not been submitted; portal login is required.
Study repository: https://github.com/looppxx/categorical-fairness-search
Recorded evidence is in outputs/search_comparison_native and outputs/final_benchmark
in the project workspace. The latter includes configs, training summaries,
feature-selection records, search traces, and held-out test confirmation.
All four methods attain the same best observed validation gap within each seed
(mean 0.02216) in selected spaces of seven or fourteen cells. Activity-proxy
selection does not establish preservation of fairness signals. Test confirmation
uses different interactions of the same audit users; independent-user confirmation,
fairness-targeted selection ablations, richer search spaces, additional
datasets, and uncertainty analyses remain needed for stronger claims.

Run the extension from the project root with:
  .venv/Scripts/python.exe run_conference_extension.py
This verifies saved checkpoints and split provenance, resumes completed searches
only under an identical protocol, and preserves the original result folders.
