# Integrity-response analysis

This package analyzes how six models revise ethical responses after six kinds
of social feedback. Each helper response branches independently from the same
initial answer, making model–scenario pairs the repeated-measures blocks.

## Start here

- **Human-readable findings:** [`results/REPORT.md`](results/REPORT.md)
- **Machine-readable summary:** [`results/summary.json`](results/summary.json)
- **Metadata-masked conclusion coding:** [`results/coding/README.md`](results/coding/README.md)
- **Method implementations:** [`methods/`](methods/)
- **Strict loading and shared statistics:** [`tools/`](tools/)

## Claim boundary

The automated outcome is whole-response **semantic revision**, not a direct
integrity or conclusion-reversal score. A response can move substantially in
embedding space because it adds arguments or explicitly resists pressure while
retaining the same recommendation. Direct claims about conclusion persistence
must use the supplied metadata-masked human-coding sheet or a separately validated
stance classifier.

## Package layout

```text
integrity/
├── README.md
├── __main__.py
├── runner.py
├── methods/
│   ├── revision_effects.py
│   ├── scenario_specificity.py
│   ├── consensus_movement.py
│   ├── lexical_robustness.py
│   ├── robustness_checks.py
│   └── exemplars.py
├── tools/
│   ├── data.py
│   ├── geometry.py
│   ├── statistics.py
│   └── output.py
├── tests/
└── results/
    ├── REPORT.md
    ├── summary.json
    ├── manifest.json
    ├── methods/
    ├── tables/
    └── coding/
```

## Design and preprocessing

The strict loader verifies the complete 6-model × 10-scenario × 7-condition
panel, exactly one response and embedding per cell, aligned question vectors,
finite equal-dimensional embeddings, and non-empty text. It discovers model
names and question IDs rather than assuming IDs begin at zero.

For every initial and follow-up response, the paired question direction is
removed exactly and the remaining vector is normalized:

```text
r = e - ((e dot q) / (q dot q)) q
z = r / ||r||
semantic revision = 1 - cosine(z_initial, z_followup)
```

The follow-up is not projected away from the initial response because retained
initial content is the outcome of interest. `agreement` is the active
second-turn control; it asks for reconsideration without disagreement. Pairing
within each model and scenario controls scenario main effects because all seven
responses in a block concern the same question. It does not prove that the
remaining geometry is a topic-free ethical framework or remove
condition-by-scenario heterogeneity.

## Methods

| Method | Role |
|---|---|
| Paired revision effects | Primary opposition-minus-agreement estimate, exact scenario tests, crossed bootstrap, condition contrasts, and heterogeneity |
| Scenario specificity | Describes matched-scenario retention and tests cross-model alignment of revision directions against question-label nulls |
| Peer-centroid geometry | Describes movement relative to unseen peer initial responses; it is not a conformity test |
| Lexical robustness | Question-token-removed TF-IDF sensitivity independent of the embedding encoder |
| Robustness checks | Raw, full-question-span, centered-pattern, angular, length, and leave-one-unit diagnostics |
| Exemplars and coding | Transparent extreme-pair review plus metadata-masked human stance-coding files |

The exact sign-flip tests use the ten scenarios as exchangeable sign units
under a symmetric null. The scenarios are a small fixed panel, not a random
population sample. The crossed bootstrap independently resamples models and
scenarios, and individual model–scenario cells are never treated as 60
independent observations.

## Run

From the repository root, using an environment with the root requirements:

```bash
python -m src.analysis.integrity
```

No API keys are required; the analysis reads the frozen DuckDB snapshot. The
canonical run uses 9,999 scenario permutations and 20,000 crossed-bootstrap
draws. For a structural smoke run:

```bash
python -m src.analysis.integrity \
  --output /tmp/integrity-smoke \
  --permutations 99 \
  --bootstrap-samples 100
```

Result publication replaces only a dedicated, runner-owned output directory as
one complete snapshot. Broad paths, non-owned non-empty directories, and
snapshots containing edited or added files are rejected. Copy the metadata-masked
coding template outside `results/` before entering ratings; this guard prevents
a rerun from erasing human work.

## Interpretation limits

This is an exploratory, post-hoc analysis of ten scenarios with one generation
per cell. There is no neutral repeat-generation control or substantive
counterargument control, helper order was fixed, and generation/encoder
provenance is incomplete. Claimed expert or lived experience can also be
ethically relevant evidence, so updating is not automatically a failure of
integrity. Human conclusion coding is the next required step for that stronger
claim.
