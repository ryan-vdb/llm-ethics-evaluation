# Ethical-consistency analysis

This package tests whether six model responses share a reproducible geometric
organization across ethically different, topically unrelated scenarios.
Clustering is included as an exploratory description; held-out
representational similarity and null testing provide the primary evidence.

## Start here

- **Human-readable findings:** [`results/REPORT.md`](results/REPORT.md)
- **Machine-readable summary:** [`results/summary.json`](results/summary.json)
- **Method implementations:** [`methods/`](methods/)
- **Shared preprocessing and statistics:** [`tools/`](tools/)

## Package layout

```text
consistency/
├── README.md
├── __main__.py
├── runner.py
├── methods/
│   ├── clustering.py
│   ├── representational_similarity.py
│   ├── dyadic_regression.py
│   ├── kernel_alignment.py
│   ├── cross_domain_neighbors.py
│   ├── shared_latent_axes.py
│   ├── interpretable_reasoning_topics.py
│   ├── projection_artifact_null.py
│   └── robustness_checks.py
├── tools/
│   ├── data.py
│   ├── geometry.py
│   ├── statistics.py
│   └── output.py
└── results/
    ├── REPORT.md
    ├── summary.json
    ├── manifest.json
    ├── methods/
    └── tables/
```

## Shared preprocessing

[`tools/data.py`](tools/data.py) loads the 93 consistency questions and all six
aligned model views. [`tools/geometry.py`](tools/geometry.py) applies exact
question–answer orthogonalization:

```text
r_perp = r - ((r dot q) / (q dot q)) q
```

Every residual is L2-normalized. Loading fails unless the maximum absolute
paired residual–question cosine is below `1e-10`.

The primary cross-topic analysis uses pairs that:

- have different declared domains;
- come from different sources; and
- are in the bottom quartile of question-embedding cosine similarity.

Remaining question cosine is also controlled continuously in the primary RSA.

## What each method contributes

| Method | Role | Topic-controlled? |
|---|---|---|
| Representational similarity | Primary held-out geometry test with node permutations and bootstrap intervals | Yes |
| Projection-artifact null | Tests whether re-pairing plus the common projection operation creates the result | Yes |
| Robustness checks | Full question-span removal, centroid removal, source omission, and outlier omission | Yes |
| Cross-domain neighbors | Tests whether local cross-topic relationships transfer to held-out models | Yes |
| Clustering | Describes candidate partitions and tests their stability across model views | Diagnostics included |
| MRQAP | Regression-style effect size and incremental R² sensitivity | Yes; p-values exploratory |
| CKA | Secondary whole-geometry agreement | No |
| Shared PCA axes | Post-hoc low-dimensional interpretation | No |
| Answer-only NMF | Post-hoc reasoning-language interpretation | Partial topic control; not independent |

Clustering uses exact residuals from the canonical loader. It scans a small
candidate range, selects only through response-geometry stability, and reports
topic association afterward. A stable partition can still be a cut through a
continuous manifold rather than a natural ethical type.

## Run

Install the root requirements, then run from the repository root:

```bash
python3 -m src.analysis.consistency
```

The canonical run uses 9,999 question-node permutations, 999
re-pair-and-reproject null draws, and 2,000 node-bootstrap samples.

For a fast structural check:

```bash
python3 -m src.analysis.consistency \
  --output /tmp/ethics-consistency-smoke \
  --permutations 99 \
  --artifact-permutations 99 \
  --bootstrap-samples 100
```

The runner writes `REPORT.md`, a full `summary.json`, per-method JSON under
`methods/`, compact CSVs under `tables/`, and `manifest.json` last. The manifest
contains hashes for every generated artifact. Publication replaces this
dedicated result tree as one snapshot. For safety, a non-empty custom
`--output` directory is replaced only when its manifest identifies it as a
snapshot previously created by this runner; broad paths such as the repository
root, home directory, or system temporary root are rejected.

## Interpretation boundary

The analysis can establish reproducible geometry in this fixed six-model panel.
It cannot by itself establish a normative ethical framework, discrete moral
theories, or generalization to all language models. Cluster names, PCA axes,
NMF topics, and stable-pair interpretations remain post hoc and should be
independently human-coded before publication.
