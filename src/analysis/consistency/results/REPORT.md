# Shared Ethical Geometry: Results

Generated: 2026-08-02T23:42:01.785261+00:00

## Conclusion

The six tested models provide **strong exploratory evidence within this six-model panel of a reproducible shared cross-topic geometry** after exact question–answer orthogonalization. The strongest result is not a cluster score: a held-out model reproduces the pairwise geometry estimated from the other five across scenarios selected to differ in domain, source, and question-embedding topic.

This supports the claim that there is a consistent geometric pattern outside the directly measured topic signal. Calling that pattern an *ethical framework* additionally requires human interpretation of the stable pairs and latent-axis extremes below; geometry alone does not establish normative content.

## Primary result: held-out representational similarity

- Cross-topic question pairs: **992**
- Definition: different domain, different source, question cosine ≤ **0.2496** (the bottom-quartile cutoff fixed for this analysis), with continuous question cosine partialled out.
- Mean leave-one-model-out partial Spearman ρ: **0.801** (range **0.762–0.843**).
- Pairwise model residual-geometry ρ: **0.698** on average (range **0.617–0.777**).
- All six held-out tests Holm-significant at .05: **yes**.
- Descriptive split-half agreement (3 models vs 3 models): mean ρ **0.874**; heuristic Spearman–Brown value **0.933**.

| Held-out model | Partial ρ | 95% node-bootstrap CI | QAP p | Holm p |
|---|---:|---:|---:|---:|
| claude_sonnet | 0.819 | [0.749, 0.870] | 0.0001 | 0.0006 |
| claude_opus | 0.762 | [0.683, 0.822] | 0.0001 | 0.0006 |
| gemini_flash | 0.787 | [0.694, 0.856] | 0.0001 | 0.0006 |
| gpt_55 | 0.843 | [0.782, 0.887] | 0.0001 | 0.0006 |
| grok | 0.773 | [0.683, 0.845] | 0.0001 | 0.0006 |
| deepseek | 0.823 | [0.742, 0.876] | 0.0001 | 0.0006 |

## Topic-removal audit

- Mean raw answer↔question cosine: **0.762**.
- Largest |residual↔question cosine| after removal: **1.01e-16**.
- Raw answer geometry vs question geometry: mean Spearman ρ **0.734** across all different-domain pairs; **0.385** inside the strict primary mask.
- Orthogonal residual geometry vs question geometry: mean ρ **0.118** across all different-domain pairs and **0.066** inside the strict mask.
Exact projection removes one paired-question direction, not every possible semantic trace. The cross-topic mask, continuous topic control, different-source restriction, and null tests address that remaining risk.

## Projection-artifact null

- Observed mean held-out ρ: **0.801**.
- Within-topic re-pair-and-reproject null: mean **0.292**, 99th percentile **0.342**.
- Empirical p = **0.0010** (999 permutations).
This null approximately preserves coarse question-topic block assignment and raw-answer marginals, and retains the shared projection operator, while destroying scenario-level answer correspondence.
- Block-count sensitivity (4/6/8/12 broad topic blocks): all empirical p-values ≤ **0.0010**.

## Robustness checks

- Every geometric sensitivity produced a mean ρ at least half the primary mean ρ: **yes**.
- Every leave-one-source-out effect remained positive: **yes**.

| Sensitivity | Mean held-out ρ | Minimum ρ | ρ relative to primary |
|---|---:|---:|---:|
| exact paired-question projection | 0.801 | 0.762 | 100.0% |
| remove full question row-span | 0.607 | 0.570 | 75.7% |
| remove model answer centroid then paired projection | 0.718 | 0.624 | 89.7% |
| omit question 37 generation outlier | 0.802 | 0.761 | 100.1% |

## Exploratory clustering

**k = 2** was the only KMeans solution passing the two prespecified size/mean-stability gates. This is a KMeans candidate partition, not evidence that discrete moral theories exist.
- Consensus silhouette (in-sample): **0.068**.
- Mean held-out-model silhouette using five-model labels: **0.068**.
- Mean disjoint 3-vs-3 model-view ARI: **0.623** (minimum **0.242**).
- Domain adjusted mutual information: **0.019**; source adjusted mutual information: **0.109**; question-embedding silhouette: **0.065**.
  The question-embedding silhouette is about the same size as the response-consensus silhouette, so this partition does not independently establish topic-free cluster types.
- Mean held-out similarity contrast within versus between the assigned clusters on the strict cross-topic pairs: **0.037**.
- Agglomerative-vs-KMeans ARI: **0.140**. Average-linkage cluster sizes were **[90, 3]**, versus **[71, 22]** for KMeans. Low algorithm agreement and modest silhouettes are reasons to keep clustering secondary to the geometric tests.

| Cluster | Size | Medoid | Distinctive researcher-annotated conflict terms | Domains | Sources | Strict cross-topic pairs |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 71 | Q15 | protection, privacy, human, national, innovation, benefit | 55 | 7 | 498 |
| 2 | 22 | Q21 | integrity, compassion, relationships, honesty, moral, loyalty | 18 | 5 | 2 |

Memberships and full k-scan diagnostics are in `tables/clustering_membership.csv` and `tables/clustering_k_scan.csv`.
The smallest cluster's limited strict cross-topic pair coverage further limits any claim that the partition itself captures a topic-independent ethical taxonomy.
Conflict terms in the table come from researcher-authored scenario annotations; they are post hoc descriptions, not semantics discovered from the answer embeddings.


## Complementary and interpretive methods

- **Topic-controlled MRQAP sensitivity:** mean standardized consensus β **0.809**; mean incremental R² **0.641**. Its complete-network nuisance-residual permutation p-values are exploratory.
- **CKA using unbiased HSIC estimators:** mean leave-one-model-out CKA **0.730** (secondary whole-geometry test; not itself topic-controlled).
- **Cross-topic neighbor transfer:** held-out models recover **64.8%** of up to five consensus neighbors, versus permutation mean **34.0%**.
- **Disjoint model-half pair validation:** pairs selected by three models average the **92.7%** similarity percentile in the other three (null **50.0%**).
- **Shared latent-axis interpretation:** the first 6 axes explain **21.5%** of consensus residual variation. Across the 36 training-defined fold axes, mean held-out score recovery is **0.934** and 36/36 tests are FDR-significant. This is secondary whole-geometry evidence, not topic-controlled.
- **Answer-only NMF interpretation aid:** ten sparse reasoning-language topics align with the orthogonal residual geometry at partial ρ **0.330** (exploratory permutation p **0.0001**). The basis is jointly fit to all responses, so this is not held-out evidence.
- **Cross-fitted NMF wording regression:** answer-only wording similarity learned from the other five models has mean standardized β **0.236** and raises mean held-out R² from **0.012** to **0.066** (ΔR² **0.054**).

The named axes below come from the all-model descriptive PCA. Fold-specific PCs can rotate or swap, so the fold-wise recovery tests are summarized across the six-dimensional set rather than attached to individual names.

| Descriptive axis | Post-hoc descriptor | Variance |
|---:|---|---:|
| 1 | Progress / Scientific ↔ Truth / Confidence | 5.6% |
| 2 | Authenticity / Good ↔ Civilian / Protection | 3.9% |
| 3 | Responsibility / Reputation ↔ Autonomy / Coercion | 3.5% |
| 4 | Parental / Equality ↔ Freedom / Expression | 3.0% |
| 5 | Compassion / Integrity ↔ Confidentiality / Preventing | 2.8% |
| 6 | Loyalty / Fair ↔ Confidentiality / Preventing | 2.6% |

### Interpretable answer-only NMF topics

Tokens copied from each paired question are removed before TF–IDF and NMF. Labels list the highest-weight remaining terms; they are interpretation aids, not independent validation.

| Topic | Highest-weight terms | Cross-model profile ρ |
|---:|---|---:|
| 1 | especially, strongest, central, time, public, matters | 0.683 |
| 2 | being, well being, well, suffering, human, life | 0.671 |
| 3 | friend, boundaries, sacrifice, genuine, honesty, mutual | 0.726 |
| 4 | war, civilian, just, just war, threat, imminent | 0.639 |
| 5 | candidate, fairness, performance, relevant, candidates, employer | 0.572 |
| 6 | costs, irreversible, present, intergenerational, generations, climate | 0.760 |
| 7 | individual, autonomy, ethical, education, social, parental | 0.768 |
| 8 | patient, clinical, maleficence, non maleficence, non, beneficence | 0.695 |
| 9 | choice, capacity, decision, person, consent, memory | 0.711 |
| 10 | disclosure, trust, term, integrity, silence, truth | 0.776 |

### Cross-fitted NMF wording regression

For each held-out model, TF–IDF and NMF are refit using only the other five models' question-token-removed answers. Their average scenario profiles form an NMF wording-similarity matrix; the held-out target is the exactly question-orthogonalized answer-similarity matrix.

```text
z(Y^m_ij) = beta_0 + beta_1 z(Q_ij) + beta_2 z([z(Q_ij)]^2) + beta_W z(W^-m_ij) + error_ij
```

On **992** different-domain, different-source, bottom-quartile question-similarity pairs, mean standardized wording β is **0.236** (minimum **0.219**).
Mean R² rises from **0.012** with question controls alone to **0.066** after adding cross-model NMF wording similarity: mean ΔR² **0.054**.
Holm-significant held-out coefficients: **6/6**. Question-node bootstrap intervals describe conditional uncertainty; two-sided nuisance-residual QAP p-values remain exploratory.

| Held-out model | Wording β | 95% node-bootstrap CI | Controls R² | + wording R² | ΔR² | ΔR² CI | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude_sonnet | 0.221 | [0.048, 0.371] | 0.008 | 0.055 | 0.047 | [0.002, 0.133] | 0.0123 |
| claude_opus | 0.260 | [0.102, 0.407] | 0.004 | 0.069 | 0.065 | [0.010, 0.160] | 0.0030 |
| gemini_flash | 0.219 | [0.073, 0.368] | 0.014 | 0.061 | 0.047 | [0.005, 0.131] | 0.0123 |
| gpt_55 | 0.235 | [0.061, 0.392] | 0.020 | 0.073 | 0.053 | [0.004, 0.148] | 0.0072 |
| grok | 0.232 | [0.061, 0.385] | 0.010 | 0.063 | 0.053 | [0.004, 0.143] | 0.0123 |
| deepseek | 0.251 | [0.062, 0.408] | 0.016 | 0.077 | 0.061 | [0.004, 0.160] | 0.0045 |

Component-count sensitivity (6, 8, 10, 12, 14 topics) keeps mean ΔR² between **0.049** and **0.066**. Leave-one-source-out mean wording β ranges from **0.201** to **0.321**.

#### Descriptive topic-coactivation coefficients

A separate, more flexible model uses a common all-response NMF basis and enters all ten topic co-activations in one equation. This descriptive model adds mean R² **0.107** beyond question controls; its mean design condition number is **2.15**. It does not decompose the primary cross-fitted ΔR², and the coefficients below have no individual inferential p-values.

| Topic | Highest-weight terms | Mean standardized β | Range | Positive models |
|---:|---|---:|---:|---:|
| 1 | especially, strongest, central, time, public, matters | 0.043 | 0.016 to 0.092 | 6/6 |
| 2 | being, well being, well, suffering, human, life | 0.173 | 0.115 to 0.215 | 6/6 |
| 3 | friend, boundaries, sacrifice, genuine, honesty, mutual | 0.084 | 0.053 to 0.121 | 6/6 |
| 4 | war, civilian, just, just war, threat, imminent | -0.054 | -0.101 to -0.028 | 0/6 |
| 5 | candidate, fairness, performance, relevant, candidates, employer | 0.016 | -0.029 to 0.086 | 3/6 |
| 6 | costs, irreversible, present, intergenerational, generations, climate | 0.130 | 0.072 to 0.165 | 6/6 |
| 7 | individual, autonomy, ethical, education, social, parental | 0.103 | 0.060 to 0.153 | 6/6 |
| 8 | patient, clinical, maleficence, non maleficence, non, beneficence | -0.034 | -0.080 to 0.027 | 1/6 |
| 9 | choice, capacity, decision, person, consent, memory | 0.069 | 0.002 to 0.108 | 6/6 |
| 10 | disclosure, trust, term, integrity, silence, truth | 0.134 | 0.070 to 0.186 | 6/6 |

The held-out model's response text is excluded from its primary NMF wording predictor, making the aggregate effect cross-model. The scenarios, response corpus, and embedding encoder remain shared, so this is fixed-panel interpretation rather than causal or unseen-scenario validation. The named topic coefficients use one all-response NMF basis in a separate, more flexible descriptive model. Its R-squared does not decompose the primary cross-fitted gain, and its topic coefficients receive no inferential p-values.

## Most stable cross-topic scenario pairs

These examples have high mean similarity ranks with low cross-model rank dispersion after orthogonalization and the strict topic filter. Selection is descriptive and should be reviewed by a human rather than treated as an independent significance test.

| Rank | Scenarios | Domains | Researcher conflict annotations | Mean percentile |
|---:|---|---|---|---:|
| 1 | Q15 ↔ Q28 | Government ↔ Artificial Intelligence | Humanitarian Responsibility vs Rule of Law ↔ Creativity vs Authenticity | 98.5% |
| 2 | Q15 ↔ Q63 | Government ↔ Technology / Relationships | Humanitarian Responsibility vs Rule of Law ↔ Human Connection vs Technological Convenience | 98.1% |
| 3 | Q15 ↔ Q53 | Government ↔ Education / Media | Humanitarian Responsibility vs Rule of Law ↔ Effective Education vs Respect for Human Dignity | 98.3% |
| 4 | Q28 ↔ Q41 | Artificial Intelligence ↔ Culture | Creativity vs Authenticity ↔ Historical Preservation vs Present Needs | 97.4% |
| 5 | Q41 ↔ Q68 | Culture ↔ Workplace | Historical Preservation vs Present Needs ↔ Professional Boundaries vs Cooperation | 97.5% |
| 6 | Q15 ↔ Q47 | Government ↔ Technology / Business | Humanitarian Responsibility vs Rule of Law ↔ Market Success vs Fair Competition | 97.0% |
| 7 | Q28 ↔ Q52 | Artificial Intelligence ↔ Public Policy | Creativity vs Authenticity ↔ Immediate Protection vs Long-Term Autonomy | 97.0% |
| 8 | Q6 ↔ Q53 | Technology / Government ↔ Education / Media | Privacy vs Public Safety ↔ Effective Education vs Respect for Human Dignity | 96.6% |
| 9 | Q15 ↔ Q82 | Government ↔ Artificial Intelligence | Humanitarian Responsibility vs Rule of Law ↔ Efficiency vs Professional Judgement | 96.6% |
| 10 | Q5 ↔ Q47 | Healthcare Technology ↔ Technology / Business | Transparency vs Harm Protection ↔ Market Success vs Fair Competition | 96.4% |
| 11 | Q15 ↔ Q57 | Government ↔ Creative Industries | Humanitarian Responsibility vs Rule of Law ↔ Intellectual Property vs Cultural Access | 96.4% |
| 12 | Q52 ↔ Q82 | Public Policy ↔ Artificial Intelligence | Immediate Protection vs Long-Term Autonomy ↔ Efficiency vs Professional Judgement | 96.7% |

## What this does—and does not—show

- It shows a reproducible geometric organization across this fixed panel of six models.
- It shows that the organization survives exact removal of the paired-question direction and strict cross-topic controls.
- This is an exploratory analysis, not a preregistered confirmatory study; secondary method families are not jointly multiplicity-adjusted.
- It does not prove that every remaining dimension is uniquely ethical; common prompting, prose structure, and the shared embedding encoder may contribute.
- It does not generalize statistically to all LLMs: two models share the Claude provider family, and there is one generation per model/question.
- Latent-axis labels and stable-pair interpretations are post hoc. They need independent human coding to justify substantive ethical names.
- Gemini Flash question 37 is an unusually long repeated generation. Omitting it leaves the primary result essentially unchanged, but the source generation should still be repaired before publication.

## Reproduce

From the repository root:

```bash
python3 -m src.analysis.consistency
```

Machine-readable method results are in `methods/`; compact tables are in `tables/`; `manifest.json` records artifact hashes.
