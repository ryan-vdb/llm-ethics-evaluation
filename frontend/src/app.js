import {
  bulletPlot,
  divergingBarPlot,
  escapeHtml,
  estimatePlot,
  forestPlot,
  formatNumber,
  formatPercent,
  formatSigned,
  horizontalBarPlot,
  modelLabel,
  pairedDotPlot,
  rSquaredComparisonPlot,
} from "./charts.js";

const app = document.querySelector("#app");

function icon(name, size = 18) {
  const paths = {
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    copy: '<rect x="9" y="9" width="10" height="10" rx="2"/><path d="M15 9V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"/>',
    data: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    moon: '<path d="M20.4 14.7A8.5 8.5 0 0 1 9.3 3.6 8.5 8.5 0 1 0 20.4 14.7Z"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    external: '<path d="M14 5h5v5M10 14 19 5"/><path d="M19 13v5a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5"/>',
  };
  return `<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${paths[name] || paths.arrow}</svg>`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en", { year: "numeric", month: "short", day: "numeric" }).format(new Date(value));
}

function metric(label, value, note = "", tone = "") {
  return `<div class="metric ${tone}"><span class="metric-label">${escapeHtml(label)}</span><strong>${value}</strong>${note ? `<small>${note}</small>` : ""}</div>`;
}

function sectionHeading(kicker, title, description) {
  return `<header class="section-heading reveal"><span class="eyebrow">${escapeHtml(kicker)}</span><h2>${title}</h2><p>${description}</p></header>`;
}

function evidenceTag(text, tone) {
  return `<span class="evidence-tag ${tone}"><i></i>${escapeHtml(text)}</span>`;
}

function renderHeader() {
  return `
    <header class="site-header">
      <a class="brand" href="#overview" aria-label="Ethical Geometry Atlas home">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
        <span><strong>Ethical Geometry</strong><small>Results atlas</small></span>
      </a>
      <nav class="site-nav" aria-label="Main navigation">
        <a href="#overview">Overview</a>
        <a href="#consistency">Shared geometry</a>
        <a href="#integrity">Feedback response</a>
        <a href="#process">Method</a>
        <a href="#reproduce">Reproduce</a>
      </nav>
      <button class="theme-toggle" type="button" aria-label="Toggle color theme" aria-pressed="false">
        <span class="theme-sun">${icon("sun")}</span><span class="theme-moon">${icon("moon")}</span>
      </button>
      <div class="sr-only" id="copy-status" role="status" aria-live="polite"></div>
    </header>`;
}

function renderHero(data) {
  const c = data.consistency.headline;
  const i = data.integrity.headline;
  const significantModels = data.consistency.heldOutModels.filter((row) => row.holm <= 0.05).length;
  const intervalIncludesZero = i.bootstrapLow <= 0 && i.bootstrapHigh >= 0;
  return `
    <section class="hero" id="overview">
      <div class="hero-backdrop" aria-hidden="true"><span></span><span></span><span></span><span></span></div>
      <div class="hero-copy reveal">
        <span class="eyebrow">LLM ethics evaluation · fixed ${data.panel.models.length}-model panel</span>
        <h1>Ethical structure<br/><em>and response</em> under pressure.</h1>
        <p class="hero-lede">${data.panel.models.length} models share a strong cross-topic response geometry. Their semantic revision under social feedback is smaller, qualified, and model-dependent.</p>
        <div class="hero-actions">
          <a class="button primary" href="#consistency">Explore the evidence ${icon("arrow")}</a>
          <a class="button secondary" href="#process">See how it works</a>
        </div>
      </div>
      <div class="finding-grid reveal">
        <article class="finding-card consistency-card">
          <div class="finding-top">${evidenceTag("Strong within this panel", "strong")}<span>Study 01</span></div>
          <p class="finding-number">${formatNumber(c.meanHeldOutRho)}</p>
          <h2>Shared geometry transfers to held-out models.</h2>
          <p>Mean partial Spearman ρ across ${c.crossTopicPairs.toLocaleString()} strict cross-topic pairs. ${significantModels} of ${data.consistency.heldOutModels.length} held-out tests survive Holm correction.</p>
          <a href="#consistency">View shared geometry ${icon("arrow", 16)}</a>
        </article>
        <article class="finding-card integrity-card">
          <div class="finding-top">${evidenceTag("Qualified · coding pending", "qualified")}<span>Study 02</span></div>
          <p class="finding-number">${formatSigned(i.extraRevision, 4)}</p>
          <h2>Pushback is associated with a small amount of extra semantic revision.</h2>
          <p>Opposition minus agreement control. Crossed 95% interval ${formatSigned(i.bootstrapLow, 3)} to ${formatSigned(i.bootstrapHigh, 3)} ${intervalIncludesZero ? "includes" : "excludes"} zero.</p>
          <a href="#integrity">View feedback response ${icon("arrow", 16)}</a>
        </article>
      </div>
      <div class="dataset-rail reveal" aria-label="Dataset summary">
        ${metric("Models", data.panel.models.length, "held fixed")}
        ${metric("Consistency scenarios", data.panel.consistencyQuestions, `${data.panel.consistencyResponses} initial answers`)}
        ${metric("Integrity scenarios", data.panel.integrityQuestions, `${data.panel.integrityResponses} response cells`)}
        ${metric("Embedding space", data.panel.embeddingDimensions.toLocaleString(), "dimensions")}
        ${metric("Question projection", "Exact", "then L2 normalized", "accent")}
      </div>
    </section>`;
}

function renderConsistency(data) {
  const c = data.consistency;
  const significantModelCount = c.heldOutModels.filter((row) => row.holm <= 0.05).length;
  const allModelsSignificant = significantModelCount === c.heldOutModels.length;
  const heldOutTitle = allModelsSignificant
    ? "A shared geometry reproduced across <em>every held-out model.</em>"
    : `Shared geometry was detected in <em>${significantModelCount} of ${c.heldOutModels.length} held-out models.</em>`;
  const topicRows = c.topicRemoval.byModel.map((row) => ({
    label: row.model,
    before: row.raw_spearman_rho,
    after: row.residual_spearman_rho,
  }));
  const robustness = c.robustness.map((row) => ({ label: row.check, value: row.mean }));
  const weakestRobustness = c.robustness.reduce((lowest, row) => row.mean < lowest.mean ? row : lowest);
  const mrqapBetaValues = c.mrqap.heldOutModels.map((row) => row.beta);
  const meanMrqapTopicR2 = c.mrqap.heldOutModels.reduce((total, row) => total + row.topicOnlyR2, 0) / c.mrqap.heldOutModels.length;
  const meanMrqapFullR2 = c.mrqap.heldOutModels.reduce((total, row) => total + row.fullR2, 0) / c.mrqap.heldOutModels.length;
  const mrqapSourceBetas = c.mrqap.leaveOneSourceOut.map((row) => row.meanBeta);
  const mrqapSignificantCount = c.mrqap.heldOutModels.filter((row) => row.holm <= 0.05).length;
  const maximumMrqapHolm = Math.max(...c.mrqap.heldOutModels.map((row) => row.holm));
  const mrqapDomainMask = c.mrqap.definition.differentExactDomain ? "different exact domains" : "all domain pairings";
  const mrqapSourceMask = c.mrqap.definition.differentSource ? "Different sources are required." : "Same-source status is modeled, not excluded.";
  const wording = c.wordingRegression;
  const wordingPlotRows = wording.heldOutModels.map((row) => ({
    model: row.model,
    topicOnlyR2: row.controlsOnlyR2,
    fullR2: row.fullR2,
    incrementalR2: row.incrementalR2,
    incrementalLow: row.incrementalLow,
    incrementalHigh: row.incrementalHigh,
    beta: row.beta,
    betaLow: row.betaLow,
    betaHigh: row.betaHigh,
    holm: row.holm,
  }));
  const wordingComponentDeltas = wording.componentSensitivity.map((row) => row.meanIncrementalR2);
  const wordingSourceBetas = wording.leaveOneSourceOut.map((row) => row.meanBeta);
  const wordingTopicRows = wording.attribution.topics.map((topic) => ({
    label: `T${topic.topic} · ${topic.terms.slice(0, 2).join(" / ")}`,
    value: topic.beta,
    highlight: topic.beta >= 0.1 && topic.positiveModels === data.panel.models.length,
    tooltip: `Topic ${topic.topic}: mean β ${formatSigned(topic.beta)}; range ${formatSigned(topic.minimumBeta)} to ${formatSigned(topic.maximumBeta)}; positive in ${topic.positiveModels}/${data.panel.models.length} held-out models`,
  }));
  const strongestWordingTopics = [...wording.attribution.topics].sort((a, b) => b.beta - a.beta).slice(0, 3);
  const mostNegativeWordingTopic = [...wording.attribution.topics].sort((a, b) => a.beta - b.beta)[0];
  const pairCards = c.stablePairs.map((pair) => `
    <details class="pair-card">
      <summary>
        <span class="pair-rank">${String(pair.rank).padStart(2, "0")}</span>
        <span class="pair-domains">${escapeHtml(pair.first.domain)} <i>↔</i> ${escapeHtml(pair.second.domain)}</span>
        <strong>${formatPercent(pair.percentile)}</strong>
      </summary>
      <div class="pair-body">
        <div><span>Q${pair.first.id}</span><h4>${escapeHtml(pair.first.conflict)}</h4><p>${escapeHtml(pair.first.question)}</p></div>
        <div><span>Q${pair.second.id}</span><h4>${escapeHtml(pair.second.conflict)}</h4><p>${escapeHtml(pair.second.question)}</p></div>
        <footer><span>Mean residual cosine <strong>${formatNumber(pair.cosine)}</strong></span><span>Cross-model rank dispersion <strong>${formatNumber(pair.rankDispersion)}</strong></span></footer>
      </div>
    </details>`).join("");
  const axes = c.latentAxes.map((axis) => `
    <details class="axis-card">
      <summary>
        <span class="axis-index">0${axis.axis}</span>
        <span><strong>${escapeHtml(axis.label)}</strong><small>${formatPercent(axis.variance)} of residual variation</small></span>
        <i class="axis-meter"><b style="--value:${axis.variance / 0.06}"></b></i>
      </summary>
      <div class="axis-detail">
        <p class="caveat-inline">Post-hoc descriptor; axis signs and identities are snapshot-local.</p>
        <div class="axis-poles"><span>${axis.lowConcepts.map(escapeHtml).join(" · ")}</span><i>↔</i><span>${axis.highConcepts.map(escapeHtml).join(" · ")}</span></div>
        <div class="axis-examples">
          <div><small>Low-pole examples</small>${axis.lowExamples.map((item) => `<p><b>Q${item.id}</b> ${escapeHtml(item.conflict)}</p>`).join("")}</div>
          <div><small>High-pole examples</small>${axis.highExamples.map((item) => `<p><b>Q${item.id}</b> ${escapeHtml(item.conflict)}</p>`).join("")}</div>
        </div>
      </div>
    </details>`).join("");
  const topics = c.topics.map((topic) => `
    <article class="topic-card"><span>Topic ${topic.topic}</span><h4>${topic.terms.slice(0, 3).map(escapeHtml).join(" · ")}</h4><div class="term-cloud">${topic.terms.map((term) => `<i>${escapeHtml(term)}</i>`).join("")}</div><p>Cross-model profile ρ <strong>${formatNumber(topic.profileRho)}</strong></p></article>`).join("");
  const clusterTotal = c.clustering.profiles.reduce((total, row) => total + row.size, 0);
  const clusterProfileCards = c.clustering.profiles.map((row) => {
    const exemplar = row.exemplars[0];
    return `<article class="cluster-profile-card">
      <header><span>Candidate group ${row.cluster}</span><strong>${row.size} of ${clusterTotal} scenarios</strong></header>
      <div class="cluster-share" aria-label="${formatPercent(row.size / clusterTotal)} of scenarios"><i style="--value:${row.size / clusterTotal}"></i></div>
      <p class="cluster-term-label">Researcher-annotation terms appearing more often here</p>
      <h4>${row.terms.slice(0, 4).map(escapeHtml).join(" · ")}</h4>
      <div class="cluster-facts"><span>${row.domains} domains</span><span>${row.sources} sources</span><span>${row.strictPairs} strict cross-topic pairs</span></div>
      ${exemplar ? `<div class="cluster-medoid"><span>Representative scenario · Q${exemplar.id} · ${escapeHtml(exemplar.domain)}</span><strong>${escapeHtml(exemplar.conflict)}</strong><details><summary>Read scenario</summary><p>${escapeHtml(exemplar.question)}</p></details></div>` : ""}
      ${row.strictPairs < 30 ? `<p class="cluster-coverage-note">This group has limited strict within-group cross-topic coverage, so its topic-independent interpretation remains tentative.</p>` : ""}
    </article>`;
  }).join("");
  const orderedClusters = [...c.clustering.profiles].sort((a, b) => b.size - a.size);
  const clusterMeaning = orderedClusters.length === 2
    ? `The larger profile is marked by ${orderedClusters[0].terms.slice(0, 3).join(", ")}; the smaller profile by ${orderedClusters[1].terms.slice(0, 3).join(", ")}.`
    : "The profiles expose recurring regions of the response geometry through their representative scenarios and annotated conflict terms.";
  const clusteringPanel = c.clustering.selectedK === null || c.clustering.metrics === null
    ? `<article class="panel clustering-panel reveal"><header><div><span class="panel-kicker">Exploratory clustering result</span><h3>No candidate partition met the declared stability rule</h3></div><span class="method-chip">K-Means</span></header><p>${escapeHtml(c.clustering.status)}</p></article>`
    : `<article class="panel clustering-panel reveal">
        <header><div><span class="panel-kicker">Exploratory clustering result</span><h3>A candidate ${c.clustering.selectedK}-group organization recurs across model views</h3></div><span class="method-chip">K-Means</span></header>
        <p class="clustering-lede">K-Means selected groups of ${c.clustering.metrics.cluster_sizes.join(" and ")} scenarios. ${escapeHtml(clusterMeaning)} Moderate cross-model and split-half agreement means this cut reappears across model views, while the separation remains soft rather than absolute.</p>
        <div class="clustering-diagnostics" aria-label="Clustering diagnostics">
          <div><span>Group sizes</span><strong>${c.clustering.metrics.cluster_sizes.join(" / ")}</strong></div>
          <div><span>Held-out partition ARI</span><strong>${formatNumber(c.clustering.metrics.mean_held_out_view_partition_ari)}</strong></div>
          <div><span>Mean split-half ARI</span><strong>${formatNumber(c.clustering.metrics.mean_split_half_ari)}</strong></div>
          <div><span>Minimum split-half ARI</span><strong>${formatNumber(c.clustering.metrics.minimum_split_half_ari)}</strong></div>
          <div><span>Response silhouette</span><strong>${formatNumber(c.clustering.metrics.consensus_silhouette_in_sample)}</strong></div>
          <div><span>Question-only silhouette</span><strong>${formatNumber(c.clustering.metrics.question_embedding_silhouette)}</strong></div>
          <div><span>Cross-topic contrast</span><strong>+${formatNumber(c.clustering.metrics.mean_strict_cross_topic_similarity_contrast)}</strong></div>
          <div><span>Domain / source AMI</span><strong>${formatNumber(c.clustering.metrics.domain_adjusted_mutual_information)} / ${formatNumber(c.clustering.metrics.source_adjusted_mutual_information)}</strong></div>
          <div><span>Alternative-algorithm ARI</span><strong>${formatNumber(c.clustering.metrics.agglomerative_vs_kmeans_ari)}</strong></div>
        </div>
        <div class="cluster-profile-grid">${clusterProfileCards}</div>
        <div class="clustering-reading">
          <div><span>What is meaningful here</span><p>The candidate split supplies an inspectable map of recurring response-geometry regions: group sizes, representative scenarios, and profile differences can now be compared directly. Held-out and mean split-half ARI of ${formatNumber(c.clustering.metrics.mean_held_out_view_partition_ari, 2)} and ${formatNumber(c.clustering.metrics.mean_split_half_ari, 2)} show that the same broad cut often reappears when model views change.</p></div>
          <div><span>How to interpret it</span><p>Response silhouette ${formatNumber(c.clustering.metrics.consensus_silhouette_in_sample)} is nearly identical to question-only silhouette ${formatNumber(c.clustering.metrics.question_embedding_silhouette)}, and one split-half ARI falls to ${formatNumber(c.clustering.metrics.minimum_split_half_ari)}. Together with low alternative-algorithm agreement, this indicates overlapping, non-unique boundaries and limits the topic-independent claim. The result is a useful candidate organization of this corpus, not exactly ${c.clustering.selectedK} fixed ethical theories.</p></div>
        </div>
        <p class="cluster-annotation-note">Profile terms come from researcher-authored conflict annotations and describe the selected groups after clustering; they were not labels discovered directly from answer embeddings.</p>
      </article>`;

  return `
    <section class="study-section consistency-section" id="consistency">
      <div class="section-shell">
        ${sectionHeading("Study 01 · Cross-topic consistency", heldOutTitle, `The primary test asks whether one model recreates the pairwise response geometry learned from the other ${data.panel.models.length - 1}—only across scenarios that differ in domain, source, and measured topic similarity.`)}
        <div class="finding-banner reveal">
          <div><span>Main estimate</span><strong>ρ ${formatNumber(c.headline.meanHeldOutRho)}</strong><small>range ${formatNumber(c.headline.minimumHeldOutRho)}–${formatNumber(Math.max(...c.heldOutModels.map((row) => row.rho)))}</small></div>
          <p><b>Interpretation:</b> there is strong exploratory evidence of reproducible organization outside the directly measured topic direction in this fixed panel. Geometry alone does not name a moral theory.</p>
          ${evidenceTag(`${significantModelCount} / ${c.heldOutModels.length} Holm-significant`, allModelsSignificant ? "strong" : "qualified")}
        </div>

        <div class="content-grid primary-grid reveal">
          <article class="panel chart-panel wide">
            <header><div><span class="panel-kicker">Primary result</span><h3>Held-out representational similarity</h3></div><span class="method-chip">partial Spearman ρ</span></header>
            ${forestPlot(c.heldOutModels, { reference: c.headline.meanHeldOutRho })}
            <footer class="chart-caption"><span>Whiskers: 95% question-node bootstrap intervals</span><span>Dyads are dependent—not n=${c.headline.crossTopicPairs.toLocaleString()} IID observations</span></footer>
          </article>
          <aside class="panel explanation-panel">
            <span class="panel-kicker">How to read this</span>
            <h3>Train on ${data.panel.models.length - 1}.<br/>Test on the held-out model.</h3>
            <ol class="numbered-list">
              <li><b>01</b><span>Average ${data.panel.models.length - 1} models' orthogonalized answer geometry.</span></li>
              <li><b>02</b><span>Hide the remaining model entirely.</span></li>
              <li><b>03</b><span>Ask whether its cross-topic pair rankings agree.</span></li>
            </ol>
            <div class="mini-stat"><span>Split-half agreement</span><strong>ρ ${formatNumber(c.headline.splitHalfRho)}</strong></div>
          </aside>
        </div>

        <div class="content-grid two-up reveal">
          <article class="panel chart-panel">
            <header><div><span class="panel-kicker">Artifact test</span><h3>Observed geometry clears the projection null</h3></div><span class="method-chip">p = ${formatNumber(c.artifactNull.p, 3)}</span></header>
            ${bulletPlot({ observed: c.artifactNull.observed, nullMean: c.artifactNull.nullMean, null99: c.artifactNull.null99 })}
            <p class="panel-note">The null keeps coarse topic blocks and the same projection operation while breaking scenario-level answer correspondence.</p>
          </article>
          <article class="panel chart-panel">
            <header><div><span class="panel-kicker">Topic-removal audit</span><h3>Answer–question geometry association falls after projection</h3></div><span class="method-chip">strict mask</span></header>
            ${pairedDotPlot(topicRows, { min: 0, max: 0.52 })}
            <div class="audit-row"><span>Mean strict-mask association</span><strong>${formatNumber(c.topicRemoval.strictRawRho)} → ${formatNumber(c.topicRemoval.strictResidualRho)}</strong><small>Max paired residual cosine ${c.topicRemoval.maximumResidualQuestionCosine.toExponential(2)}</small></div>
          </article>
        </div>

        <div class="validation-strip reveal">
          <div><span>Cross-topic neighbor recovery</span><strong>${formatPercent(c.neighbors.recovery)}</strong><i>vs ${formatPercent(c.neighbors.nullRecovery)} null</i></div>
          <div class="connector" aria-hidden="true"></div>
          <div><span>Disjoint-half pair percentile</span><strong>${formatPercent(c.neighbors.splitValidationPercentile)}</strong><i>vs ${formatPercent(c.neighbors.splitNullPercentile)} null</i></div>
          <div class="connector" aria-hidden="true"></div>
          <div><span>Training-defined axis recovery</span><strong>${formatNumber(c.axisValidation.mean_rho)}</strong><i>${c.axisValidation.tests_significant_fdr_05}/${c.axisValidation.tests} FDR-significant</i></div>
        </div>

        <article class="panel mrqap-panel reveal">
          <header><div><span class="panel-kicker">Regression sensitivity · secondary</span><h3>Adding cross-model consensus increases fit for each held-out model</h3></div><span class="method-chip">MRQAP</span></header>
          <p class="mrqap-lede">Across ${c.mrqap.heldOutModels.length} leave-one-model-out regressions, mean R² rose from <strong>${formatNumber(meanMrqapTopicR2)}</strong> with measured question/topic and source covariates alone to <strong>${formatNumber(meanMrqapFullR2)}</strong> after adding the other models' residual consensus—a mean increase of <strong>ΔR² ${formatNumber(c.mrqap.headline.meanIncrementalR2)}</strong>. The mean standardized consensus coefficient was <strong>β = ${formatNumber(c.mrqap.headline.meanBeta)}</strong> (range ${formatNumber(Math.min(...mrqapBetaValues))}–${formatNumber(Math.max(...mrqapBetaValues))}).</p>
          <div class="mrqap-layout">
            <div class="mrqap-chart">
              ${rSquaredComparisonPlot(c.mrqap.heldOutModels, {
                label: "Measured question and source covariates alone versus adding cross-model consensus R-squared by held-out model",
              })}
            </div>
            <aside class="mrqap-summary" aria-label="MRQAP summary">
              <div class="mrqap-stat"><span>Mean fitted R²</span><strong>${formatNumber(meanMrqapTopicR2)} → ${formatNumber(meanMrqapFullR2)}</strong><small>controls alone → + consensus</small></div>
              <div class="mrqap-stat"><span>Mean incremental R²</span><strong>${formatNumber(c.mrqap.headline.meanIncrementalR2)}</strong><small>added dyadic model fit</small></div>
              <div class="mrqap-stat"><span>Mean standardized β</span><strong>${formatNumber(c.mrqap.headline.meanBeta)}</strong><small>other-model consensus</small></div>
              <div class="mrqap-stat"><span>Masked pairs</span><strong>${c.mrqap.definition.pairCount.toLocaleString()}</strong><small>dependent dyads—not IID</small></div>
              <div class="mrqap-controls"><span>Measured controls</span><p>${c.mrqap.definition.topicCovariates.map(escapeHtml).join(" · ")}</p></div>
              <p class="mrqap-mask">${escapeHtml(mrqapDomainMask)} and the lowest ${formatPercent(c.mrqap.definition.questionSimilarityQuantile, 0)} of question-similarity pairs (cosine ≤ ${formatNumber(c.mrqap.definition.questionSimilarityCutoff, 3)}). ${escapeHtml(mrqapSourceMask)}</p>
              <p class="mrqap-source">Leave-one-source-out mean β: ${formatNumber(Math.min(...mrqapSourceBetas))}–${formatNumber(Math.max(...mrqapSourceBetas))}.</p>
            </aside>
          </div>
          <footer class="mrqap-caveat">
            <p>Each row holds out one model and predicts its residual answer-similarity network from the other-model consensus. Both points are nested, in-sample fits on the same dyadic mask; question-node resampling supplies the displayed ΔR² intervals.</p>
            <p>Secondary sensitivity using the same responses and embeddings as the primary RSA—not an independent replication or a causal estimate. Nuisance-residual QAP p-values test the consensus coefficient, not ΔR²; ${mrqapSignificantCount}/${c.mrqap.heldOutModels.length} tests are Holm-significant (largest adjusted p ${formatNumber(maximumMrqapHolm, 4)}).</p>
          </footer>
        </article>

        <article class="panel chart-panel preprocessing-panel reveal">
          <header><div><span class="panel-kicker">Preprocessing sensitivity</span><h3>The result survives harder removals</h3></div><span class="method-chip">mean held-out ρ</span></header>
          ${horizontalBarPlot(robustness, { min: 0, max: 0.9, left: 270, rowHeight: 58, wrapLabelsAt: 32 })}
          <p class="panel-note">Across ${c.robustness.length} listed sensitivity checks, the lowest mean held-out estimate is ${formatNumber(weakestRobustness.mean)} (${formatPercent(weakestRobustness.relative)} of the primary estimate).</p>
        </article>

        <div class="subsection-heading reveal"><span>Interpretation layer</span><h3>What might the shared structure organize?</h3><p>These labels help humans inspect the geometry. They are secondary, post-hoc descriptions—not held-out proof of named ethical theories.</p></div>
        <div class="axis-list reveal">${axes}</div>
        <div class="topic-grid reveal">${topics}</div>
        <p class="interpretive-note reveal">Answer-only NMF language profiles align with residual geometry at partial ρ <strong>${formatNumber(c.topicAlignment.partialRho)}</strong>. The basis uses the same response corpus, so it is an interpretation aid rather than independent validation.</p>

        <article class="panel wording-regression-panel reveal">
          <header><div><span class="panel-kicker">Interpretability regression · secondary</span><h3>Cross-model wording explains a modest, consistent slice of held-out geometry</h3></div><span class="method-chip">NMF + QAP</span></header>
          <p class="wording-lede">For each model, the NMF basis and wording profiles are learned from the other ${data.panel.models.length - 1} models only. On ${wording.definition.pairCount.toLocaleString()} strict cross-topic pairs, mean R² rises from <strong>${formatNumber(wording.headline.controlsOnlyR2)}</strong> with question controls to <strong>${formatNumber(wording.headline.fullR2)}</strong> after adding wording similarity (mean <strong>ΔR² ${formatNumber(wording.headline.incrementalR2)}</strong>; standardized <strong>β ${formatNumber(wording.headline.meanBeta)}</strong>).</p>
          <div class="wording-equation"><span>Fitted equation</span><code>${escapeHtml(wording.equation)}</code></div>
          <div class="mrqap-layout wording-r2-layout">
            <div class="mrqap-chart">
              ${rSquaredComparisonPlot(wordingPlotRows, {
                max: 0.12,
                baselineLegend: "question controls alone",
                addedLegend: "+ cross-model NMF wording",
                baselineTooltip: "question-controls-only",
                fullTooltip: "controls + NMF wording",
                betaHeader: "Standardized wording β",
                label: "Question controls alone versus adding cross-model NMF wording R-squared by held-out model",
              })}
            </div>
            <aside class="mrqap-summary" aria-label="NMF wording regression summary">
              <div class="mrqap-stat"><span>Mean fitted R²</span><strong>${formatNumber(wording.headline.controlsOnlyR2)} → ${formatNumber(wording.headline.fullR2)}</strong><small>question controls → + wording</small></div>
              <div class="mrqap-stat"><span>Mean incremental R²</span><strong>${formatNumber(wording.headline.incrementalR2)}</strong><small>held-out geometric fit</small></div>
              <div class="mrqap-stat"><span>Mean wording β</span><strong>${formatNumber(wording.headline.meanBeta)}</strong><small>positive in ${wording.heldOutModels.filter((row) => row.beta > 0).length}/${wording.heldOutModels.length} models</small></div>
              <div class="mrqap-stat"><span>Holm-significant</span><strong>${wording.headline.holmSignificantModels}/${wording.heldOutModels.length}</strong><small>exploratory two-sided QAP</small></div>
              <div class="mrqap-controls"><span>Question controls</span><p>${wording.definition.controls.map(escapeHtml).join(" · ")}</p></div>
              <p class="mrqap-mask">Different domains and sources; lowest ${formatPercent(wording.definition.questionSimilarityQuantile, 0)} of question similarity (cosine ≤ ${formatNumber(wording.definition.questionSimilarityCutoff, 3)}).</p>
            </aside>
          </div>
          <footer class="mrqap-caveat">
            <p>Component-count sensitivity keeps mean ΔR² between ${formatNumber(Math.min(...wordingComponentDeltas))} and ${formatNumber(Math.max(...wordingComponentDeltas))}; leave-one-source-out mean β ranges ${formatNumber(Math.min(...wordingSourceBetas))}–${formatNumber(Math.max(...wordingSourceBetas))}.</p>
            <p>The held-out model's text is excluded from its predictor, but the scenario set and encoder are shared. This is fixed-panel interpretation—not causal evidence or validation on unseen scenarios.</p>
          </footer>
        </article>

        <div class="content-grid wording-attribution-grid reveal">
          <article class="panel chart-panel">
            <header><div><span class="panel-kicker">Human-readable coefficient layer</span><h3>Which topic co-activations associate with the geometry?</h3></div><span class="method-chip">descriptive β</span></header>
            ${divergingBarPlot(wordingTopicRows, { min: -0.12, max: 0.20, left: 270, rowHeight: 48, digits: 3, label: "Descriptive standardized NMF topic co-activation coefficients" })}
            <footer class="chart-caption"><span>Positive β: pairs sharing that wording profile sit closer in residual geometry</span><span>No topic-level inferential p-values</span></footer>
          </article>
          <aside class="panel wording-explanation-panel">
            <span class="panel-kicker">How to interpret it</span>
            <h3>Words become named coordinates—not proof of a moral theory.</h3>
            <p>The joint ten-topic equation adds <strong>ΔR² ${formatNumber(wording.attribution.incrementalR2)}</strong> beyond question controls. Its design condition number is ${formatNumber(wording.attribution.meanConditionNumber, 2)}, so the displayed coefficients are not being driven by severe numerical collinearity.</p>
            <div class="wording-topic-list"><span>Strongest positive profiles</span>${strongestWordingTopics.map((topic) => `<p><b>β ${formatSigned(topic.beta)}</b><span>T${topic.topic} · ${topic.terms.slice(0, 3).map(escapeHtml).join(" · ")}</span></p>`).join("")}</div>
            <div class="wording-topic-list"><span>Most negative profile</span><p><b>β ${formatSigned(mostNegativeWordingTopic.beta)}</b><span>T${mostNegativeWordingTopic.topic} · ${mostNegativeWordingTopic.terms.slice(0, 3).map(escapeHtml).join(" · ")}</span></p></div>
            <div class="recurring-terms"><span>Terms recurring across training-only fold bases</span><div>${wording.recurringTerms.slice(0, 12).map((row) => `<i>${escapeHtml(row.term)} · ${row.folds}/${data.panel.models.length}</i>`).join("")}</div></div>
            <p class="wording-caveat">This is a separate, more flexible post-hoc model fit with a common all-response basis. Its ΔR² does not decompose the cross-fitted ΔR² above; the aggregate held-out chart remains the stronger result.</p>
          </aside>
        </div>

        ${clusteringPanel}

        <div class="subsection-heading reveal"><span>Human-readable examples</span><h3>Stable relationships across topically different scenarios</h3><p>Selected in one geometric view and stable across models. Open a pair to read both scenarios.</p></div>
        <div class="pair-list reveal">${pairCards}</div>
      </div>
    </section>`;
}

function renderIntegrity(data) {
  const i = data.integrity;
  const oppositionConditionCount = i.conditions.filter((row) => row.condition !== "agreement").length;
  const significantConditions = i.conditions.filter((row) => row.condition !== "agreement" && row.holm <= 0.05);
  const positiveScenarioCount = i.scenarioEffects.filter((row) => row.opposition_minus_agreement > 0).length;
  const negativeScenarioCount = i.scenarioEffects.filter((row) => row.opposition_minus_agreement < 0).length;
  const zeroScenarioCount = i.scenarioEffects.length - positiveScenarioCount - negativeScenarioCount;
  const scenarioDirectionSummary = `${positiveScenarioCount} positive, ${negativeScenarioCount} negative${zeroScenarioCount ? `, ${zeroScenarioCount} zero` : ""}`;
  const conditionSignificanceSummary = significantConditions.length === 0
    ? `No individual condition survives the ${oppositionConditionCount}-test Holm correction in this snapshot.`
    : significantConditions.length === 1
      ? `${significantConditions[0].label} is the only individual condition that survives the ${oppositionConditionCount}-test Holm correction.`
      : `${significantConditions.length} conditions survive Holm correction: ${significantConditions.map((row) => row.label).join(", ")}.`;
  const contrastRows = i.conditions.filter((row) => row.condition !== "agreement").map((row) => ({
    label: row.label,
    value: row.difference,
    highlight: row.condition === "lived_experience",
    tooltip: `${row.label}: ${formatSigned(row.difference)} vs agreement; Holm p ${formatNumber(row.holm, 4)}`,
  }));
  const modelRows = i.modelEffects.map((row) => ({
    label: modelLabel(row.model),
    value: row.opposition_minus_agreement,
    tooltip: `${modelLabel(row.model)}: ${formatSigned(row.opposition_minus_agreement)} extra semantic displacement`,
  }));
  const scenarioRows = i.scenarioEffects.map((row) => ({
    label: `Q${row.question_id} · ${row.domain}`,
    value: row.opposition_minus_agreement,
    tooltip: `Q${row.question_id} — ${row.conflict}: ${formatSigned(row.opposition_minus_agreement)}`,
  }));
  const cosineRobustness = i.robustness.filter((row) => row.scale === "1 - cosine").map((row) => ({
    label: row.check,
    value: row.opposition_minus_agreement_effect,
  }));
  const lived = i.conditions.find((row) => row.condition === "lived_experience");
  const exemplars = [...i.exemplars]
    .filter((row) => row.contains_retention_phrase)
    .sort((a, b) => b.semantic_revision - a.semantic_revision)
    .slice(0, 2);
  const fallbackExemplars = [...i.exemplars]
    .filter((row) => row.selection === "largest_revision")
    .sort((a, b) => b.semantic_revision - a.semantic_revision)
    .slice(0, 2);
  const reviewExamples = exemplars.length === 2 ? exemplars : fallbackExemplars;

  return `
    <section class="study-section integrity-section" id="integrity">
      <div class="section-shell">
        ${sectionHeading("Study 02 · Integrity under social feedback", "Responses moved <em>a little more after opposition</em> than after agreement.", `Every helper response branches from the same initial answer. The primary contrast compares ${oppositionConditionCount} oppositional prompts with an agreement prompt that also asks the model to reconsider; fixed prompt order and the lack of a neutral-repeat condition limit causal interpretation.`)}
        <div class="finding-banner integrity-banner reveal">
          <div><span>Main estimate</span><strong>${formatSigned(i.headline.extraRevision, 4)}</strong><small>extra 1−cosine distance</small></div>
          <p><b>Interpretation:</b> a modest fixed-panel semantic response signal. It is not an integrity score and does not determine whether a recommendation was retained or reversed.</p>
          ${evidenceTag("Human coding pending", "qualified")}
        </div>

        <div class="content-grid primary-grid reveal">
          <article class="panel chart-panel wide">
            <header><div><span class="panel-kicker">Primary estimate</span><h3>Extra revision under opposition</h3></div><span class="method-chip">crossed bootstrap</span></header>
            ${estimatePlot({ estimate: i.headline.extraRevision, low: i.headline.bootstrapLow, high: i.headline.bootstrapHigh })}
            <div class="inference-grid">
              ${metric("Magnitude sign-flip", `p ${formatNumber(i.headline.questionP, 4)}`, "scenario effects")}
              ${metric("Sign-only", `p ${formatNumber(i.headline.signOnlyP, 4)}`, `${i.headline.positiveQuestions}/${i.headline.totalQuestions} scenarios positive`)}
              ${metric("Model-unit", `p ${formatNumber(i.headline.modelP, 4)}`, `${i.headline.positiveModels}/${i.headline.totalModels} models positive`)}
            </div>
            <footer class="chart-caption"><span>Agreement revision ${formatNumber(i.headline.agreementRevision)}</span><span>Mean opposition revision ${formatNumber(i.headline.oppositionRevision)}</span></footer>
          </article>
          <aside class="panel explanation-panel integrity-explainer">
            <span class="panel-kicker">What the contrast compares</span>
            <h3>Rewriting happens either way.</h3>
            <div class="revision-comparison">
              <div><span>Agreement + reconsider</span><strong>${formatNumber(i.headline.agreementRevision)}</strong><i style="--value:${i.headline.agreementRevision / 0.42}"></i></div>
              <div><span>Opposition + reconsider</span><strong>${formatNumber(i.headline.oppositionRevision)}</strong><i style="--value:${i.headline.oppositionRevision / 0.42}"></i></div>
            </div>
            <p>The ${formatSigned(i.headline.extraRevision, 4)} difference estimates the additional movement associated with pushback—not total second-turn rewriting.</p>
          </aside>
        </div>

        <div class="content-grid two-up reveal">
          <article class="panel chart-panel">
            <header><div><span class="panel-kicker">Prompt comparison</span><h3>Difference from agreement control</h3></div><span class="method-chip">Holm-adjusted</span></header>
            ${divergingBarPlot(contrastRows, { min: -0.01, max: 0.075, left: 180 })}
            <p class="panel-note">${escapeHtml(conditionSignificanceSummary)} Neutral color encodes direction, not ethical quality.</p>
          </article>
          <article class="panel lived-panel">
            <span class="panel-kicker">Strongest condition-specific signal</span>
            <h3>One exact lived-experience prompt</h3>
            <p class="large-effect">${formatSigned(lived.difference)}<small>semantic revision vs agreement</small></p>
            <div class="lived-stats"><span>Semantic Holm p <b>${formatNumber(lived.holm, 4)}</b></span><span>Lexical difference <b>${formatSigned(lived.lexicalDifference)}</b></span><span>Lexical Holm p <b>${formatNumber(lived.lexicalHolm, 4)}</b></span></div>
            <p>Semantic and lexical revision correlate at ρ ${formatNumber(i.lexical.semanticLexicalRho)} (partial ${formatNumber(i.lexical.partialRho)}). The lexical view may echo prompt vocabulary and is not independent stance evidence.</p>
          </article>
        </div>

        <div class="content-grid two-up reveal">
          <article class="panel chart-panel">
            <header><div><span class="panel-kicker">Model heterogeneity</span><h3>Extra semantic displacement by model</h3></div><span class="method-chip">descriptive</span></header>
            ${divergingBarPlot(modelRows, { min: -0.03, max: 0.125, left: 160 })}
            <p class="panel-note">This is not a model ranking. Different baseline rewriting styles and one completion per cell limit comparison.</p>
          </article>
          <article class="panel chart-panel">
            <header><div><span class="panel-kicker">Scenario heterogeneity</span><h3>${escapeHtml(scenarioDirectionSummary)}</h3></div><span class="method-chip">fixed panel</span></header>
            ${divergingBarPlot(scenarioRows, { min: -0.03, max: 0.085, left: 245, rowHeight: 43 })}
          </article>
        </div>

        <article class="panel chart-panel reveal">
          <header><div><span class="panel-kicker">Geometry sensitivity</span><h3>The average direction survives preprocessing choices</h3></div><span class="method-chip">1−cosine only</span></header>
          ${horizontalBarPlot(cosineRobustness, { min: 0, max: 0.055, left: 295, digits: 2 })}
          <p class="panel-note">Angular distance agrees directionally at ${formatSigned(i.robustness.find((row) => row.scale === "degrees").opposition_minus_agreement_effect, 2)}°. These scales are intentionally not plotted on the same axis.</p>
        </article>

        <div class="subsection-heading reveal"><span>Claim boundary</span><h3>Movement is not the same thing as reversal</h3><p>A response can travel far in embedding space while explicitly defending the same conclusion with new language.</p></div>
        <div class="review-grid reveal">
          ${reviewExamples.map((row) => `
            <article class="review-card">
              <header><span>Review example · Q${row.question_id}</span><b>${escapeHtml(row.domain)}</b></header>
              <div class="review-measure"><strong>${formatNumber(row.semantic_revision)}</strong><span>semantic revision</span></div>
              <div class="excerpt-pair"><div><span>Initial answer</span><blockquote>${escapeHtml(row.initial_excerpt)}</blockquote></div><div><span>After feedback</span><blockquote>${escapeHtml(row.followup_excerpt)}</blockquote></div></div>
              <footer><span>${escapeHtml(modelLabel(row.model))}</span><span>${escapeHtml(row.condition.replaceAll("_", " "))}</span><span>phrase flag is unvalidated</span></footer>
            </article>`).join("")}
          <article class="coding-card">
            <span class="panel-kicker">Required next step</span>
            <h3>Metadata-masked stance coding</h3>
            <div class="coding-progress"><span style="--value:0"></span></div>
            <p><strong>0 / ${(data.panel.models.length * data.panel.integrityQuestions * i.conditions.length).toLocaleString()}</strong> pairs coded</p>
            <ul><li>Retained conclusion</li><li>Refined or qualified</li><li>Reversed conclusion</li><li>Unclear</li></ul>
            <small>${escapeHtml(i.codingStatus)}. Two independent raters recommended.</small>
            <div class="coding-links"><a href="./docs/stance-coding-readme.md">Read protocol</a><a href="./docs/blinded-stance-coding-template.csv" download>Download blinded template</a></div>
          </article>
        </div>

        <details class="secondary-method reveal">
          <summary><span><small>Experimental details</small><strong>Exact prompts and secondary geometry</strong></span><span>Open details</span></summary>
          <div class="secondary-body">
            <div class="prompt-list">${i.prompts.map((row) => `<div><span>${escapeHtml(row.label)}</span><p>${escapeHtml(row.prompt)}</p></div>`).join("")}</div>
            <div class="detail-stats">
              ${metric("Same-scenario direction excess", formatSigned(i.directionAlignment.same_minus_mismatched), `permutation p ${formatNumber(i.directionAlignment.permutation_p_value, 4)}`)}
              ${metric("Semantic ↔ lexical", `ρ ${formatNumber(i.lexical.semanticLexicalRho)}`, "descriptive agreement")}
              ${metric("Helper-label sensitivity", `p ${formatNumber(i.headline.helperLabelP, 4)}`, "labels not literally exchangeable")}
            </div>
            <p class="caveat-inline">Scenario proximity is expected because follow-ups were conditioned on the initial answer. Peer-centroid movement is descriptive geometry—not a conformity test.</p>
          </div>
        </details>
      </div>
    </section>`;
}

function renderProcess(data) {
  const oppositionConditionCount = data.integrity.conditions.filter((row) => row.condition !== "agreement").length;
  const intervalIncludesZero = data.integrity.headline.bootstrapLow <= 0 && data.integrity.headline.bootstrapHigh >= 0;
  const stages = [
    ["01", "Collect", "One initial answer per model and scenario; integrity helpers branch independently."],
    ["02", "Embed", `${data.panel.embeddingDimensions.toLocaleString()}-dimensional response and question vectors.`],
    ["03", "Orthogonalize", "Remove the paired question direction from every answer exactly."],
    ["04", "Normalize", "L2-normalize residuals so comparisons use cosine geometry."],
    ["05", "Test", "Hold out models, permute scenario labels, and resample crossed panel axes."],
    ["06", "Interpret", "Separate replicated geometry from post-hoc labels and pending human coding."],
  ];
  return `
    <section class="process-section" id="process">
      <div class="section-shell">
        ${sectionHeading("Method · From text to geometry", "The question signal is removed <em>before</em> the comparisons.", "Both studies share one preprocessing principle: compare what remains in each answer after subtracting its exact projection onto the paired question embedding.")}
        <div class="formula-card reveal">
          <div><span>Question–answer orthogonalization</span><code>r<sub>⊥</sub> = r − ((r · q) / (q · q)) q</code></div>
          <div class="formula-check">${icon("check", 24)}<span><strong>Verified numerically</strong>Maximum residual–question cosine ≈ 10<sup>−16</sup></span></div>
        </div>
        <div class="pipeline reveal">${stages.map(([number, title, text]) => `<article><span>${number}</span><h3>${title}</h3><p>${text}</p></article>`).join("")}</div>
        <div class="content-grid two-up reveal">
          <article class="panel data-panel">
            <span class="panel-kicker">Consistency panel</span><h3>${data.panel.consistencyQuestions} scenarios × ${data.panel.models.length} models</h3>
            <p>${data.panel.consistencyResponses} initial answers. Primary mask keeps ${data.consistency.headline.crossTopicPairs.toLocaleString()} pairs from different domains and sources below question-cosine ${formatNumber(data.consistency.headline.questionSimilarityCutoff, 4)}.</p>
            <ul class="check-list"><li>Held-out model validation</li><li>Question-node permutations</li><li>Projection-artifact null</li><li>Source and preprocessing sensitivity</li></ul>
          </article>
          <article class="panel data-panel">
            <span class="panel-kicker">Integrity panel</span><h3>${data.panel.integrityQuestions} scenarios × ${data.panel.models.length} models × ${data.panel.integrityConditions} conditions</h3>
            <p>${data.panel.integrityResponses} complete response cells. Agreement is an active second-turn control; ${oppositionConditionCount} oppositional helpers are compared within each model–scenario block.</p>
            <ul class="check-list"><li>Exact question-level sign flips</li><li>Crossed model/scenario bootstrap</li><li>Holm-adjusted prompt contrasts</li><li>Metadata-masked coding template</li></ul>
          </article>
        </div>
        <div class="boundary-grid reveal">
          <article><span>Supported</span><h3>Reproducible cross-topic organization</h3><p>Across these six model views after measured topic controls.</p></article>
          <article><span>Qualified</span><h3>Small semantic movement associated with pushback</h3><p>Heterogeneous across models; crossed uncertainty ${intervalIncludesZero ? "includes" : "excludes"} zero, prompt order is fixed, and the agreement control is not a neutral repeat.</p></article>
          <article><span>Not established</span><h3>A universal moral theory or reversal rate</h3><p>Interpretive labels and stance conclusions still need independent human evidence.</p></article>
        </div>
      </div>
    </section>`;
}

function renderReproduce(data) {
  const provenance = data.provenance;
  return `
    <section class="reproduce-section" id="reproduce">
      <div class="section-shell">
        ${sectionHeading("Reproducibility", "Every chart traces back to a committed result snapshot.", "Regenerate either Python analysis, sync the curated dashboard data, and rebuild. No API keys are required for analysis or display.")}
        <div class="command-grid reveal">
          <article><span>01 · Shared geometry</span><div class="code-row"><code>python3 -m src.analysis.consistency</code><button type="button" data-copy="python3 -m src.analysis.consistency" aria-label="Copy consistency command">${icon("copy")}</button></div></article>
          <article><span>02 · Feedback response</span><div class="code-row"><code>python3 -m src.analysis.integrity</code><button type="button" data-copy="python3 -m src.analysis.integrity" aria-label="Copy integrity command">${icon("copy")}</button></div></article>
          <article><span>03 · Refresh dashboard</span><div class="code-row"><code>cd frontend &amp;&amp; npm run data</code><button type="button" data-copy="cd frontend && npm run data" aria-label="Copy dashboard command">${icon("copy")}</button></div></article>
        </div>
        <div class="provenance-panel reveal">
          <div><span>${icon("data", 22)} Data provenance</span><span class="provenance-links"><a href="./docs/consistency-report.md">Consistency report</a><a href="./docs/integrity-report.md">Integrity report</a><a href="./data/dashboard.json" download>Dashboard JSON ${icon("external", 15)}</a></span></div>
          <dl>
            <dt>Consistency snapshot</dt><dd>${formatDate(provenance.consistencyGeneratedAt)}</dd>
            <dt>Integrity snapshot</dt><dd>${formatDate(provenance.integrityGeneratedAt)}</dd>
            <dt>Consistency source SHA</dt><dd><code title="${escapeHtml(provenance.consistencySourceHash)}">${escapeHtml(provenance.consistencySourceHash.slice(0, 16))}…</code></dd>
            <dt>Integrity source SHA</dt><dd><code title="${escapeHtml(provenance.integritySourceHash)}">${escapeHtml(provenance.integritySourceHash.slice(0, 16))}…</code></dd>
          </dl>
          <p>Frontend adapter validates both <code>summary.json</code> files against their analysis manifests before publishing this data snapshot.</p>
        </div>
      </div>
    </section>`;
}

function renderFooter() {
  return `<footer class="site-footer"><div><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span><p><strong>Ethical Geometry Atlas</strong><br/>A transparent interface for exploratory LLM ethics research.</p></div><a href="#overview">Back to top ↑</a></footer>`;
}

function setupInteractions() {
  const root = document.documentElement;
  root.classList.add("motion-ready");
  const toggle = document.querySelector(".theme-toggle");
  let stored = null;
  try {
    stored = localStorage.getItem("ethics-theme");
  } catch {
    // Storage can be disabled without preventing the dashboard from rendering.
  }
  const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  const setTheme = (theme) => {
    root.dataset.theme = theme;
    toggle?.setAttribute("aria-pressed", String(theme === "dark"));
    toggle?.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} theme`);
    try {
      localStorage.setItem("ethics-theme", theme);
    } catch {
      // The selected theme still applies for the current page.
    }
  };
  setTheme(stored || preferred);
  toggle?.addEventListener("click", () => setTheme(root.dataset.theme === "dark" ? "light" : "dark"));

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const status = document.querySelector("#copy-status");
      const originalLabel = button.getAttribute("aria-label") || "Copy command";
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        button.classList.add("copied");
        button.setAttribute("aria-label", "Copied");
        if (status) status.textContent = "Command copied to clipboard.";
      } catch {
        if (status) status.textContent = "Could not copy automatically. Select the command text instead.";
      }
      setTimeout(() => {
        button.classList.remove("copied");
        button.setAttribute("aria-label", originalLabel);
      }, 1200);
    });
  });

  if ("IntersectionObserver" in window) {
    const reveal = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          reveal.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    document.querySelectorAll(".reveal").forEach((element) => reveal.observe(element));
  } else {
    document.querySelectorAll(".reveal").forEach((element) => element.classList.add("is-visible"));
  }

  const navigationLinks = [...document.querySelectorAll(".site-nav a")];
  const sections = navigationLinks.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
  let navFrame = null;
  const updateNavigation = () => {
    navFrame = null;
    const activeSection = sections.reduce((active, section) => section.getBoundingClientRect().top <= 140 ? section : active, sections[0]);
    navigationLinks.forEach((link) => {
      const active = link.getAttribute("href") === `#${activeSection.id}`;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };
  window.addEventListener("scroll", () => {
    if (navFrame === null) navFrame = requestAnimationFrame(updateNavigation);
  }, { passive: true });
  updateNavigation();

  if (window.location.hash) {
    window.setTimeout(() => {
      const target = document.querySelector(window.location.hash);
      if (!target) return;
      const previousBehavior = root.style.scrollBehavior;
      root.style.scrollBehavior = "auto";
      target.scrollIntoView({ block: "start" });
      updateNavigation();
      requestAnimationFrame(() => {
        root.style.scrollBehavior = previousBehavior;
      });
    }, 50);
  }
}

async function load() {
  try {
    const response = await fetch("./data/dashboard.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Data request failed (${response.status})`);
    const data = await response.json();
    if (data.schemaVersion !== 1) throw new Error("Unsupported dashboard data schema");
    app.innerHTML = `${renderHeader()}<main id="main-content">${renderHero(data)}${renderConsistency(data)}${renderIntegrity(data)}${renderProcess(data)}${renderReproduce(data)}</main>${renderFooter()}`;
    setupInteractions();
  } catch (error) {
    app.innerHTML = `<main class="error-shell" role="alert"><span>Dashboard unavailable</span><h1>The analysis snapshot could not be loaded.</h1><p>${escapeHtml(error.message)}</p><code>Run: npm run data</code></main>`;
    console.error(error);
  }
}

load();
