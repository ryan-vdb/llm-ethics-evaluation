import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(frontendRoot, "..");

const paths = {
  consistencySummary: resolve(repositoryRoot, "src/analysis/consistency/results/summary.json"),
  consistencyManifest: resolve(repositoryRoot, "src/analysis/consistency/results/manifest.json"),
  integritySummary: resolve(repositoryRoot, "src/analysis/integrity/results/summary.json"),
  integrityManifest: resolve(repositoryRoot, "src/analysis/integrity/results/manifest.json"),
  output: resolve(frontendRoot, "public/data/dashboard.json"),
  publicDocs: resolve(frontendRoot, "public/docs"),
};

const publicDocuments = [
  ["src/analysis/consistency/results/REPORT.md", "consistency-report.md"],
  ["src/analysis/integrity/results/REPORT.md", "integrity-report.md"],
  ["src/analysis/integrity/results/coding/README.md", "stance-coding-readme.md"],
  ["src/analysis/integrity/results/coding/blinded_stance_coding_template.csv", "blinded-stance-coding-template.csv"],
];

const mean = (values) => values.reduce((total, value) => total + value, 0) / values.length;
const titleCase = (value) => value.split("_").map((word) => word[0].toUpperCase() + word.slice(1)).join(" ");

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function readJsonArtifact(path) {
  const text = await readFile(path, "utf8");
  return { text, value: JSON.parse(text) };
}

function verifySummary(manifest, text, expectedProducer, label) {
  if (manifest.producer !== expectedProducer || manifest.manifest_schema_version !== 1) {
    throw new Error(`${label} manifest producer/schema is not recognized`);
  }
  const artifact = manifest.artifacts?.find((row) => row.path === "summary.json");
  if (!artifact) throw new Error(`${label} manifest does not list summary.json`);
  const digest = createHash("sha256").update(text).digest("hex");
  if (artifact.sha256 !== digest || artifact.bytes !== Buffer.byteLength(text)) {
    throw new Error(`${label} summary.json does not match its manifest`);
  }
}

function requireValue(value, label) {
  if (value === undefined || value === null) throw new Error(`Missing dashboard field: ${label}`);
  return value;
}

function requireJoin(map, key, label) {
  const value = map.get(key);
  if (!value) throw new Error(`Missing ${label} join for condition: ${key}`);
  return value;
}

function assertUnique(values, label) {
  if (new Set(values).size !== values.length) throw new Error(`${label} contains duplicate identifiers`);
}

function assertSameSet(left, right, label) {
  const normalizedLeft = [...left].sort();
  const normalizedRight = [...right].sort();
  if (JSON.stringify(normalizedLeft) !== JSON.stringify(normalizedRight)) {
    throw new Error(`${label} do not match`);
  }
}

function assertFiniteNumbers(value, label = "dashboard") {
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new Error(`${label} contains a non-finite number`);
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertFiniteNumbers(item, `${label}[${index}]`));
  } else if (value && typeof value === "object") {
    Object.entries(value).forEach(([key, item]) => assertFiniteNumbers(item, `${label}.${key}`));
  }
}

export async function buildDashboardData() {
  const [consistencyArtifact, consistencyManifest, integrityArtifact, integrityManifest] = await Promise.all([
    readJsonArtifact(paths.consistencySummary),
    readJson(paths.consistencyManifest),
    readJsonArtifact(paths.integritySummary),
    readJson(paths.integrityManifest),
  ]);
  verifySummary(
    consistencyManifest,
    consistencyArtifact.text,
    "src.analysis.consistency",
    "consistency",
  );
  verifySummary(
    integrityManifest,
    integrityArtifact.text,
    "src.analysis.integrity",
    "integrity",
  );
  const consistency = consistencyArtifact.value;
  const integrity = integrityArtifact.value;

  if (consistencyManifest.analysis_source_sha256 !== consistency.meta.reproducibility.analysis_source_sha256) {
    throw new Error("consistency source hash differs between manifest and summary");
  }
  if (integrityManifest.analysis_source_sha256 !== integrity.meta.reproducibility.analysis_source_sha256) {
    throw new Error("integrity source hash differs between manifest and summary");
  }
  if (consistency.meta.reproducibility.input_database_sha256 !== integrity.meta.reproducibility.input_database_sha256) {
    throw new Error("analysis snapshots were generated from different databases");
  }
  if (consistency.meta.embedding_dimensions !== integrity.meta.panel.embedding_dimension) {
    throw new Error("analysis snapshots use different embedding dimensions");
  }
  assertUnique(consistency.meta.models, "consistency model panel");
  assertUnique(integrity.meta.panel.models, "integrity model panel");
  assertSameSet(consistency.meta.models, integrity.meta.panel.models, "analysis model panels");
  assertUnique(integrity.meta.panel.question_ids, "integrity question panel");

  const rsa = consistency.representational_similarity;
  const crossTopic = consistency.cross_domain_neighbors;
  const axes = consistency.shared_latent_axes;
  const topics = consistency.interpretable_reasoning_topics;
  const clustering = consistency.clustering;
  const strictTopic = rsa.topic_leakage_primary_cross_topic_mask;

  const revision = integrity.revision_effects;
  const primary = revision.primary;
  const helperContrasts = new Map(
    revision.per_helper_vs_agreement.map((row) => [row.numerator_condition, row]),
  );
  const lexicalContrasts = new Map(
    integrity.lexical_robustness.condition_vs_agreement.map((row) => [row.condition, row]),
  );
  const displayedConditions = integrity.meta.panel.conditions.filter((condition) => condition !== "initial");
  assertUnique(revision.per_helper_vs_agreement.map((row) => row.numerator_condition), "semantic condition contrasts");
  assertUnique(integrity.lexical_robustness.condition_vs_agreement.map((row) => row.condition), "lexical condition contrasts");
  assertUnique(displayedConditions, "integrity condition panel");
  assertUnique(revision.condition_summaries.map((row) => row.condition), "integrity condition summaries");
  assertSameSet(
    displayedConditions,
    revision.condition_summaries.map((row) => row.condition),
    "integrity condition panels",
  );
  assertUnique(rsa.held_out_models.map((row) => row.model), "held-out models");
  assertUnique(revision.heterogeneity.by_model.map((row) => row.model), "model heterogeneity rows");
  assertUnique(revision.heterogeneity.by_question.map((row) => row.question_id), "scenario heterogeneity rows");
  assertSameSet(consistency.meta.models, rsa.held_out_models.map((row) => row.model), "held-out model panels");
  assertSameSet(consistency.meta.models, revision.heterogeneity.by_model.map((row) => row.model), "heterogeneity model panels");
  assertSameSet(integrity.meta.panel.question_ids, revision.heterogeneity.by_question.map((row) => row.question_id), "scenario heterogeneity panels");
  if ((clustering.selected_k === null) !== (clustering.selected_metrics === null)) {
    throw new Error("clustering selected_k and selected_metrics must both be present or both be null");
  }

  const dashboard = {
    schemaVersion: 1,
    provenance: {
      consistencyGeneratedAt: consistency.meta.generated_at,
      integrityGeneratedAt: integrity.meta.generated_at,
      consistencySourceHash: consistencyManifest.analysis_source_sha256,
      integritySourceHash: integrityManifest.analysis_source_sha256,
      consistencyDatabaseHash: consistency.meta.reproducibility.input_database_sha256,
      integrityDatabaseHash: integrity.meta.reproducibility.input_database_sha256,
      sourceFiles: [
        "src/analysis/consistency/results/summary.json",
        "src/analysis/integrity/results/summary.json",
      ],
    },
    panel: {
      models: consistency.meta.models,
      consistencyQuestions: consistency.meta.questions,
      integrityQuestions: integrity.meta.panel.question_count,
      integrityConditions: integrity.meta.panel.condition_count,
      consistencyResponses: consistency.meta.questions * consistency.meta.models.length,
      integrityResponses: integrity.meta.panel.response_count,
      embeddingDimensions: consistency.meta.embedding_dimensions,
    },
    consistency: {
      headline: {
        meanHeldOutRho: rsa.mean_held_out_rho,
        minimumHeldOutRho: rsa.min_held_out_rho,
        crossTopicPairs: rsa.primary_definition.pair_count,
        questionSimilarityCutoff: rsa.primary_definition.question_similarity_cutoff,
        allModelsHolmSignificant: rsa.held_out_models.every((row) => row.p_value_holm <= 0.05),
        splitHalfRho: rsa.split_half_reliability.mean_rho,
        splitHalfSpearmanBrown: rsa.split_half_reliability.mean_spearman_brown,
      },
      heldOutModels: rsa.held_out_models.map((row) => ({
        model: row.model,
        rho: row.rho,
        low: row.ci_95_low,
        high: row.ci_95_high,
        p: row.p_value,
        holm: row.p_value_holm,
      })),
      topicRemoval: {
        rawQuestionCosine: consistency.meta.orthogonalization.raw_question_cosine_mean,
        maximumResidualQuestionCosine: consistency.meta.orthogonalization.post_question_cosine_abs_max,
        allPairRawRho: mean(rsa.topic_leakage_before_projection.map((row) => row.spearman_rho)),
        allPairResidualRho: mean(rsa.topic_leakage_after_projection.map((row) => row.spearman_rho)),
        strictRawRho: mean(strictTopic.map((row) => row.raw_spearman_rho)),
        strictResidualRho: mean(strictTopic.map((row) => row.residual_spearman_rho)),
        byModel: strictTopic,
      },
      artifactNull: {
        observed: consistency.projection_artifact_null.observed_mean_held_out_rho,
        nullMean: consistency.projection_artifact_null.null_mean,
        null99: consistency.projection_artifact_null.null_99_percentile,
        p: consistency.projection_artifact_null.p_value,
        permutations: consistency.projection_artifact_null.permutations,
        blockSensitivity: consistency.projection_artifact_null.block_count_sensitivity,
      },
      robustness: consistency.robustness_checks.checks.map((row) => ({
        check: row.check,
        mean: row.mean_held_out_rho,
        minimum: row.min_held_out_rho,
        relative: row.rho_relative_to_primary,
      })),
      neighbors: {
        recovery: crossTopic.mean_recovery,
        nullRecovery: crossTopic.mean_permutation_null,
        splitValidationPercentile: crossTopic.split_half_pair_validation.mean_validation_percentile,
        splitNullPercentile: crossTopic.split_half_pair_validation.permutation_null_mean,
        p: crossTopic.split_half_pair_validation.permutation_p_value,
      },
      latentAxes: axes.axis_profiles.map((row) => ({
        axis: row.axis,
        variance: row.explained_variance_ratio,
        label: row.descriptive_label,
        status: row.label_status,
        lowConcepts: row.low_concepts,
        highConcepts: row.high_concepts,
        lowExamples: row.low_extreme_questions.slice(0, 2).map((item) => ({
          id: item.question_id,
          domain: item.domain,
          conflict: item.conflict,
          score: item.score,
        })),
        highExamples: row.high_extreme_questions.slice(0, 2).map((item) => ({
          id: item.question_id,
          domain: item.domain,
          conflict: item.conflict,
          score: item.score,
        })),
      })),
      axisValidation: axes.fold_axis_recovery,
      topics: topics.topic_profiles.map((row) => ({
        topic: row.topic,
        label: row.descriptive_label,
        terms: row.top_terms.slice(0, 6),
        profileRho: row.mean_cross_model_profile_rho,
      })),
      topicAlignment: {
        partialRho: topics.topic_vs_residual_partial_rho,
        p: topics.permutation_p_value,
      },
      clustering: {
        selectedK: clustering.selected_k,
        status: clustering.selection_status,
        metrics: clustering.selected_metrics,
        profiles: (clustering.cluster_profiles || []).map((row) => ({
          cluster: row.cluster,
          size: row.size,
          medoid: row.medoid_question_id,
          terms: row.distinctive_researcher_conflict_terms.slice(0, 6),
          domains: row.unique_domains,
          sources: row.unique_sources,
          strictPairs: row.strict_cross_topic_within_pair_count,
        })),
      },
      stablePairs: crossTopic.stable_cross_topic_pairs.slice(0, 12).map((row) => ({
        rank: row.rank,
        first: {
          id: row.question_id_1,
          domain: row.domain_1,
          conflict: row.conflict_1,
          question: row.question_1,
        },
        second: {
          id: row.question_id_2,
          domain: row.domain_2,
          conflict: row.conflict_2,
          question: row.question_2,
        },
        cosine: row.mean_residual_cosine,
        percentile: row.mean_within_model_percentile,
        rankDispersion: row.cross_model_rank_std,
      })),
    },
    integrity: {
      headline: {
        agreementRevision: primary.agreement_semantic_revision,
        oppositionRevision: primary.mean_opposition_semantic_revision,
        extraRevision: primary.mean_difference,
        questionP: primary.question_sign_flip.p_value,
        signOnlyP: primary.question_direction_sign_test.p_value,
        positiveQuestions: primary.question_direction_sign_test.positive_questions,
        totalQuestions: primary.question_direction_sign_test.n_nonzero_questions,
        modelP: primary.model_sign_flip_sensitivity.p_value,
        positiveModels: primary.model_sign_flip_sensitivity.model_effects.filter((value) => value > 0).length,
        totalModels: primary.model_sign_flip_sensitivity.n_models,
        bootstrapLow: primary.crossed_model_question_bootstrap.ci_lower,
        bootstrapHigh: primary.crossed_model_question_bootstrap.ci_upper,
        helperLabelP: primary.helper_label_exchangeability_sensitivity.p_value_two_sided,
      },
      conditions: revision.condition_summaries.map((row) => {
        const isAgreement = row.condition === "agreement";
        const contrast = isAgreement ? null : requireJoin(helperContrasts, row.condition, "semantic contrast");
        const lexical = isAgreement ? null : requireJoin(lexicalContrasts, row.condition, "lexical contrast");
        return {
          condition: row.condition,
          label: titleCase(row.condition),
          revision: row.mean_semantic_revision,
          standardDeviation: row.standard_deviation_cells,
          difference: isAgreement ? 0 : requireValue(contrast.mean_difference, `${row.condition} mean difference`),
          p: contrast?.question_sign_flip.p_value ?? null,
          holm: contrast?.p_value_holm ?? null,
          lexicalDifference: lexical?.mean_lexical_revision_difference_vs_agreement ?? null,
          lexicalHolm: lexical?.holm_adjusted_p ?? null,
        };
      }),
      modelEffects: revision.heterogeneity.by_model,
      scenarioEffects: revision.heterogeneity.by_question,
      robustness: integrity.robustness_checks.checks,
      lexical: {
        semanticLexicalRho: integrity.lexical_robustness.cellwise_semantic_lexical_spearman,
        partialRho: integrity.lexical_robustness.semantic_lexical_partial_spearman_controlling_length,
      },
      directionAlignment: integrity.scenario_specificity.revision_direction_alignment.aggregate_non_agreement,
      peerGeometry: integrity.consensus_movement.condition_summary,
      prompts: Object.entries(integrity.meta.panel.helper_prompts).map(([condition, prompt]) => ({
        condition,
        label: titleCase(condition),
        prompt,
      })),
      exemplars: integrity.exemplars.exemplars,
      codingStatus: integrity.exemplars.coding_status,
    },
  };

  requireValue(dashboard.consistency.headline.meanHeldOutRho, "consistency meanHeldOutRho");
  requireValue(dashboard.integrity.headline.extraRevision, "integrity extraRevision");
  assertFiniteNumbers(dashboard);
  return dashboard;
}

export async function syncDashboardData() {
  const dashboard = await buildDashboardData();
  await mkdir(dirname(paths.output), { recursive: true });
  await mkdir(paths.publicDocs, { recursive: true });
  await writeFile(paths.output, `${JSON.stringify(dashboard, null, 2)}\n`, "utf8");
  await Promise.all(publicDocuments.map(([source, destination]) => copyFile(
    resolve(repositoryRoot, source),
    resolve(paths.publicDocs, destination),
  )));
  console.log(`Synced analysis snapshot: ${paths.output}`);
  return dashboard;
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  await syncDashboardData();
}
