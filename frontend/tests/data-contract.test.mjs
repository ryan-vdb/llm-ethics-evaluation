import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { buildDashboardData } from "../scripts/sync-data.mjs";
import { modelLabel } from "../src/charts.js";

test("dashboard data preserves the canonical panel and headline results", async () => {
  const data = await buildDashboardData();
  assert.equal(data.schemaVersion, 1);
  assert.ok(data.panel.models.length >= 2);
  assert.ok(data.panel.consistencyQuestions > 0);
  assert.ok(data.panel.integrityQuestions > 0);
  assert.ok(data.panel.embeddingDimensions > 0);
  assert.equal(data.consistency.heldOutModels.length, data.panel.models.length);
  assert.equal(data.consistency.questionSimilarityHeatmap.questions.length, data.panel.consistencyQuestions);
  assert.equal(data.consistency.questionSimilarityHeatmap.matrix.length, data.panel.consistencyQuestions);
  assert.equal(data.consistency.mrqap.heldOutModels.length, data.panel.models.length);
  assert.equal(data.consistency.mrqap.rawHeldOutModels.length, data.panel.models.length);
  assert.equal(data.consistency.wordingRegression.heldOutModels.length, data.panel.models.length);
  assert.equal(data.integrity.conditions.length, data.panel.integrityConditions - 1);
  assert.equal(data.integrity.scenarioEffects.length, data.panel.integrityQuestions);
  assert.equal(data.integrity.modelEffects.length, data.panel.models.length);
  assert.ok(Number.isFinite(data.consistency.headline.meanHeldOutRho));
  assert.ok(Number.isFinite(data.consistency.mrqap.headline.meanBeta));
  assert.ok(Number.isFinite(data.consistency.mrqap.headline.meanIncrementalR2));
  assert.ok(Number.isFinite(data.consistency.wordingRegression.headline.meanBeta));
  assert.ok(Number.isFinite(data.consistency.wordingRegression.headline.incrementalR2));
  assert.ok(Number.isFinite(data.integrity.headline.extraRevision));
  assert.equal(modelLabel("gpt_55"), "GPT-5.5");
});

test("claim-boundary diagnostics remain available to the frontend", async () => {
  const data = await buildDashboardData();
  assert.ok(data.integrity.headline.bootstrapLow <= data.integrity.headline.bootstrapHigh);
  assert.ok(data.integrity.headline.signOnlyP >= 0 && data.integrity.headline.signOnlyP <= 1);
  assert.ok(data.integrity.headline.modelP >= 0 && data.integrity.headline.modelP <= 1);
  assert.equal(
    data.integrity.headline.positiveQuestions,
    data.integrity.scenarioEffects.filter((row) => row.opposition_minus_agreement > 0).length,
  );
  assert.equal(
    data.consistency.headline.allModelsHolmSignificant,
    data.consistency.heldOutModels.every((row) => row.holm <= 0.05),
  );
  assert.ok(Number.isFinite(data.consistency.topicRemoval.strictResidualRho));
  const heatmap = data.consistency.questionSimilarityHeatmap;
  assert.equal(heatmap.modelCount, data.panel.models.length);
  assert.equal(heatmap.displayOrdering.display_only, true);
  assert.deepEqual(
    heatmap.questions.map((row) => row.id).sort((first, second) => first - second),
    Array.from({ length: data.panel.consistencyQuestions }, (_, index) => index),
  );
  const offDiagonal = [];
  heatmap.matrix.forEach((row, rowIndex) => {
    assert.equal(row.length, data.panel.consistencyQuestions);
    assert.ok(Math.abs(row[rowIndex] - 1) < 1e-10);
    row.forEach((value, columnIndex) => {
      assert.ok(value >= -1 && value <= 1);
      assert.ok(Math.abs(value - heatmap.matrix[columnIndex][rowIndex]) < 1e-10);
      if (columnIndex > rowIndex) offDiagonal.push(value);
    });
  });
  assert.equal(heatmap.summary.uniquePairCount, offDiagonal.length);
  assert.ok(Math.abs(heatmap.summary.mean - offDiagonal.reduce((total, value) => total + value, 0) / offDiagonal.length) < 1e-10);
  assert.equal(heatmap.summary.minimum, Math.min(...offDiagonal));
  assert.equal(heatmap.summary.maximum, Math.max(...offDiagonal));
  assert.ok(data.consistency.mrqap.definition.pairCount > 0);
  assert.ok(data.consistency.mrqap.definition.questionSimilarityQuantile > 0);
  assert.ok(data.consistency.mrqap.definition.questionSimilarityQuantile < 1);
  assert.ok(data.consistency.mrqap.definition.topicCovariates.length > 0);
  assert.deepEqual(
    data.consistency.mrqap.heldOutModels.map((row) => row.model).sort(),
    [...data.panel.models].sort(),
  );
  for (const row of data.consistency.mrqap.heldOutModels) {
    assert.ok(row.betaLow <= row.beta && row.beta <= row.betaHigh);
    assert.ok(row.incrementalLow <= row.incrementalR2 && row.incrementalR2 <= row.incrementalHigh);
    assert.ok(Math.abs((row.fullR2 - row.topicOnlyR2) - row.incrementalR2) < 1e-10);
  }
  assert.deepEqual(
    data.consistency.wordingRegression.heldOutModels.map((row) => row.model).sort(),
    [...data.panel.models].sort(),
  );
  for (const row of data.consistency.wordingRegression.heldOutModels) {
    assert.equal(row.alternative, "two-sided");
    assert.ok(row.betaLow <= row.beta && row.beta <= row.betaHigh);
    assert.ok(row.incrementalLow <= row.incrementalR2 && row.incrementalR2 <= row.incrementalHigh);
    assert.ok(Math.abs((row.fullR2 - row.controlsOnlyR2) - row.incrementalR2) < 1e-10);
    assert.ok(row.holm >= 0 && row.holm <= 1);
  }
  assert.equal(
    data.consistency.wordingRegression.attribution.topics.length,
    data.consistency.wordingRegression.definition.components,
  );
  assert.ok(data.consistency.wordingRegression.componentSensitivity.length >= 3);
  assert.ok(data.consistency.wordingRegression.leaveOneSourceOut.length > 0);
  if (data.consistency.clustering.selectedK !== null) {
    const clustering = data.consistency.clustering;
    assert.equal(clustering.profiles.length, clustering.selectedK);
    assert.equal(clustering.metrics.k, clustering.selectedK);
    assert.ok(Number.isFinite(clustering.metrics.question_embedding_silhouette));
    assert.ok(Number.isFinite(clustering.metrics.minimum_split_half_ari));
    assert.equal(
      clustering.profiles.reduce((total, row) => total + row.size, 0),
      data.panel.consistencyQuestions,
    );
    for (const row of clustering.profiles) {
      assert.ok(row.exemplars.length > 0);
      assert.ok(row.domains > 0);
      assert.ok(row.sources > 0);
      assert.ok(row.strictPairs >= 0);
    }
  }
});

test("the committed browser snapshot matches the canonical adapters", async () => {
  const expected = await buildDashboardData();
  const committed = JSON.parse(await readFile(new URL("../public/data/dashboard.json", import.meta.url), "utf8"));
  assert.deepEqual(committed, expected);
});
