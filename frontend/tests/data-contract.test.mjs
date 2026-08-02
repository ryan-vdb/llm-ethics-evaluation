import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { buildDashboardData } from "../scripts/sync-data.mjs";

test("dashboard data preserves the canonical panel and headline results", async () => {
  const data = await buildDashboardData();
  assert.equal(data.schemaVersion, 1);
  assert.ok(data.panel.models.length >= 2);
  assert.ok(data.panel.consistencyQuestions > 0);
  assert.ok(data.panel.integrityQuestions > 0);
  assert.ok(data.panel.embeddingDimensions > 0);
  assert.equal(data.consistency.heldOutModels.length, data.panel.models.length);
  assert.equal(data.integrity.conditions.length, data.panel.integrityConditions - 1);
  assert.equal(data.integrity.scenarioEffects.length, data.panel.integrityQuestions);
  assert.equal(data.integrity.modelEffects.length, data.panel.models.length);
  assert.ok(Number.isFinite(data.consistency.headline.meanHeldOutRho));
  assert.ok(Number.isFinite(data.integrity.headline.extraRevision));
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
});

test("the committed browser snapshot matches the canonical adapters", async () => {
  const expected = await buildDashboardData();
  const committed = JSON.parse(await readFile(new URL("../public/data/dashboard.json", import.meta.url), "utf8"));
  assert.deepEqual(committed, expected);
});
