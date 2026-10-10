import assert from "node:assert/strict";
import test from "node:test";

import {
  buildNlpTrainingRequest,
  createLatestRequestTracker,
  getFeatureColumnOptions,
  getGroupColumnOptions,
  getTuningDecision,
} from "./training.js";

test("group selector options come from uploaded columns and exclude the target", () => {
  assert.deepEqual(
    getGroupColumnOptions(["student_id", "hours", "completed"], "completed"),
    ["student_id", "hours"],
  );
  assert.deepEqual(getGroupColumnOptions(null, "completed"), []);
});

test("training request sends the selected group column", () => {
  assert.deepEqual(
    buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
      groupColumn: "student_id",
    }),
    {
      filename: "courses.csv",
      text: "Predict completion",
      group_column: "student_id",
    },
  );
});

test("training request sends null when no group is selected", () => {
  assert.deepEqual(
    buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
      groupColumn: "",
    }),
    {
      filename: "courses.csv",
      text: "Predict completion",
      group_column: null,
    },
  );
});

test("request tracker ignores responses after a newer request starts", () => {
  const tracker = createLatestRequestTracker();
  const olderRequest = tracker.begin();
  const newerRequest = tracker.begin();

  assert.equal(tracker.isLatest(olderRequest), false);
  assert.equal(tracker.isLatest(newerRequest), true);

  tracker.invalidate(newerRequest);
  assert.equal(tracker.isLatest(newerRequest), false);
});

test("feature options exclude the target and selected group column", () => {
  assert.deepEqual(
    getFeatureColumnOptions(
      ["student_id", "income", "age", "hours"],
      "income",
      "student_id",
    ),
    ["age", "hours"],
  );
});

test("training request sends the selected feature schema", () => {
  assert.deepEqual(
    buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
      groupColumn: "student_id",
      featureColumns: ["hours", "quiz_average"],
    }),
    {
      filename: "courses.csv",
      text: "Predict completion",
      group_column: "student_id",
      feature_columns: ["hours", "quiz_average"],
    },
  );
});

test("Bayesian tuning selection and trial budget are sent only when requested", () => {
  assert.deepEqual(
    buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
      tuningStrategy: "bayesian",
      tuningTrials: 12,
    }),
    {
      filename: "courses.csv",
      text: "Predict completion",
      group_column: null,
      tuning_strategy: "bayesian",
      tuning_trials: 12,
    },
  );
  assert.equal(
    "tuning_strategy" in buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
    }),
    false,
  );
});

test("automatic and manual grid tuning strategies are serialized explicitly", () => {
  assert.deepEqual(
    buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
      tuningStrategy: "auto",
    }),
    {
      filename: "courses.csv",
      text: "Predict completion",
      group_column: null,
      tuning_strategy: "auto",
    },
  );
  assert.deepEqual(
    buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
      tuningStrategy: "grid",
    }),
    {
      filename: "courses.csv",
      text: "Predict completion",
      group_column: null,
      tuning_strategy: "grid",
    },
  );
});

test("tuning decision reads actual backend metadata and tolerates older responses", () => {
  assert.deepEqual(
    getTuningDecision({
      automl_training: {
        tuning: {
          requested_mode: "auto",
          selected_strategy: "random",
          trial_budget: 3,
          budget_unit: "trials_per_model",
          reason: "The dataset has a moderate training workload.",
        },
      },
    }),
    {
      strategy: "random",
      trialBudget: 3,
      budgetUnit: "trials_per_model",
      reason: "The dataset has a moderate training workload.",
    },
  );
  assert.equal(getTuningDecision({ automl_training: { status: "success" } }), null);
  assert.deepEqual(
    getTuningDecision({ tuning: { strategy: "bayesian", trials_per_model: 5 } }),
    {
      strategy: "bayesian",
      trialBudget: 5,
      budgetUnit: null,
      reason: null,
    },
  );
});