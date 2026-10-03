import assert from "node:assert/strict";
import test from "node:test";

import {
  buildPredictionRecord,
  getEvaluationSummary,
  getPredictionConfig,
} from "./prediction.js";

function trainedModel(problemType, targetColumn, metrics, rawFeatureNames) {
  return {
    automl_training: {
      problem_type: problemType,
      target_column: targetColumn,
      model: { problem_type: problemType, target_column: targetColumn, name: "Saved model" },
      best_model: { name: "Saved model", metrics },
      evaluation: {
        results: [{ model_name: "Saved model", metrics }],
      },
      preprocessing: {
        target_column: targetColumn,
        raw_feature_names: rawFeatureNames,
      },
    },
  };
}

test("classification config exposes raw features and holdout accuracy/F1", () => {
  const result = trainedModel(
    "Binary Classification",
    "Exited",
    { accuracy: 0.91, f1_score: 0.88 },
    ["Age", "Balance", "Exited"],
  );

  const config = getPredictionConfig(result);
  assert.equal(config.problemType, "classification");
  assert.deepEqual(config.featureNames, ["Age", "Balance"]);
  assert.deepEqual(config.metrics.map(({ label, value }) => [label, value]), [
    ["Holdout Accuracy", 0.91],
    ["Holdout F1", 0.88],
  ]);
});

test("prediction config falls back to best-model metrics when evaluation results are missing", () => {
  const result = trainedModel(
    "Regression",
    "price",
    { rmse: 12.4, mae: 8.2, r2: 0.79 },
    ["size", "price"],
  );
  delete result.automl_training.evaluation.results;

  const config = getPredictionConfig(result);

  assert.deepEqual(config.metrics.map(({ label, value }) => [label, value]), [
    ["Holdout RMSE", 12.4],
    ["Holdout MAE", 8.2],
    ["Holdout R²", 0.79],
  ]);
});

test("prediction config falls back to best-model metrics when no evaluation model matches", () => {
  const result = trainedModel(
    "Regression",
    "price",
    { rmse: 12.4, mae: 8.2, r2: 0.79 },
    ["size", "price"],
  );
  result.automl_training.evaluation.results = [
    { model_name: "Different model", metrics: { rmse: 99 } },
  ];

  const config = getPredictionConfig(result);

  assert.deepEqual(config.metrics.map(({ label, value }) => [label, value]), [
    ["Holdout RMSE", 12.4],
    ["Holdout MAE", 8.2],
    ["Holdout R²", 0.79],
  ]);
});

test("regression config exposes holdout RMSE, MAE, and R2", () => {
  const result = trainedModel(
    "Regression",
    "price",
    { rmse: 12.4, mae: 8.2, r2: 0.79 },
    ["size", "age", "price"],
  );

  const config = getPredictionConfig(result);
  assert.equal(config.problemType, "regression");
  assert.deepEqual(config.metrics.map(({ label, value }) => [label, value]), [
    ["Holdout RMSE", 12.4],
    ["Holdout MAE", 8.2],
    ["Holdout R²", 0.79],
  ]);
});

test("prediction record rejects invalid numeric input", () => {
  const result = trainedModel("Regression", "price", {}, ["size", "price"]);

  for (const value of ["not-a-number", "Infinity"]) {
    assert.throws(
      () =>
        buildPredictionRecord({
          trainingResult: result,
          predictionMode: "new",
          predictionInputs: { size: value },
          featureTypes: { size: "number" },
        }),
      /'size' must be a valid number\./,
    );
  }
});

test("clustering config filters persisted stroke target and predicts from clustering features", () => {
  const result = trainedModel(
    "Clustering",
    "stroke",
    { silhouette_score: 0.42 },
    ["age", "hypertension", "avg_glucose_level", "stroke"],
  );
  const config = getPredictionConfig(result);

  assert.equal(config.problemType, "clustering");
  assert.deepEqual(config.featureNames, ["age", "hypertension", "avg_glucose_level"]);
  assert.deepEqual(config.metrics.map(({ label, value }) => [label, value]), [
    ["Silhouette score", 0.42],
  ]);
  assert.deepEqual(
    buildPredictionRecord({
      trainingResult: result,
      predictionMode: "sample",
      sampleRow: { age: 67, hypertension: 1, avg_glucose_level: 180, stroke: 1 },
    }),
    { age: 67, hypertension: 1, avg_glucose_level: 180 },
  );
});

test("clustering reads excluded target from persisted preprocessing metadata", () => {
  const result = trainedModel(
    "Clustering",
    null,
    { silhouette_score: 0.42 },
    ["age", "hypertension", "stroke"],
  );
  result.automl_training.preprocessing.target_column = "stroke";

  const config = getPredictionConfig(result);
  assert.deepEqual(config.featureNames, ["age", "hypertension"]);
  assert.deepEqual(
    buildPredictionRecord({
      trainingResult: result,
      predictionMode: "new",
      predictionInputs: { age: "67", hypertension: "1", stroke: "1" },
      featureTypes: { age: "number", hypertension: "number", stroke: "number" },
    }),
    { age: 67, hypertension: 1 },
  );
});

test("prediction config excludes target columns case-insensitively", () => {
  const result = trainedModel(
    "Binary Classification",
    "Exited",
    { accuracy: 0.91, f1_score: 0.88 },
    ["Age", "exited", "Balance"],
  );

  const config = getPredictionConfig(result);

  assert.deepEqual(config.featureNames, ["Age", "Balance"]);
});

test("classification evaluation summary exposes holdout, balanced, macro, and class metrics", () => {
  const result = trainedModel(
    "Binary Classification",
    "outcome",
    {
      accuracy: 0.9,
      precision: 0.81,
      recall: 0.9,
      f1_score: 0.8526,
      balanced_accuracy: 0.5,
      macro_precision: 0.45,
      macro_recall: 0.5,
      macro_f1_score: 0.4737,
      confusion_matrix: [[90, 0], [10, 0]],
      confusion_matrix_labels: ["0", "1"],
      per_class_metrics: {
        "0": { precision: 0.9, recall: 1.0, f1_score: 0.9474, support: 90 },
        "1": { precision: 0.0, recall: 0.0, f1_score: 0.0, support: 10 },
      },
    },
    ["feature", "outcome"],
  );
  result.automl_training.preprocessing.target_encoded = true;
  result.automl_training.preprocessing.target_classes = ["negative", "positive"];

  const summary = getEvaluationSummary(result);

  assert.equal(summary.problemType, "classification");
  assert.equal(summary.metrics.accuracy, 0.9);
  assert.equal(summary.metrics.f1_score, 0.8526);
  assert.equal(summary.metrics.balanced_accuracy, 0.5);
  assert.deepEqual(summary.confusionMatrix, [[90, 0], [10, 0]]);
  assert.deepEqual(summary.confusionMatrixLabels, ["negative", "positive"]);
  assert.equal(summary.perClassMetrics[1].label, "positive");
  assert.equal(summary.perClassMetrics[1].recall, 0.0);
  assert.equal(summary.perClassMetrics[1].support, 10);
});

test("evaluation summary handles missing optional classification metrics", () => {
  const result = trainedModel(
    "Binary Classification",
    "outcome",
    { accuracy: 0.9, f1_score: 0.85 },
    ["feature", "outcome"],
  );

  const summary = getEvaluationSummary(result);

  assert.equal(summary.metrics.accuracy, 0.9);
  assert.equal(summary.metrics.balanced_accuracy, undefined);
  assert.equal(summary.confusionMatrix, null);
  assert.deepEqual(summary.confusionMatrixLabels, []);
  assert.deepEqual(summary.perClassMetrics, []);
});

test("evaluation summary remains compatible with regression and clustering", () => {
  const regression = getEvaluationSummary(
    trainedModel("Regression", "price", { rmse: 4.2, mae: 3.1, r2: 0.7 }, ["size", "price"]),
  );
  const clustering = getEvaluationSummary(
    trainedModel("Clustering", null, { silhouette_score: 0.42 }, ["size"]),
  );

  assert.equal(regression.problemType, "regression");
  assert.equal(regression.metrics.rmse, 4.2);
  assert.equal(regression.confusionMatrix, null);
  assert.equal(clustering.problemType, "clustering");
  assert.equal(clustering.metrics.silhouette_score, 0.42);
  assert.deepEqual(clustering.perClassMetrics, []);
});
