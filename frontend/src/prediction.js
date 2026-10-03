function normalizeProblemType(value) {
  const problemType = String(value || "").trim().toLowerCase();
  if (problemType.includes("cluster")) return "clustering";
  if (problemType.includes("regression")) return "regression";
  if (problemType.includes("classification")) return "classification";
  return "unknown";
}

function getMetrics(training, model) {
  const modelName = training.best_model?.name || model.name;
  const holdoutResult = training.evaluation?.results?.find(
    (result) => result.model_name === modelName,
  );
  return holdoutResult?.metrics || training.best_model?.metrics || model.metrics || {};
}

export function getEvaluationSummary(trainingResult) {
  const training = trainingResult?.automl_training || {};
  const model = training.model || {};
  const preprocessing = training.preprocessing || {};
  const problemType = normalizeProblemType(model.problem_type || training.problem_type);
  const metrics = getMetrics(training, model);
  const rawLabels = Array.isArray(metrics.confusion_matrix_labels)
    ? metrics.confusion_matrix_labels.map(String)
    : Object.keys(metrics.per_class_metrics || {});
  const targetClasses = preprocessing.target_classes || [];
  const displayLabel = (label) => {
    if (!preprocessing.target_encoded || !Array.isArray(targetClasses)) return label;
    const index = Number(label);
    return Number.isInteger(index) && targetClasses[index] !== undefined
      ? String(targetClasses[index])
      : label;
  };
  const matrix = metrics.confusion_matrix;
  const confusionMatrix =
    Array.isArray(matrix) &&
    matrix.length === rawLabels.length &&
    matrix.every((row) => Array.isArray(row) && row.length === rawLabels.length)
      ? matrix
      : null;
  const perClass = metrics.per_class_metrics || {};
  const perClassMetrics = rawLabels
    .filter((label) => perClass[label] && typeof perClass[label] === "object")
    .map((label) => ({ label: displayLabel(label), ...perClass[label] }));

  return {
    problemType,
    metrics,
    confusionMatrix,
    confusionMatrixLabels: rawLabels.map(displayLabel),
    perClassMetrics,
  };
}

export function getPredictionConfig(trainingResult) {
  const training = trainingResult?.automl_training || {};
  const model = training.model || {};
  const preprocessing = training.preprocessing || {};
  const evaluation = getEvaluationSummary(trainingResult);
  const problemType = evaluation.problemType;
  const targetColumn =
    model.target_column || training.target_column || preprocessing.target_column || "";
  const normalizedTarget = String(targetColumn).trim().toLowerCase();
  const featureNames = Array.isArray(preprocessing.raw_feature_names)
    ? [...new Set(preprocessing.raw_feature_names.map(String))]
    : [];

  const metrics = evaluation.metrics;
  const metricDefinitions = {
    classification: [
      { label: "Holdout Accuracy", key: "accuracy", format: "percentage" },
      { label: "Holdout F1", key: "f1_score", format: "percentage" },
    ],
    regression: [
      { label: "Holdout RMSE", key: "rmse", format: "number" },
      { label: "Holdout MAE", key: "mae", format: "number" },
      { label: "Holdout R²", key: "r2", format: "number" },
    ],
    clustering: [
      { label: "Silhouette score", key: "silhouette_score", format: "number" },
    ],
  };

  return {
    problemType,
    featureNames: featureNames.filter(
      (name) => !normalizedTarget || name.trim().toLowerCase() !== normalizedTarget,
    ),
    metrics: (metricDefinitions[problemType] || []).map((metric) => ({
      ...metric,
      value: metrics[metric.key] ?? null,
    })),
  };
}

export function buildPredictionRecord({
  trainingResult,
  predictionMode,
  predictionInputs,
  sampleRow,
  featureTypes,
}) {
  const { featureNames } = getPredictionConfig(trainingResult);
  if (predictionMode === "sample") {
    return Object.fromEntries(
      featureNames
        .filter((column) => Object.prototype.hasOwnProperty.call(sampleRow || {}, column))
        .map((column) => [column, sampleRow[column]]),
    );
  }

  const record = {};
  for (const column of featureNames) {
    const value = predictionInputs?.[column];
    if (value === undefined || value === "") continue;

    const type = String(featureTypes?.[column] || "").toLowerCase();
    if (type.includes("bool")) {
      const normalized = String(value).trim().toLowerCase();
      if (!["true", "false"].includes(normalized)) {
        throw new Error(`'${column}' must be true or false.`);
      }
      record[column] = normalized === "true";
    } else if (/(int|float|double|number)/.test(type)) {
      const numericValue = Number(value);
      if (!Number.isFinite(numericValue)) {
        throw new Error(`'${column}' must be a valid number.`);
      }
      record[column] = numericValue;
    } else {
      record[column] = value;
    }
  }
  return record;
}
