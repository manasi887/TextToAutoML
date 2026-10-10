export function getGroupColumnOptions(columns, targetColumn) {
  if (!Array.isArray(columns)) return [];
  return columns.filter(
    (column) => typeof column === "string" && column !== targetColumn,
  );
}

export function getFeatureColumnOptions(columns, targetColumn, groupColumn) {
  if (!Array.isArray(columns)) return [];
  const excludedColumns = new Set([targetColumn, groupColumn].filter(Boolean));
  return columns.filter(
    (column) => typeof column === "string" && !excludedColumns.has(column),
  );
}

export function buildNlpTrainingRequest({
  filename,
  text,
  groupColumn,
  featureColumns,
  tuningStrategy,
  tuningTrials,
}) {
  const request = {
    filename,
    text,
    group_column: groupColumn || null,
  };
  if (Array.isArray(featureColumns)) {
    request.feature_columns = featureColumns;
  }
  if (["auto", "grid", "bayesian"].includes(tuningStrategy)) {
    request.tuning_strategy = tuningStrategy;
  }
  if (tuningStrategy === "bayesian") {
    request.tuning_trials = Number.isInteger(tuningTrials) ? tuningTrials : 8;
  }
  return request;
}

export function getTuningDecision(trainingResult) {
  const tuning = trainingResult?.automl_training?.tuning || trainingResult?.tuning;
  if (!tuning || typeof tuning !== "object") return null;

  const strategy = tuning.selected_strategy || tuning.strategy;
  if (typeof strategy !== "string" || !strategy) return null;

  const trialBudget = Number.isFinite(tuning.trial_budget)
    ? tuning.trial_budget
    : Number.isFinite(tuning.trials_per_model)
      ? tuning.trials_per_model
      : null;

  return {
    strategy,
    trialBudget,
    budgetUnit: typeof tuning.budget_unit === "string" ? tuning.budget_unit : null,
    reason: typeof tuning.reason === "string" ? tuning.reason : null,
  };
}

export function createLatestRequestTracker() {
  let latestRequest = 0;
  return {
    begin() {
      latestRequest += 1;
      return latestRequest;
    },
    isLatest(requestId) {
      return requestId === latestRequest;
    },
    invalidate(requestId) {
      if (requestId === latestRequest) latestRequest += 1;
    },
  };
}