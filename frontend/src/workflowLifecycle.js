export const TABULAR_DATASET_RESET = Object.freeze({
  analyzeState: "idle",
  analyzeError: "",
  analyzeResult: null,
  trainingConfirmed: false,
  groupColumn: "",
  trainingState: "idle",
  trainingError: "",
  trainingResult: null,
  excludedFeatureColumns: [],
  diagnosticState: "idle",
  diagnosticError: "",
  diagnosticResult: null,
  diagnosticGroupColumn: null,
  diagnosticRetry: 0,
  predictionState: "idle",
  predictionError: "",
  predictionMode: "sample",
  predictionInputs: {},
  predictionResult: null,
});

export const IMAGE_DATASET_RESET = Object.freeze({
  imageTrainingState: "idle",
  imageTrainingError: "",
  imageTrainingResult: null,
  imageModelId: "",
  imagePredictionFile: null,
  imagePredictionState: "idle",
  imagePredictionError: "",
  imagePredictionResult: null,
});

function applyResetValues(resetValues, setters) {
  for (const [key, value] of Object.entries(resetValues)) {
    setters[key]?.(value);
  }
}

export function resetTabularDatasetWorkflow(setters) {
  applyResetValues(TABULAR_DATASET_RESET, setters);
}

export function resetImageDatasetWorkflow(setters) {
  applyResetValues(IMAGE_DATASET_RESET, setters);
}

export function getUploadValidationError(payload) {
  if (payload?.validation?.valid === true) return null;

  const parts = [];
  if (payload?.validation?.valid === false && typeof payload.message === "string") {
    parts.push(payload.message);
  }
  if (Array.isArray(payload?.validation?.errors)) {
    parts.push(
      ...payload.validation.errors.filter(
        (error) => typeof error === "string" && error.trim(),
      ),
    );
  }
  return parts.length > 0
    ? [...new Set(parts)].join(" ")
    : "The server did not confirm that dataset validation passed.";
}

export function getNlpTrainingOutcome(payload) {
  const training = payload?.automl_training;
  const status = typeof training?.status === "string"
    ? training.status.trim().toLowerCase()
    : "";
  const successful = payload?.ready_for_training === true &&
    ["success", "completed"].includes(status);

  if (successful) return { successful: true, message: "", result: payload };

  const messageCandidates = [
    training?.error,
    training?.failure_reason,
    training?.best_model?.reason,
    payload?.reason,
  ];
  const message = messageCandidates.find(
    (candidate) => typeof candidate === "string" && candidate.trim(),
  );

  return {
    successful: false,
    message: message || "AutoML training did not complete successfully.",
    result: null,
  };
}

export function shouldShowTargetClasses(problemType) {
  return typeof problemType === "string" &&
    problemType.toLowerCase().includes("classification");
}

export function shouldRunTargetProxyDiagnostic({
  datasetType,
  diagnosticApplicable,
  fileName,
  task,
}) {
  return Boolean(
    datasetType === "tabular" &&
      diagnosticApplicable &&
      fileName &&
      typeof task === "string" &&
      task.trim(),
  );
}