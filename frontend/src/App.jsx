import { useEffect, useRef, useState } from "react";
import {
  buildPredictionRecord as createPredictionRecord,
  buildPredictionCsv,
  getPredictionConfig,
  getEvaluationSummary,
} from "./prediction.js";
import {
  buildNlpTrainingRequest,
  createLatestRequestTracker,
  getFeatureColumnOptions,
  getGroupColumnOptions,
  getTuningDecision,
} from "./training.js";
import {
  getNlpTrainingOutcome,
  getUploadValidationError,
  resetImageDatasetWorkflow,
  resetTabularDatasetWorkflow,
  shouldShowTargetClasses,
  shouldRunTargetProxyDiagnostic,
} from "./workflowLifecycle.js";
import {
  analyzeImageDataset,
  predictImageModel,
  routeImageRequest,
  trainImageModel,
} from "./image.js";

const pageStyle = {
  minHeight: "100vh",
  background: "#f4f7f5",
  color: "#17221d",
  fontFamily: "Georgia, 'Times New Roman', serif",
};

const shellStyle = {
  width: "min(1120px, calc(100% - 40px))",
  margin: "0 auto",
};

const panelStyle = {
  background: "#ffffff",
  border: "1px solid #dbe5de",
  borderRadius: "12px",
  boxShadow: "0 14px 40px rgba(33, 63, 45, 0.07)",
};

const tuningStrategyLabels = {
  auto: "Automatic selection",
  none: "No tuning",
  grid: "Grid Search",
  random: "Random Search",
  bayesian: "Bayesian Optimization (Optuna TPE)",
};

const imageIntentLabels = {
  IMAGE_TRAINING_REQUEST: "Image Training",
  IMAGE_PREDICTION: "Image Prediction",
  IMAGE_TRAINING_REPORT: "Image Training Report",
  IMAGE_DATASET_ANALYSIS: "Image Dataset Analysis",
  IMAGE_MODEL_EXPLANATION: "Image Model Explanation",
  UNKNOWN_AMBIGUOUS: "Needs Clarification",
};

const imageContextLabels = {
  image: "a single image",
  image_dataset: "an image dataset ZIP",
  image_model_id: "a trained image model",
  image_training_result: "an image training result",
};

function formatDuration(seconds) {
  if (seconds < 60) {
    return `${Math.round(seconds)} seconds`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return remainingSeconds
    ? `${minutes} min ${remainingSeconds} sec`
    : `${minutes} min`;
}

function displayAnalysisNumber(value, maximumFractionDigits = 2) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not available";
  return value.toLocaleString(undefined, { maximumFractionDigits });
}

function displayAnalysisBytes(value) {
  const formatted = displayAnalysisNumber(value, 0);
  return formatted === "Not available" ? formatted : `${formatted} bytes`;
}

function analysisEntries(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? Object.entries(value)
    : [];
}

function formatPredictionMetric(metric) {
  if (metric.value === null || metric.value === undefined) return "Not available";
  const value = Number(metric.value);
  if (!Number.isFinite(value)) return "Not available";
  return metric.format === "percentage"
    ? `${(value * 100).toFixed(1)}%`
    : value.toFixed(3);
}

function formatProblemType(value) {
  const problemType = String(value || "").trim().toLowerCase();
  if (problemType.includes("multi")) return "multi-class classification";
  if (problemType.includes("binary")) return "binary classification";
  if (problemType.includes("classification")) return "classification";
  if (problemType.includes("regression")) return "regression";
  if (problemType.includes("cluster")) return "clustering";
  return value || "not available";
}

function formatColumnType(value) {
  const dataType = String(value || "").toLowerCase();
  if (dataType.includes("bool")) return "boolean";
  if (/(int|float|double|number|decimal)/.test(dataType)) return "numeric";
  if (/(datetime|date|time)/.test(dataType)) return "date/time";
  if (/(object|string|category)/.test(dataType)) return "categorical";
  return value || "type unavailable";
}

function formatMetricLabel(metric) {
  const labels = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1_score: "F1-score",
    balanced_accuracy: "Balanced accuracy",
    macro_precision: "Macro precision",
    macro_recall: "Macro recall",
    macro_f1_score: "Macro F1",
    rmse: "RMSE",
    mae: "MAE",
    r2: "R²",
    r2_score: "R²",
    silhouette_score: "Silhouette score",
  };
  return labels[metric] || metric || "Not available";
}

function getRemovedColumnReason(removedColumns, column) {
  const reasonLabels = {
    identifier_columns: "identified as an identifier",
    constant_columns: "contains a constant value",
    group_columns: "used as the group column",
    target_columns: "excluded as a target candidate",
    clustering_label_columns: "identified as an existing outcome, segment, or cluster label",
  };
  for (const [reason, columns] of Object.entries(removedColumns || {})) {
    if (Array.isArray(columns) && columns.includes(column)) {
      return reasonLabels[reason] || "excluded during preprocessing";
    }
  }
  return null;
}

function summarizePerClassF1(perClassMetrics) {
  const availableMetrics = (perClassMetrics || []).filter(
    (item) => Number.isFinite(Number(item.f1_score)),
  );
  if (availableMetrics.length === 0) return null;

  const highest = Math.max(...availableMetrics.map((item) => Number(item.f1_score)));
  const lowest = Math.min(...availableMetrics.map((item) => Number(item.f1_score)));
  const highestClasses = availableMetrics
    .filter((item) => Number(item.f1_score) === highest)
    .map((item) => item.label)
    .join(", ");
  const lowestClasses = availableMetrics
    .filter((item) => Number(item.f1_score) === lowest)
    .map((item) => item.label)
    .join(", ");

  return {
    highest: `${highestClasses} (${(highest * 100).toFixed(1)}%)`,
    lowest: `${lowestClasses} (${(lowest * 100).toFixed(1)}%)`,
  };
}

export default function App() {
  const [datasetType, setDatasetType] = useState("tabular");
  const [fileName, setFileName] = useState("");
  const [task, setTask] = useState("");
  const [uploadState, setUploadState] = useState("idle");
  const [uploadError, setUploadError] = useState("");
  const [uploadResult, setUploadResult] = useState(null);
  const [analyzeState, setAnalyzeState] = useState("idle");
  const [analyzeError, setAnalyzeError] = useState("");
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [trainingConfirmed, setTrainingConfirmed] = useState(false);
  const [groupColumn, setGroupColumn] = useState("");
  const [trainingState, setTrainingState] = useState("idle");
  const [trainingError, setTrainingError] = useState("");
  const [trainingResult, setTrainingResult] = useState(null);
  const [tuningStrategy, setTuningStrategy] = useState("auto");
  const [manualTuningStrategy, setManualTuningStrategy] = useState("");
  const [tuningTrials, setTuningTrials] = useState("8");
  const [excludedFeatureColumns, setExcludedFeatureColumns] = useState([]);
  const [diagnosticState, setDiagnosticState] = useState("idle");
  const [diagnosticError, setDiagnosticError] = useState("");
  const [diagnosticResult, setDiagnosticResult] = useState(null);
  const [diagnosticGroupColumn, setDiagnosticGroupColumn] = useState(null);
  const [diagnosticRetry, setDiagnosticRetry] = useState(0);
  const diagnosticTrackerRef = useRef(null);
  if (diagnosticTrackerRef.current === null) {
    diagnosticTrackerRef.current = createLatestRequestTracker();
  }
  const tabularWorkflowRequestRef = useRef(0);
  const savedPredictionRequestRef = useRef(0);
  const imageTrainingRequestRef = useRef(0);
  const imagePredictionRequestRef = useRef(0);
  const imagePredictionInFlightRef = useRef(null);
  const [predictionState, setPredictionState] = useState("idle");
  const [predictionError, setPredictionError] = useState("");
  const [predictionMode, setPredictionMode] = useState("sample");
  const [predictionInputs, setPredictionInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [savedModelIdInput, setSavedModelIdInput] = useState("");
  const [savedModelFile, setSavedModelFile] = useState(null);
  const [savedModelPredictionState, setSavedModelPredictionState] = useState("idle");
  const [savedModelPredictionError, setSavedModelPredictionError] = useState("");
  const [savedModelPredictionResult, setSavedModelPredictionResult] = useState(null);
  const [imageZipFile, setImageZipFile] = useState(null);
  const [imageZipValidationError, setImageZipValidationError] = useState("");
  const [imageRoutingState, setImageRoutingState] = useState("idle");
  const [imageRoutingError, setImageRoutingError] = useState("");
  const [imageRoutingResult, setImageRoutingResult] = useState(null);
  const [imageRoutedText, setImageRoutedText] = useState("");
  const imageRoutingRequestRef = useRef(0);
  const [imageAnalysisResult, setImageAnalysisResult] = useState(null);
  const [imageAnalysisLoading, setImageAnalysisLoading] = useState(false);
  const [imageAnalysisError, setImageAnalysisError] = useState("");
  const [imageTrainingState, setImageTrainingState] = useState("idle");
  const [imageTrainingError, setImageTrainingError] = useState("");
  const [imageTrainingResult, setImageTrainingResult] = useState(null);
  const [imageModelId, setImageModelId] = useState("");
  const [imageModelSource, setImageModelSource] = useState("none");
  const [existingImageModelIdInput, setExistingImageModelIdInput] = useState("");
  const [existingImageModelIdError, setExistingImageModelIdError] = useState("");
  const [imagePredictionFile, setImagePredictionFile] = useState(null);
  const [imagePredictionState, setImagePredictionState] = useState("idle");
  const [imagePredictionError, setImagePredictionError] = useState("");
  const [imagePredictionResult, setImagePredictionResult] = useState(null);
  const predictionConfig = getPredictionConfig(trainingResult);
  const evaluationSummary = getEvaluationSummary(trainingResult);
  const tuningDecision = getTuningDecision(trainingResult);
  const trainedTabularModelId =
    trainingResult?.automl_training?.model?.model_id ||
    trainingResult?.report?.model_id ||
    "";
  const resolvedTargetColumn =
    analyzeResult?.dataset_resolution?.target_column ||
    analyzeResult?.target?.matched_column;
  const sampleRow = uploadResult?.sample_rows?.[0];
  const uploadedColumns = Object.keys(
    sampleRow && Object.keys(sampleRow).length > 0
      ? sampleRow
      : uploadResult?.analysis?.data_types || {},
  );
  const groupColumnOptions = getGroupColumnOptions(
    uploadedColumns,
    resolvedTargetColumn,
  );
  const featureColumnOptions = getFeatureColumnOptions(
    uploadedColumns,
    resolvedTargetColumn,
    groupColumn,
  );
  const selectedFeatureColumns = featureColumnOptions.filter(
    (column) => !excludedFeatureColumns.includes(column),
  );
  const resolvedProblemType = String(
    analyzeResult?.dataset_resolution?.problem_type || "",
  ).toLowerCase();
  const diagnosticApplicable = Boolean(
    analyzeState === "success" &&
      analyzeResult?.ready_for_training &&
      resolvedTargetColumn &&
      !resolvedProblemType.includes("cluster"),
  );
  const diagnosticReady = !diagnosticApplicable || (
    diagnosticState === "success" && diagnosticGroupColumn === groupColumn
  );
  const automlTraining = trainingResult?.automl_training || {};
  const trainingData = automlTraining.data || {};
  const trainingPreprocessing = automlTraining.preprocessing || {};
  const bestModel = automlTraining.best_model || {};
  const trainingProblemType =
    automlTraining.problem_type ||
    trainingResult?.report?.task?.problem_type ||
    analyzeResult?.dataset_resolution?.problem_type ||
    "";
  const isClassificationTask = shouldShowTargetClasses(trainingProblemType);
  const isClusteringTask = String(trainingProblemType).toLowerCase().includes("cluster");
  const trainingTargetColumn =
    isClusteringTask
      ? ""
      : resolvedTargetColumn ||
        automlTraining.target_column ||
        trainingPreprocessing.target_column ||
        trainingResult?.report?.task?.target ||
        "";
  const datasetColumns = Array.isArray(uploadResult?.analysis?.columns)
    ? uploadResult.analysis.columns
    : Object.keys(uploadResult?.analysis?.data_types || {});
  const trainingFeatureNames = Array.isArray(trainingPreprocessing.raw_feature_names)
    ? trainingPreprocessing.raw_feature_names
    : null;
  const displayedFeatureNames = trainingFeatureNames || selectedFeatureColumns;
  const removedColumnEntries = Array.isArray(trainingPreprocessing.excluded_features)
    ? trainingPreprocessing.excluded_features.filter(
      (item) => item && typeof item.column === "string",
    )
    : Object.entries(trainingPreprocessing.removed_columns || {}).flatMap(
      ([reason, columns]) =>
        Array.isArray(columns) ? columns.map((column) => ({ column, reason })) : [],
    );
  const encoderEntries = Object.entries(trainingPreprocessing.encoders || {}).filter(
    ([, encoder]) => encoder && typeof encoder.type === "string",
  );
  const tuningMetadata = automlTraining.tuning || trainingResult?.tuning || null;
  const selectionMetric = bestModel.selection_metric || "";
  const selectionScore = isClusteringTask
    ? bestModel.metrics?.[selectionMetric]
    : bestModel.validation_metrics?.[selectionMetric];
  const perClassF1Summary = summarizePerClassF1(evaluationSummary.perClassMetrics);
  const imageAnalysisClasses = Array.isArray(imageAnalysisResult?.classes?.names)
    ? imageAnalysisResult.classes.names
    : [];
  const imageAnalysisFindings = Array.isArray(imageAnalysisResult?.quality?.findings)
    ? imageAnalysisResult.quality.findings
    : null;
  const imageAnalysisSplitCounts = analysisEntries(imageAnalysisResult?.splits?.counts);
  const imageAnalysisPerClassSplits = analysisEntries(
    imageAnalysisResult?.splits?.per_class_counts,
  );
  const imageAnalysisProperties = imageAnalysisResult?.image_properties || {};
  const imageQualityMetrics = imageAnalysisResult?.quality?.metrics || {};
  const imageDuplicateMetrics = imageQualityMetrics.duplicates || {};
  const imageDuplicateGroups = Array.isArray(imageDuplicateMetrics.groups)
    ? imageDuplicateMetrics.groups
    : [];
  const imageCrossSplitMetrics = imageQualityMetrics.cross_split_duplicates || {};
  const imageCrossSplitMatches = Array.isArray(imageCrossSplitMetrics.matches)
    ? imageCrossSplitMetrics.matches
    : [];

  useEffect(() => {
    const tracker = diagnosticTrackerRef.current;
    const requestId = tracker.begin();
    const controller = new AbortController();

    if (!shouldRunTargetProxyDiagnostic({
      datasetType,
      diagnosticApplicable,
      fileName,
      task,
    })) {
      setDiagnosticState("idle");
      setDiagnosticError("");
      setDiagnosticResult(null);
      setDiagnosticGroupColumn(null);
      return () => {
        controller.abort();
        tracker.invalidate(requestId);
      };
    }

    setDiagnosticState("loading");
    setDiagnosticError("");
    setDiagnosticResult(null);
    setDiagnosticGroupColumn(null);

    async function loadDiagnostic() {
      try {
        const response = await fetch("/nlp/diagnose-target-proxies", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            buildNlpTrainingRequest({
              filename: fileName,
              text: task.trim(),
              groupColumn,
            }),
          ),
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) {
          const detail = payload.detail;
          throw new Error(
            typeof detail === "string"
              ? detail
              : detail?.message || "Target-proxy diagnostic failed.",
          );
        }
        if (!tracker.isLatest(requestId)) return;
        setDiagnosticResult(payload);
        setDiagnosticGroupColumn(groupColumn);
        setDiagnosticState("success");
      } catch (error) {
        if (controller.signal.aborted || !tracker.isLatest(requestId)) return;
        setDiagnosticError(error.message || "Target-proxy diagnostic failed.");
        setDiagnosticState("error");
      }
    }

    loadDiagnostic();
    return () => {
      controller.abort();
      tracker.invalidate(requestId);
    };
  }, [
    analyzeState,
    diagnosticApplicable,
    diagnosticRetry,
    datasetType,
    fileName,
    groupColumn,
    task,
  ]);

  function selectDatasetType(type) {
    if (type === datasetType) return;

    setDatasetType(type);
    clearTabularWorkflow();
    imageTrainingRequestRef.current += 1;
    imagePredictionRequestRef.current += 1;
    clearImageRouting();
    clearImageAnalysis();
    setImagePredictionFile(null);
    imagePredictionInFlightRef.current = null;
    setImagePredictionState("idle");
    setImagePredictionError("");
    setImagePredictionResult(null);
    setImageTrainingState((current) => current === "loading" ? "idle" : current);
    setImagePredictionState("idle");
    if (uploadState === "loading") {
      setUploadState("idle");
      setUploadError("");
    }
  }

  function clearTabularWorkflow() {
    tabularWorkflowRequestRef.current += 1;
    diagnosticTrackerRef.current.begin();
    resetTabularDatasetWorkflow({
      setAnalyzeState,
      setAnalyzeError,
      setAnalyzeResult,
      setTrainingConfirmed,
      setGroupColumn,
      setTrainingState,
      setTrainingError,
      setTrainingResult,
      setExcludedFeatureColumns,
      setDiagnosticState,
      setDiagnosticError,
      setDiagnosticResult,
      setDiagnosticGroupColumn,
      setDiagnosticRetry,
      setPredictionState,
      setPredictionError,
      setPredictionMode,
      setPredictionInputs,
      setPredictionResult,
    });
  }

  function handleTabularTaskChange(event) {
    clearTabularWorkflow();
    setTask(event.target.value);
  }

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file || datasetType !== "tabular") {
      return;
    }

    clearTabularWorkflow();
    setFileName(file.name);
    setUploadState("loading");
    setUploadError("");
    setUploadResult(null);
    const requestId = ++tabularWorkflowRequestRef.current;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/upload/", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (requestId !== tabularWorkflowRequestRef.current) return;

      if (!response.ok) {
        throw new Error(payload.detail || payload.message || "Upload failed.");
      }

      const validationError = getUploadValidationError(payload);
      if (validationError) {
        setUploadError(validationError);
        setUploadState("error");
        return;
      }

      setUploadResult(payload);
      setUploadState("success");
    } catch (error) {
      if (requestId !== tabularWorkflowRequestRef.current) return;
      setUploadState("error");
      setUploadError(error.message || "Upload failed. Please try again.");
    }
  }

  async function handleImageDatasetChange(event) {
    const file = event.target.files?.[0];
    if (!file || datasetType !== "image") return;
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setImageZipFile(null);
      setImageZipValidationError("Choose a ZIP file containing class folders of images.");
      clearImageRouting();
      return;
    }

    imageTrainingRequestRef.current += 1;
    setImageZipFile(file);
    setImageZipValidationError("");
    resetImageDatasetWorkflow({
      setImageTrainingState,
      setImageTrainingError,
      setImageTrainingResult,
    });
    clearImageRouting();
    clearImageAnalysis();
  }

  function clearImageAnalysis() {
    setImageAnalysisResult(null);
    setImageAnalysisLoading(false);
    setImageAnalysisError("");
  }

  function clearImageRouting() {
    imageRoutingRequestRef.current += 1;
    setImageRoutingState("idle");
    setImageRoutingError("");
    setImageRoutingResult(null);
    setImageRoutedText("");
  }

  function handleImageRequestChange(event) {
    setTask(event.target.value);
    clearImageRouting();
    clearImageAnalysis();
  }

  function useExistingImageModel() {
    const modelId = existingImageModelIdInput.trim();
    if (!modelId) {
      setExistingImageModelIdError("Enter a trained image model ID to continue.");
      return;
    }
    imagePredictionRequestRef.current += 1;
    imagePredictionInFlightRef.current = null;
    setImageModelId(modelId);
    setImageModelSource("existing");
    setExistingImageModelIdInput("");
    setExistingImageModelIdError("");
    setImagePredictionState("idle");
    setImagePredictionError("");
    setImagePredictionResult(null);
    clearImageRouting();
  }

  async function handleImageRoute() {
    const text = task.trim();
    if (!text) {
      setImageRoutingState("error");
      setImageRoutingError("Describe the image action you want before routing.");
      setImageRoutingResult(null);
      return;
    }

    const refersToSingleImage = /\b(?:this|that|single)\s+(?:image|photo|picture)\b/i.test(text);
    const assetType = imageZipFile
      ? "image_dataset"
      : imagePredictionFile || refersToSingleImage || imageModelId || imageTrainingResult
        ? "image"
        : "image_dataset";
    const requestId = ++imageRoutingRequestRef.current;
    clearImageAnalysis();
    setImageRoutingState("loading");
    setImageRoutingError("");
    setImageRoutingResult(null);
    setImageRoutedText("");

    try {
      const decision = await routeImageRequest({
        text,
        asset_type: assetType,
        image_dataset_available: Boolean(imageZipFile),
        image_available: Boolean(imagePredictionFile),
        image_model_id: imageModelId || null,
        image_training_result_available: Boolean(imageTrainingResult),
      });
      if (requestId !== imageRoutingRequestRef.current) return;
      setImageRoutingResult(decision);
      setImageRoutedText(text);
      setImageRoutingState("success");

      if (decision.intent === "IMAGE_DATASET_ANALYSIS") {
        if (!imageZipFile) {
          setImageAnalysisError("Select an image dataset ZIP before analyzing it.");
          return;
        }

        setImageAnalysisLoading(true);
        try {
          const analysis = await analyzeImageDataset(imageZipFile);
          if (requestId !== imageRoutingRequestRef.current) return;
          setImageAnalysisResult(analysis);
        } catch (error) {
          if (requestId !== imageRoutingRequestRef.current) return;
          setImageAnalysisError(
            error.message || "Image dataset analysis failed. Please try again.",
          );
        } finally {
          if (requestId === imageRoutingRequestRef.current) {
            setImageAnalysisLoading(false);
          }
        }
      }
    } catch (error) {
      if (requestId !== imageRoutingRequestRef.current) return;
      setImageRoutingError(error.message || "Image request routing failed. Please try again.");
      setImageRoutingState("error");
    }
  }

  async function startImageTraining() {
    const currentRoute =
      imageRoutingState === "success" && imageRoutedText === task.trim();
    const explicitTrainingReady =
      imageRoutingResult?.intent === "IMAGE_TRAINING_REQUEST" &&
      imageRoutingResult.ready;
    const bothImageInputsSelected = Boolean(imageZipFile && imagePredictionFile);
    if (
      datasetType !== "image" ||
      imageTrainingState === "loading" ||
      !currentRoute ||
      (!explicitTrainingReady && !bothImageInputsSelected) ||
      !imageZipFile
    ) {
      return;
    }

    setImageTrainingState("loading");
    setImageTrainingError("");
    setImageTrainingResult(null);
    setImagePredictionState("idle");
    setImagePredictionError("");
    setImagePredictionResult(null);
    const requestId = ++imageTrainingRequestRef.current;

    try {
      const payload = await trainImageModel(imageZipFile);
      if (requestId !== imageTrainingRequestRef.current) return;
      if (typeof payload.model_id !== "string" || !payload.model_id.trim()) {
        throw new Error("Training completed without returning a model ID.");
      }
      setImageTrainingResult(payload);
      setImageModelId(payload.model_id.trim());
      setImageModelSource("trained");
      setImageTrainingState("success");
    } catch (error) {
      if (requestId !== imageTrainingRequestRef.current) return;
      setImageTrainingState("error");
      setImageTrainingError(error.message || "Image training failed. Please try again.");
    }
  }

  async function handleImagePredict() {
    if (datasetType !== "image" || imagePredictionInFlightRef.current !== null) return;
    if (!imageModelId.trim()) {
      setImagePredictionState("error");
      setImagePredictionError("Enter a trained image model ID before predicting.");
      return;
    }
    if (!imagePredictionFile) {
      setImagePredictionState("error");
      setImagePredictionError("Choose a single image before predicting.");
      return;
    }

    const requestId = ++imagePredictionRequestRef.current;
    imagePredictionInFlightRef.current = requestId;
    setImagePredictionState("loading");
    setImagePredictionError("");
    setImagePredictionResult(null);

    try {
      const payload = await predictImageModel(imageModelId.trim(), imagePredictionFile);
      if (requestId !== imagePredictionRequestRef.current) return;
      setImagePredictionResult(payload);
      setImagePredictionState("success");
    } catch (error) {
      if (requestId !== imagePredictionRequestRef.current) return;
      setImagePredictionState("error");
      setImagePredictionError(error.message || "Image prediction failed. Please try again.");
    } finally {
      if (imagePredictionInFlightRef.current === requestId) {
        imagePredictionInFlightRef.current = null;
      }
    }
  }

  function useExample() {
    clearTabularWorkflow();
    setTask("Predict customer churn");
  }

  async function handleAnalyze() {
    if (datasetType !== "tabular") return;
    if (!fileName || uploadState !== "success" || !task.trim()) {
      setAnalyzeState("error");
      setAnalyzeError(
        uploadState !== "success"
          ? "Choose and validate a dataset before analyzing."
          : "Choose a dataset and describe the task before analyzing.",
      );
      setAnalyzeResult(null);
      setTrainingConfirmed(false);
      setTrainingState("idle");
      setTrainingError("");
      setTrainingResult(null);
      return;
    }

    const requestId = ++tabularWorkflowRequestRef.current;
    setAnalyzeState("loading");
    setAnalyzeError("");
    setAnalyzeResult(null);
    setTrainingConfirmed(false);
    setGroupColumn("");
    setExcludedFeatureColumns([]);
    setTrainingState("idle");
    setTrainingError("");
    setTrainingResult(null);
    setPredictionState("idle");
    setPredictionError("");
    setPredictionMode("sample");
    setPredictionInputs({});
    setPredictionResult(null);

    try {
      const response = await fetch("/nlp/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filename: fileName, text: task.trim() }),
      });
      const payload = await response.json();
      if (requestId !== tabularWorkflowRequestRef.current) return;

      if (!response.ok) {
        throw new Error(payload.detail || "Analysis failed.");
      }

      setTrainingConfirmed(false);
      setTrainingState("idle");
      setTrainingError("");
      setTrainingResult(null);
      setAnalyzeResult(payload);
      setAnalyzeState("success");
    } catch (error) {
      if (requestId !== tabularWorkflowRequestRef.current) return;
      setAnalyzeState("error");
      setAnalyzeError(error.message || "Analysis failed. Please try again.");
    }
  }

  async function confirmTraining() {
    if (
      datasetType !== "tabular" ||
      uploadState !== "success" ||
      analyzeState !== "success" ||
      analyzeResult?.ready_for_training !== true
    ) {
      return;
    }

    const requestId = ++tabularWorkflowRequestRef.current;
    setTrainingState("loading");
    setTrainingError("");
    setTrainingResult(null);
    setPredictionState("idle");
    setPredictionError("");
    setPredictionResult(null);

    try {
      const response = await fetch("/nlp/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          buildNlpTrainingRequest({
            filename: fileName,
            text: task.trim(),
            groupColumn,
            featureColumns: selectedFeatureColumns,
            tuningStrategy,
            tuningTrials: Number(tuningTrials) || 8,
          }),
        ),
      });
      const payload = await response.json();
      if (requestId !== tabularWorkflowRequestRef.current) return;

      if (!response.ok) {
        throw new Error(payload.detail || "Training failed.");
      }

      const outcome = getNlpTrainingOutcome(payload);
      if (!outcome.successful) {
        setTrainingResult(outcome.result);
        setTrainingState("error");
        setTrainingError(outcome.message);
        setTrainingConfirmed(false);
        return;
      }

      setTrainingResult(outcome.result);
      setTrainingState("success");
      setTrainingConfirmed(true);
      setSavedModelIdInput(
        outcome.result?.automl_training?.model?.model_id ||
          outcome.result?.report?.model_id ||
          "",
      );
      setPredictionState("idle");
      setPredictionError("");
      setPredictionMode("sample");
      setPredictionResult(null);
      const rawFeatureNames = getPredictionConfig(payload).featureNames;
      setPredictionInputs(
        Object.fromEntries(rawFeatureNames.map((column) => [column, ""]))
      );
    } catch (error) {
      if (requestId !== tabularWorkflowRequestRef.current) return;
      setTrainingState("error");
      setTrainingError(error.message || "Training failed. Please try again.");
    }
  }

  function getPredictionFeatures() {
    return predictionConfig.featureNames;
  }

  function getFeatureType(column) {
    const preprocessing = trainingResult?.automl_training?.preprocessing || {};
    if (preprocessing.imputed_columns?.numeric?.includes(column)) return "number";
    if (preprocessing.imputed_columns?.categorical?.includes(column)) return "object";
    return uploadResult?.analysis?.data_types?.[column] || "object";
  }

  function updatePredictionInput(column, value) {
    setPredictionInputs((current) => ({ ...current, [column]: value }));
  }

  function buildPredictionRecord() {
    const featureTypes = Object.fromEntries(
      getPredictionFeatures().map((column) => [column, getFeatureType(column)]),
    );
    return createPredictionRecord({
      trainingResult,
      predictionMode,
      predictionInputs,
      sampleRow: uploadResult?.sample_rows?.[0] || {},
      featureTypes,
    });
  }

  async function handlePredict() {
    if (
      datasetType !== "tabular" ||
      uploadState !== "success" ||
      trainingState !== "success" ||
      !trainingResult
    ) {
      return;
    }
    const modelId = trainingResult?.automl_training?.model?.model_id || trainingResult?.report?.model_id;
    if (!modelId) {
      setPredictionState("error");
      setPredictionError("No saved model ID was returned by training.");
      return;
    }

    const requestId = ++tabularWorkflowRequestRef.current;
    setPredictionState("loading");
    setPredictionError("");
    setPredictionResult(null);

    try {
      const record = buildPredictionRecord();
      const response = await fetch("/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, data: [record] }),
      });
      const payload = await response.json();
      if (requestId !== tabularWorkflowRequestRef.current) return;

      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed.");
      }

      setPredictionResult(payload);
      setPredictionState("success");
    } catch (error) {
      if (requestId !== tabularWorkflowRequestRef.current) return;
      setPredictionState("error");
      setPredictionError(error.message || "Prediction failed. Please try again.");
    }
  }

  async function handleSavedModelFilePredict() {
    const modelId = savedModelIdInput.trim();
    if (!modelId) {
      setSavedModelPredictionState("error");
      setSavedModelPredictionError("Enter a saved tabular model ID.");
      setSavedModelPredictionResult(null);
      return;
    }
    if (!savedModelFile) {
      setSavedModelPredictionState("error");
      setSavedModelPredictionError("Choose a CSV or Excel file to test.");
      setSavedModelPredictionResult(null);
      return;
    }

    const requestId = ++savedPredictionRequestRef.current;
    setSavedModelPredictionState("loading");
    setSavedModelPredictionError("");
    setSavedModelPredictionResult(null);

    try {
      const formData = new FormData();
      formData.append("model_id", modelId);
      formData.append("file", savedModelFile);
      const response = await fetch("/predict/file", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (requestId !== savedPredictionRequestRef.current) return;

      if (!response.ok) {
        const detail = payload.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : detail?.message || "Saved-model prediction failed.",
        );
      }

      setSavedModelPredictionResult(payload);
      setSavedModelPredictionState("success");
    } catch (error) {
      if (requestId !== savedPredictionRequestRef.current) return;
      setSavedModelPredictionState("error");
      setSavedModelPredictionError(
        error.message || "Saved-model prediction failed. Please try again.",
      );
    }
  }

  function downloadSavedModelPredictions() {
    if (!savedModelPredictionResult) return;
    const csv = buildPredictionCsv(savedModelPredictionResult);
    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `${savedModelPredictionResult.model_id || "model"}_predictions.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div style={pageStyle}>
      <header
        style={{
          borderBottom: "1px solid #dbe5de",
          background: "#fbfcfa",
        }}
      >
        <div
          style={{
            ...shellStyle,
            minHeight: "72px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "24px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div
              aria-hidden="true"
              style={{
                width: "30px",
                height: "30px",
                borderRadius: "8px",
                background: "#1e6041",
                color: "#ffffff",
                display: "grid",
                placeItems: "center",
                fontFamily: "'Courier New', monospace",
                fontWeight: 700,
                fontSize: "14px",
              }}
            >
              T
            </div>
            <strong style={{ fontSize: "18px", letterSpacing: "0.01em" }}>
              TextToAutoML
            </strong>
          </div>
          <span
            style={{
              color: "#66756b",
              fontFamily: "'Courier New', monospace",
              fontSize: "11px",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Natural language model studio
          </span>
        </div>
      </header>

      <main style={{ ...shellStyle, padding: "72px 0 96px" }}>
        <section style={{ maxWidth: "720px", marginBottom: "42px" }}>
          <p
            style={{
              margin: "0 0 14px",
              color: "#1e6041",
              fontFamily: "'Courier New', monospace",
              fontSize: "12px",
              fontWeight: 700,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
            }}
          >
            Start a new experiment
          </p>
          <h1
            style={{
              margin: "0 0 16px",
              maxWidth: "680px",
              fontSize: "clamp(38px, 6vw, 68px)",
              fontWeight: 500,
              lineHeight: 1.02,
              letterSpacing: "-0.03em",
            }}
          >
            Turn a question into a model.
          </h1>
          <p
            style={{
              margin: 0,
              maxWidth: "600px",
              color: "#66756b",
              fontFamily: "Arial, sans-serif",
              fontSize: "17px",
              lineHeight: 1.6,
            }}
          >
            Bring your dataset, describe what you want to understand, and let
            TextToAutoML shape the next step.
          </p>
        </section>

        <section aria-label="Dataset type selection" style={{ marginBottom: "32px" }}>
          <h2 style={{ margin: "0 0 14px", fontSize: "22px", fontWeight: 500 }}>
            What type of dataset do you have?
          </h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "12px",
            }}
          >
            {[
              { value: "tabular", title: "Tabular Dataset", detail: "CSV / XLSX files" },
              { value: "image", title: "Image Dataset", detail: "ZIP containing folders of images" },
            ].map((option) => (
              <button
                key={option.value}
                type="button"
                aria-pressed={datasetType === option.value}
                onClick={() => selectDatasetType(option.value)}
                style={{
                  padding: "16px 18px",
                  border: `1px solid ${datasetType === option.value ? "#1e6041" : "#dbe5de"}`,
                  borderRadius: "8px",
                  background: datasetType === option.value ? "#e3f0e6" : "#ffffff",
                  color: "#17221d",
                  textAlign: "left",
                  cursor: "pointer",
                  boxShadow: datasetType === option.value ? "inset 0 0 0 1px #1e6041" : "none",
                }}
              >
                <strong style={{ display: "block", color: "#1e6041", font: "600 15px Arial, sans-serif" }}>
                  {option.title}
                </strong>
                <span style={{ display: "block", marginTop: "5px", color: "#66756b", font: "13px Arial, sans-serif" }}>
                  {option.detail}
                </span>
              </button>
            ))}
          </div>
        </section>

        {datasetType === "tabular" ? (
          <div data-workflow="tabular">
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "20px",
            alignItems: "stretch",
          }}
        >
          <div style={{ ...panelStyle, padding: "28px" }}>
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                justifyContent: "space-between",
                gap: "16px",
                marginBottom: "20px",
              }}
            >
              <div>
                <p
                  style={{
                    margin: "0 0 8px",
                    color: "#8a988e",
                    fontFamily: "'Courier New', monospace",
                    fontSize: "11px",
                    letterSpacing: "0.1em",
                    textTransform: "uppercase",
                  }}
                >
                  Step 01
                </p>
                <h2 style={{ margin: 0, fontSize: "26px", fontWeight: 500 }}>
                  Add your dataset
                </h2>
              </div>
              <span style={{ color: "#8a988e", fontSize: "13px" }}>
                CSV / XLSX
              </span>
            </div>

            <label
              htmlFor="dataset-upload"
              style={{
                minHeight: "190px",
                padding: "20px",
                border: "1px dashed #aebfb3",
                borderRadius: "8px",
                background: "#f8fbf8",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                cursor: "pointer",
                fontFamily: "Arial, sans-serif",
              }}
            >
              <span
                aria-hidden="true"
                style={{
                  width: "42px",
                  height: "42px",
                  marginBottom: "14px",
                  borderRadius: "50%",
                  background: "#e3f0e6",
                  color: "#1e6041",
                  display: "grid",
                  placeItems: "center",
                  fontSize: "22px",
                }}
              >
                +
              </span>
              <strong style={{ color: "#294534", fontSize: "15px" }}>
                {fileName || "Choose a dataset"}
              </strong>
              <span style={{ marginTop: "7px", color: "#7b8a80", fontSize: "13px" }}>
                {uploadState === "loading"
                  ? "Uploading dataset..."
                  : fileName
                    ? "Ready for the next step"
                    : "or drop it here"}
              </span>
              <input
                id="dataset-upload"
                type="file"
                accept=".csv,.xlsx"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
            </label>

            <div
              aria-live="polite"
              style={{
                minHeight: "24px",
                marginTop: "18px",
                color: uploadState === "error" ? "#a13d35" : "#526158",
                font: "13px/1.5 Arial, sans-serif",
              }}
            >
              {uploadState === "error" && <p style={{ margin: 0 }}>{uploadError}</p>}
              {uploadState === "success" && uploadResult && (
                <div
                  style={{
                    paddingTop: "16px",
                    borderTop: "1px solid #edf2ee",
                  }}
                >
                  <strong style={{ color: "#1e6041" }}>Dataset ready</strong>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                      gap: "8px 16px",
                      marginTop: "10px",
                    }}
                  >
                    <span>
                      Rows: {uploadResult.analysis?.dataset_info?.rows ?? "-"}
                    </span>
                    <span>
                      Columns: {uploadResult.analysis?.dataset_info?.columns ?? "-"}
                    </span>
                    <span>
                      Missing values: {uploadResult.analysis?.quality?.missing_values ?? "-"}
                    </span>
                    <span>
                      Recommended target: {uploadResult.automl_recommendation?.recommended_target || uploadResult.dataset_intelligence?.recommended_target || "None"}
                    </span>
                  </div>
                  <p style={{ margin: "10px 0 0" }}>
                    Suggested task: {uploadResult.automl_recommendation?.problem_type || "Review the dataset"}
                  </p>
                </div>
              )}
            </div>
          </div>

          <div style={{ ...panelStyle, padding: "28px" }}>
            <div style={{ marginBottom: "20px" }}>
              <p
                style={{
                  margin: "0 0 8px",
                  color: "#8a988e",
                  fontFamily: "'Courier New', monospace",
                  fontSize: "11px",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}
              >
                Step 02
              </p>
              <h2 style={{ margin: 0, fontSize: "26px", fontWeight: 500 }}>
                Describe the task
              </h2>
            </div>

            <label
              htmlFor="task-input"
              style={{
                display: "block",
                marginBottom: "10px",
                color: "#526158",
                fontFamily: "Arial, sans-serif",
                fontSize: "14px",
              }}
            >
              What should the model predict?
            </label>
            <textarea
              id="task-input"
              value={task}
              onChange={handleTabularTaskChange}
              placeholder="e.g. Predict customer churn"
              rows={5}
              style={{
                width: "100%",
                boxSizing: "border-box",
                resize: "vertical",
                padding: "14px",
                border: "1px solid #cbd8ce",
                borderRadius: "6px",
                background: "#fbfcfa",
                color: "#17221d",
                font: "16px/1.5 Arial, sans-serif",
                outlineColor: "#1e6041",
              }}
            />
            <button
              type="button"
              onClick={useExample}
              style={{
                marginTop: "12px",
                padding: 0,
                border: 0,
                background: "transparent",
                color: "#1e6041",
                cursor: "pointer",
                font: "13px Arial, sans-serif",
                textDecoration: "underline",
                textUnderlineOffset: "3px",
              }}
            >
              Use example: Predict customer churn
            </button>
          </div>
        </section>

        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            margin: "24px 0 56px",
          }}
        >
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={analyzeState === "loading" || uploadState !== "success"}
            style={{
              padding: "14px 24px",
              border: 0,
              borderRadius: "6px",
              background: analyzeState === "loading" ? "#7f9a89" : "#1e6041",
              color: "#ffffff",
              cursor: analyzeState === "loading" ? "wait" : "pointer",
              font: "600 15px Arial, sans-serif",
              boxShadow: "0 8px 18px rgba(30, 96, 65, 0.2)",
            }}
          >
            {analyzeState === "loading" ? "Analyzing..." : "Analyze task"}
            <span aria-hidden="true" style={{ marginLeft: "10px" }}>
              -&gt;
            </span>
          </button>
        </div>

        <section
          aria-label="Analysis results"
          style={{
            ...panelStyle,
            minHeight: "230px",
            padding: "28px",
            borderStyle: "dashed",
            boxShadow: "none",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "16px",
              marginBottom: "18px",
            }}
          >
            <h2 style={{ margin: 0, fontSize: "22px", fontWeight: 500 }}>
              Understanding Your Task
            </h2>
            <span
              style={{
                color: "#8a988e",
                fontFamily: "'Courier New', monospace",
                fontSize: "11px",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
              }}
            >
              {analyzeResult?.needs_clarification || analyzeResult?.dataset_resolution?.needs_clarification
                ? "Needs clarification"
                : analyzeResult
                  ? "Analysis complete"
                  : "Awaiting analysis"}
            </span>
          </div>
          <div
            style={{
              minHeight: "150px",
              borderTop: "1px solid #edf2ee",
            }}
          >
            {analyzeState === "error" && (
              <p
                style={{
                  margin: "18px 0 0",
                  color: "#a13d35",
                  font: "14px/1.5 Arial, sans-serif",
                }}
              >
                {analyzeError}
              </p>
            )}
            {analyzeState === "success" && analyzeResult?.needs_clarification ? (
              <div
                style={{
                  paddingTop: "18px",
                  borderTop: "1px solid #edf2ee",
                  font: "14px/1.5 Arial, sans-serif",
                }}
              >
                <strong style={{ color: "#8c5a16" }}>Needs clarification</strong>
                <p style={{ margin: "10px 0 0", color: "#526158" }}>
                  {analyzeResult.intent_analysis?.clarification_reason ||
                    "Please clarify what you would like to do with this dataset."}
                </p>
                {uploadResult && (
                  <section aria-label="Dataset snapshot pending clarification" style={{ marginTop: "18px", paddingTop: "16px", borderTop: "1px solid #edf2ee" }}>
                    <h3 style={{ margin: "0 0 10px", fontSize: "17px", fontWeight: 600 }}>Understanding Your Dataset</h3>
                    <div style={{ display: "flex", gap: "14px 22px", flexWrap: "wrap", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                      <span>Rows: {uploadResult.analysis?.dataset_info?.rows ?? "Not available"}</span>
                      <span>Columns: {uploadResult.analysis?.dataset_info?.columns ?? "Not available"}</span>
                      <span>Total missing values: {uploadResult.analysis?.quality?.missing_values ?? "Not available"}</span>
                      {uploadResult.analysis?.quality?.duplicate_rows != null && <span>Duplicate rows: {uploadResult.analysis.quality.duplicate_rows}</span>}
                      <span>Target column: Not resolved yet</span>
                    </div>
                    <details style={{ marginTop: "12px" }}>
                      <summary style={{ cursor: "pointer", color: "#1e6041", fontWeight: 600 }}>Column Details ({datasetColumns.length})</summary>
                      <p style={{ margin: "8px 0", color: "#66756b", font: "12px/1.5 Arial, sans-serif" }}>
                        Column roles are not assigned until the task and target are resolved. No semantic meaning is inferred from column names.
                      </p>
                      <ul style={{ margin: 0, paddingLeft: "20px", color: "#526158", font: "13px/1.6 Arial, sans-serif" }}>
                        {datasetColumns.map((column) => (
                          <li key={column}>{column} — {formatColumnType(uploadResult.analysis?.data_types?.[column])}</li>
                        ))}
                      </ul>
                    </details>
                  </section>
                )}
              </div>
            ) : analyzeState === "success" && analyzeResult ? (
              <div style={{ font: "14px/1.5 Arial, sans-serif" }}>
                <div style={{ paddingTop: "18px", borderTop: "1px solid #edf2ee" }}>
                  <p style={{ margin: "0 0 14px", color: "#26372d", fontSize: "16px" }}>
                    {analyzeResult.dataset_resolution?.problem_type
                      ? <>TextToAutoML understood your request as a <strong>{formatProblemType(analyzeResult.dataset_resolution.problem_type)}</strong> problem.</>
                      : "TextToAutoML analyzed the task you described."}
                  </p>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                      gap: "14px 20px",
                    }}
                  >
                    <div>
                      <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>What you asked</span>
                      <strong>{task.trim() || analyzeResult.user_text || "Not available"}</strong>
                    </div>
                    <div>
                      <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>Problem type</span>
                      <strong>{analyzeResult.dataset_resolution?.problem_type || "Not resolved"}</strong>
                    </div>
                    <div>
                      <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>Target</span>
                      <strong>
                        {String(analyzeResult.dataset_resolution?.problem_type || "").toLowerCase().includes("cluster")
                          ? "No target is used for clustering"
                          : resolvedTargetColumn || "Not resolved"}
                      </strong>
                      {resolvedTargetColumn && !String(analyzeResult.dataset_resolution?.problem_type || "").toLowerCase().includes("cluster") && (
                        <span style={{ display: "block", marginTop: "3px", color: "#66756b", fontSize: "12px" }}>
                          The model will learn to predict this column from the selected input features.
                        </span>
                      )}
                    </div>
                    <div>
                      <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>Dataset size</span>
                      <strong>
                        {uploadResult?.analysis?.dataset_info?.rows != null
                          ? `${uploadResult.analysis.dataset_info.rows} rows`
                          : "Rows not available"}
                        {" × "}
                        {uploadResult?.analysis?.dataset_info?.columns != null
                          ? `${uploadResult.analysis.dataset_info.columns} columns`
                          : "columns not available"}
                      </strong>
                    </div>
                    {analyzeResult.intent?.confidence != null && (
                      <div>
                        <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>Task interpretation confidence</span>
                        <strong>{Math.round(Number(analyzeResult.intent.confidence) * 100)}%</strong>
                      </div>
                    )}
                    {analyzeResult.dataset_resolution?.confidence != null && (
                      <div>
                        <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>Task and target resolution confidence</span>
                        <strong>{Math.round(Number(analyzeResult.dataset_resolution.confidence) * 100)}%</strong>
                      </div>
                    )}
                  </div>
                  {resolvedTargetColumn && !String(analyzeResult.dataset_resolution?.problem_type || "").toLowerCase().includes("cluster") && analyzeResult.dataset_resolution?.reason && (
                    <p style={{ margin: "14px 0 0", color: "#526158" }}>
                      <strong>Why this target was selected:</strong> {analyzeResult.dataset_resolution.reason}
                    </p>
                  )}
                  {analyzeResult.training_time_estimate && (
                    <p style={{ margin: "12px 0 0", color: "#66756b", fontSize: "13px" }}>
                      Estimated training time: {formatDuration(analyzeResult.training_time_estimate.min_seconds)}
                      {" - "}{formatDuration(analyzeResult.training_time_estimate.max_seconds)}
                    </p>
                  )}
                </div>

                {uploadResult && (
                  <section
                    aria-label="Understanding Your Dataset"
                    style={{ marginTop: "24px", paddingTop: "20px", borderTop: "1px solid #edf2ee" }}
                  >
                    <h3 style={{ margin: "0 0 14px", fontSize: "19px", fontWeight: 600 }}>
                      Understanding Your Dataset
                    </h3>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px 18px", font: "13px/1.5 Arial, sans-serif" }}>
                      <div><span>Rows</span><strong style={{ display: "block" }}>{uploadResult.analysis?.dataset_info?.rows ?? "Not available"}</strong></div>
                      <div><span>Columns</span><strong style={{ display: "block" }}>{uploadResult.analysis?.dataset_info?.columns ?? "Not available"}</strong></div>
                      <div><span>Total missing values</span><strong style={{ display: "block" }}>{uploadResult.analysis?.quality?.missing_values ?? "Not available"}</strong></div>
                      {uploadResult.analysis?.quality?.duplicate_rows != null && (
                        <div><span>Duplicate rows</span><strong style={{ display: "block" }}>{uploadResult.analysis.quality.duplicate_rows}</strong></div>
                      )}
                      <div><span>Target column</span><strong style={{ display: "block" }}>{trainingTargetColumn || (isClusteringTask ? "Not used (clustering)" : "Not resolved")}</strong></div>
                      <div><span>{trainingFeatureNames ? "Input features used" : "Input features selected"}</span><strong style={{ display: "block" }}>{displayedFeatureNames.length}</strong></div>
                    </div>

                    <details style={{ marginTop: "16px" }}>
                      <summary style={{ cursor: "pointer", color: "#1e6041", fontWeight: 600 }}>
                        Column Details ({datasetColumns.length})
                      </summary>
                      <p style={{ margin: "8px 0", color: "#66756b", font: "12px/1.5 Arial, sans-serif" }}>
                        Names and types come from the uploaded dataset. No semantic meaning is inferred from a column name.
                      </p>
                      <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", font: "13px/1.5 Arial, sans-serif" }}>
                          <thead>
                            <tr>
                              <th style={{ padding: "8px", borderBottom: "1px solid #dbe5de" }}>Column</th>
                              <th style={{ padding: "8px", borderBottom: "1px solid #dbe5de" }}>Type</th>
                              <th style={{ padding: "8px", borderBottom: "1px solid #dbe5de" }}>Role</th>
                            </tr>
                          </thead>
                          <tbody>
                            {datasetColumns.map((column) => {
                              const dataType = uploadResult.analysis?.data_types?.[column];
                              const removedReason = getRemovedColumnReason(
                                trainingPreprocessing.removed_columns,
                                column,
                              );
                              let role = "Not selected for training";
                              if (column === trainingTargetColumn) {
                                role = "Target column — value the model is learning to predict";
                              } else if (column === groupColumn) {
                                role = "Group column — kept out of model inputs";
                              } else if (removedReason) {
                                role = `Not used for training — ${removedReason}`;
                              } else if (!selectedFeatureColumns.includes(column)) {
                                role = "Not selected for training";
                              } else if (trainingFeatureNames) {
                                role = trainingFeatureNames.includes(column)
                                  ? "Input feature"
                                  : "Not used for training";
                              } else {
                                role = "Input feature";
                              }
                              return (
                                <tr key={column}>
                                  <th scope="row" style={{ padding: "8px", borderBottom: "1px solid #edf2ee", fontWeight: 600 }}>{column}</th>
                                  <td style={{ padding: "8px", borderBottom: "1px solid #edf2ee" }}>{formatColumnType(dataType)}</td>
                                  <td style={{ padding: "8px", borderBottom: "1px solid #edf2ee" }}>{role}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      {isClassificationTask && Array.isArray(trainingPreprocessing.target_classes) && trainingPreprocessing.target_classes.length > 0 && (
                        <p style={{ margin: "10px 0 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                          Target classes: {trainingPreprocessing.target_classes.join(", ")}
                        </p>
                      )}
                    </details>
                  </section>
                )}

                {analyzeResult.ready_for_training && (
                  <section
                    aria-label="How Your Model Will Be Trained"
                    style={{ marginTop: "24px", paddingTop: "20px", borderTop: "1px solid #edf2ee" }}
                  >
                    <h3 style={{ margin: "0 0 12px", fontSize: "19px", fontWeight: 600 }}>
                      How Your Model Will Be Trained
                    </h3>
                    {String(analyzeResult.dataset_resolution?.problem_type || "").toLowerCase().includes("cluster") ? (
                      <>
                        <p style={{ margin: "0 0 10px", color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>
                          Clustering is unsupervised, so this workflow does not use a target column or create separate validation and test partitions.
                        </p>
                        <ol style={{ margin: 0, paddingLeft: "22px", color: "#526158", font: "14px/1.7 Arial, sans-serif" }}>
                          <li>Prepare the selected input features.</li>
                          <li>Train KMeans candidates using the prepared data.</li>
                          <li>Compare candidates using silhouette score and select the highest-scoring result.</li>
                          <li>Save the selected model for prediction.</li>
                        </ol>
                      </>
                    ) : (
                      <>
                        <ol style={{ margin: 0, paddingLeft: "22px", color: "#526158", font: "14px/1.7 Arial, sans-serif" }}>
                          <li>Prepare the dataset and remove duplicate rows.</li>
                          <li>Separate the resolved target from the input features.</li>
                          <li>Prepare numeric and categorical features, including missing-value handling and encoding.</li>
                          <li>Split rows into training, validation, and test partitions.</li>
                          <li>Train suitable candidate models for the detected task.</li>
                          <li>Compare candidates using the validation partition.</li>
                          <li>Select the best candidate using the task-appropriate metric.</li>
                          <li>Refit the selected model using the training and validation data.</li>
                          <li>Evaluate the refitted model on held-out test data.</li>
                          <li>Save the trained model for prediction.</li>
                        </ol>
                        {trainingState === "success" && (
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "10px 16px", marginTop: "14px", color: "#26372d", font: "13px Arial, sans-serif" }}>
                            <div>Training data: <strong>{trainingData.training_rows ?? "Not returned"} rows</strong></div>
                            <div>Validation data: <strong>{trainingData.validation_rows ?? "Not returned"} rows</strong></div>
                            <div>Test data: <strong>{trainingData.test_rows ?? "Not returned"} rows</strong></div>
                          </div>
                        )}
                        <p style={{ margin: "10px 0 0", color: "#66756b", font: "12px/1.5 Arial, sans-serif" }}>
                          Training data teaches the models. Validation data is used to compare candidates. Test data is kept separate for final evaluation.
                        </p>
                      </>
                    )}
                  </section>
                )}
                <div style={{ marginTop: "22px" }}>
                  <p style={{ margin: "0 0 10px", color: "#526158", font: "14px Arial, sans-serif" }}>
                    <strong>{displayedFeatureNames.length} input features selected</strong>
                  </p>
                  <details aria-label="Advanced Settings">
                    <summary style={{ cursor: "pointer", color: "#1e6041", fontWeight: 600 }}>
                      Advanced Settings
                    </summary>
                    <div style={{ marginTop: "14px" }}>
                  <div style={{ marginBottom: "16px", maxWidth: "420px" }}>
                    <label
                      htmlFor="group-column"
                      style={{
                        display: "block",
                        marginBottom: "6px",
                        color: "#26372d",
                        fontWeight: 600,
                      }}
                    >
                      Group Column
                    </label>
                    <select
                      id="group-column"
                      aria-label="Group Column"
                      value={groupColumn}
                      onChange={(event) => setGroupColumn(event.target.value)}
                      disabled={trainingState === "loading" || trainingConfirmed}
                      style={{
                        width: "100%",
                        minHeight: "40px",
                        padding: "8px 10px",
                        border: "1px solid #cbd8ce",
                        borderRadius: "4px",
                        background: "#ffffff",
                        color: "#17221d",
                        font: "14px Arial, sans-serif",
                      }}
                    >
                      <option value="">None</option>
                      {groupColumnOptions.map((column) => (
                        <option key={column} value={column}>
                          {column}
                        </option>
                      ))}
                    </select>
                    <p style={{ margin: "6px 0 0", color: "#617066", fontSize: "12px", lineHeight: 1.5 }}>
                      Use a group column when multiple rows belong to the same person, customer, patient, device, or other entity. Records from the same group are kept together when the dataset is split.
                    </p>
                    <p style={{ margin: "6px 0 0", color: "#617066", fontSize: "12px", lineHeight: 1.5 }}>
                      {groupColumn
                        ? "A group column is selected for this run."
                        : "No group column selected. Standard dataset splitting will be used."}
                    </p>
                    <p style={{ margin: "6px 0 0", color: "#617066", fontSize: "12px", lineHeight: 1.5 }}>
                      Group-aware splitting keeps records from the same entity together across train, validation, and test sets.
                    </p>
                  </div>
                  <fieldset
                    aria-label="Hyperparameter tuning"
                    style={{
                      maxWidth: "640px",
                      margin: "0 0 16px",
                      padding: "10px 12px",
                      border: "1px solid #cbd8ce",
                      borderRadius: "4px",
                    }}
                    disabled={trainingState === "loading" || trainingConfirmed}
                  >
                    <legend style={{ padding: "0 4px", color: "#26372d", fontWeight: 600 }}>
                      Hyperparameter tuning
                    </legend>
                    <p style={{ margin: "0 0 6px", color: "#26372d", fontWeight: 600 }}>
                      {tuningStrategyLabels[tuningStrategy] || tuningStrategy}
                    </p>
                    <p style={{ margin: "0 0 10px", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                      Hyperparameter tuning allows TextToAutoML to try different model settings to find a better configuration.
                    </p>
                    {tuningStrategy === "auto" && (
                      <p style={{ margin: "0 0 10px", color: "#526158" }}>
                        TextToAutoML selects a suitable tuning strategy and trial budget based on your dataset and training requirements.
                      </p>
                    )}
                      <label style={{ display: "block", marginTop: "8px", color: "#26372d" }}>
                        Manual strategy override
                        <select
                          aria-label="Manual tuning strategy"
                          value={manualTuningStrategy}
                          onChange={(event) => {
                            const strategy = event.target.value;
                            setManualTuningStrategy(strategy);
                            setTuningStrategy(strategy);
                          }}
                          style={{
                            display: "block",
                            width: "100%",
                            minHeight: "38px",
                            marginTop: "4px",
                            padding: "7px 9px",
                            border: "1px solid #cbd8ce",
                            borderRadius: "4px",
                            background: "#ffffff",
                          }}
                        >
                          <option value="" disabled>Select a manual strategy</option>
                          <option value="grid">Grid Search</option>
                          <option value="bayesian">Bayesian Optimization (Optuna TPE)</option>
                        </select>
                      </label>
                      {tuningStrategy === "bayesian" && manualTuningStrategy === "bayesian" && (
                        <label style={{ display: "block", marginTop: "8px", color: "#26372d" }}>
                          Trials per model (1-30)
                          <input
                            aria-label="Bayesian trials per model"
                            type="number"
                            min="1"
                            max="30"
                            step="1"
                            value={tuningTrials}
                            onChange={(event) => {
                              const value = event.target.value;
                              if (value === "") {
                                setTuningTrials("");
                                return;
                              }
                              const parsed = Number(value);
                              if (Number.isFinite(parsed)) {
                                setTuningTrials(String(Math.max(1, Math.min(30, Math.trunc(parsed)))));
                              }
                            }}
                            style={{
                              display: "block",
                              width: "120px",
                              minHeight: "36px",
                              marginTop: "4px",
                              padding: "6px 8px",
                              border: "1px solid #cbd8ce",
                              borderRadius: "4px",
                            }}
                          />
                        </label>
                      )}
                      {tuningStrategy !== "auto" && (
                        <button
                          type="button"
                          onClick={() => setTuningStrategy("auto")}
                          style={{
                            marginTop: "10px",
                            padding: "6px 10px",
                            border: "1px solid #cbd8ce",
                            borderRadius: "4px",
                            background: "#ffffff",
                            color: "#26372d",
                            cursor: "pointer",
                          }}
                        >
                          Use automatic selection
                        </button>
                      )}
                  </fieldset>
                  <details style={{ maxWidth: "640px", marginBottom: "16px" }}>
                    <summary style={{ cursor: "pointer", color: "#1e6041", fontWeight: 600 }}>
                      Choose input features
                    </summary>
                    <fieldset
                      aria-label="Features included in the model"
                      style={{
                        maxHeight: "220px",
                        overflowY: "auto",
                        margin: "10px 0 0",
                        padding: "10px 12px",
                        border: "1px solid #cbd8ce",
                        borderRadius: "4px",
                      }}
                      disabled={trainingState === "loading" || trainingConfirmed}
                    >
                      <legend style={{ padding: "0 4px", color: "#26372d", fontWeight: 600 }}>
                        Features included in the model
                      </legend>
                    {featureColumnOptions.map((column) => (
                      <label
                        key={column}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          padding: "4px 0",
                          color: "#26372d",
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={!excludedFeatureColumns.includes(column)}
                          onChange={(event) => {
                            setExcludedFeatureColumns((current) => (
                              event.target.checked
                                ? current.filter((excluded) => excluded !== column)
                                : [...current, column]
                            ));
                          }}
                        />
                        <span>{column}</span>
                      </label>
                    ))}
                    {featureColumnOptions.length === 0 && (
                      <p style={{ margin: 0, color: "#617066" }}>
                        No eligible feature columns are available.
                      </p>
                    )}
                    {featureColumnOptions.length > 0 && selectedFeatureColumns.length === 0 && (
                      <p role="alert" style={{ margin: "6px 0 0", color: "#a13d35" }}>
                        Select at least one feature to train a model.
                      </p>
                    )}
                    </fieldset>
                  </details>
                    </div>
                  </details>
                  {diagnosticApplicable && (
                    <details aria-label="Advanced Diagnostics" style={{ marginBottom: "16px" }}>
                      <summary style={{ cursor: "pointer", color: "#1e6041", fontWeight: 600 }}>
                        Advanced Diagnostics
                      </summary>
                      <p style={{ margin: "8px 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        TextToAutoML performed exploratory checks for unusual relationships between input features and the target. These checks do not by themselves prove data leakage or guarantee predictive performance.
                      </p>
                      <details>
                        <summary style={{ cursor: "pointer", color: "#526158" }}>
                          View detailed diagnostics
                        </summary>
                    <section
                      aria-label="Target-proxy diagnostic"
                      style={{
                        maxWidth: "640px",
                        marginTop: "10px",
                        padding: "12px",
                        border: "1px solid #dbe5de",
                        borderRadius: "4px",
                        background: "#fbfcfa",
                        font: "13px/1.5 Arial, sans-serif",
                      }}
                    >
                      <strong>Feature relationship check (exploratory)</strong>
                      <p style={{ margin: "5px 0 8px", color: "#526158" }}>
                        This check uses only the outer training partition and is not repeated within each CV fold. Statistical association does not prove leakage or prediction-time unavailability, and warnings do not change model inputs.
                      </p>
                      {groupColumn && (
                        <p style={{ margin: "0 0 8px", color: "#526158" }}>
                          Grouped tests use one summary per group, assume independent groups, are not cluster-robust, and discard within-group variation.
                        </p>
                      )}
                      {diagnosticState === "loading" && (
                        <p role="status" style={{ margin: 0 }}>Checking training features...</p>
                      )}
                      {diagnosticState === "error" && (
                        <div role="alert" style={{ color: "#a13d35" }}>
                          <span>{diagnosticError}</span>{" "}
                          <button
                            type="button"
                            onClick={() => setDiagnosticRetry((value) => value + 1)}
                          >
                            Retry check
                          </button>
                        </div>
                      )}
                      {diagnosticState === "success" && diagnosticReady && (
                        diagnosticResult?.warnings?.length ? (
                          <ul style={{ margin: "6px 0 0", paddingLeft: "20px" }}>
                            {diagnosticResult.warnings.map((warning, index) => (
                              <li key={`${warning.feature}-${warning.detection_type}-${index}`}>
                                <strong>{warning.feature}</strong>{" "}
                                <span>({warning.detection_type})</span>
                                <div>{warning.message}</div>
                                {warning.evidence?.method && (
                                  <div>Diagnostic: {warning.evidence.method}</div>
                                )}
                                {warning.evidence?.effect_size_name && (
                                  <div>
                                    Effect size: {warning.evidence.effect_size_name} = {warning.evidence.effect_size?.toPrecision(3)}
                                  </div>
                                )}
                                {typeof warning.evidence?.q_value === "number" && (
                                  <div>
                                    Adjusted significance: q = {warning.evidence.q_value < 0.001 ? "< 0.001" : warning.evidence.q_value.toPrecision(3)} (Benjamini-Hochberg)
                                  </div>
                                )}
                                <code style={{ display: "block", overflowWrap: "anywhere" }}>
                                  {JSON.stringify({
                                    ...warning.evidence,
                                    ...(warning.evidence?.p_value === 0
                                      ? { p_value: "underflowed to 0 at floating-point precision" }
                                      : {}),
                                    ...(typeof warning.evidence?.q_value === "number" && warning.evidence.q_value < 0.001
                                      ? { q_value: "< 0.001" }
                                      : {}),
                                  })}
                                </code>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p style={{ margin: 0 }}>
                            No exact relationship or statistically significant association at the 5% FDR level was detected in the outer training partition. This does not establish that the features are safe or available at prediction time.
                          </p>
                        )
                      )}
                    </section>
                      </details>
                    </details>
                  )}
                  <button
                    type="button"
                    onClick={confirmTraining}
                    disabled={
                      trainingState === "loading" ||
                      trainingConfirmed ||
                      selectedFeatureColumns.length === 0 ||
                      (diagnosticApplicable && !diagnosticReady)
                    }
                    style={{
                      padding: "11px 18px",
                      border: 0,
                      borderRadius: "6px",
                      background: trainingState === "loading" || trainingConfirmed ? "#7f9a89" : "#1e6041",
                      color: "#ffffff",
                      cursor: trainingState === "loading" || trainingConfirmed ? "default" : "pointer",
                      font: "600 14px Arial, sans-serif",
                    }}
                  >
                    {trainingState === "loading" ? "Training..." : "Confirm and train"}
                  </button>
                  {trainingState === "error" && (
                    <p style={{ margin: "10px 0 0", color: "#a13d35" }}>
                      {trainingError}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div
                style={{
                  minHeight: "150px",
                  borderTop: "1px solid #edf2ee",
                }}
              />
            )}
          </div>
        </section>

        {trainingState === "success" && trainingResult && (
          <section
            aria-label="Model Training Report"
            style={{ ...panelStyle, marginTop: "20px", padding: "28px" }}
          >
            {(() => {
              const candidates = Array.isArray(automlTraining.models?.trained)
                ? automlTraining.models.trained
                : [];
              const evaluationResults = Array.isArray(automlTraining.evaluation?.results)
                ? automlTraining.evaluation.results
                : [];
              const validationDiagnostics = Array.isArray(automlTraining.validation_diagnostics)
                ? automlTraining.validation_diagnostics
                : [];
              const numericColumns = trainingPreprocessing.imputed_columns?.numeric;
              const categoricalColumns = trainingPreprocessing.imputed_columns?.categorical;
              const dateColumns = trainingPreprocessing.imputed_columns?.date;
              const encodedTargets = trainingPreprocessing.target_classes;
              const clusteringEvaluation = trainingPreprocessing.clustering_evaluation || {};
              const clusteringMetrics = clusteringEvaluation.final_model_metrics || {};
              const clusterDistribution = Array.isArray(clusteringEvaluation.cluster_distribution)
                ? clusteringEvaluation.cluster_distribution
                : [];
              const isClustering = String(trainingProblemType).toLowerCase().includes("cluster");
              const selectionReason = String(bestModel.reason || "");
              const readableSelectionReason = selectionReason.toLowerCase().includes("highest f1_score")
                ? "It had the highest weighted F1-score among candidates on the validation data."
                : selectionReason.toLowerCase().includes("lowest rmse")
                  ? "It had the lowest RMSE among candidates on the validation data."
                  : selectionReason.toLowerCase().includes("highest silhouette_score")
                    ? "It had the highest silhouette score among the clustering candidates."
                    : selectionReason.toLowerCase().includes("best mean rank")
                      ? "It had the best mean rank across silhouette, Calinski-Harabasz, and Davies-Bouldin scores."
                    : selectionReason;
                const diagnosticMetricSummary = (metrics) => {
                  if (!metrics || typeof metrics !== "object") return "Not available";
                  const percentageMetrics = new Set([
                    "accuracy", "precision", "recall", "f1_score", "balanced_accuracy",
                    "macro_precision", "macro_recall", "macro_f1_score",
                  ]);
                  const entries = Object.entries(metrics).filter(
                    ([, value]) => typeof value === "number" && Number.isFinite(value),
                  );
                  return entries.length > 0
                    ? entries.map(([name, value]) => `${formatMetricLabel(name)}: ${formatPredictionMetric({ value, format: percentageMetrics.has(name) ? "percentage" : "number" })}`).join("; ")
                    : "Not available";
                };
                const renderClassDetails = (label, metrics, fallback = {}) => {
                  const confusionMatrix = metrics?.confusion_matrix || fallback.confusion_matrix;
                  const confusionLabels = metrics?.confusion_matrix_labels || fallback.confusion_matrix_labels || [];
                  const perClassMetrics = metrics?.per_class_metrics || fallback.per_class_metrics || {};
                  const classEntries = perClassMetrics && typeof perClassMetrics === "object"
                    ? Object.entries(perClassMetrics)
                    : [];
                  if (!Array.isArray(confusionMatrix) && classEntries.length === 0) return null;
                  return (
                    <div style={{ marginTop: "5px" }}>
                      <strong>{label}</strong>
                      {Array.isArray(confusionMatrix) && (
                        <div>Confusion matrix ({confusionLabels.join(", ")}): {confusionMatrix.map((row) => `[${row.join(", ")}]`).join(" ")}</div>
                      )}
                      {classEntries.length > 0 && (
                        <div>{classEntries.map(([className, values]) => `${className}: recall ${formatPredictionMetric({ value: values?.recall, format: "percentage" })}, support ${values?.support ?? "Not available"}`).join("; ")}</div>
                      )}
                    </div>
                  );
                };

              return (
                <>
                  <h2 style={{ margin: "0 0 20px", fontSize: "24px", fontWeight: 500 }}>
                    Model Training Report
                  </h2>

                  <section aria-label="What was trained" style={{ paddingBottom: "18px", borderBottom: "1px solid #edf2ee" }}>
                    <h3 style={{ margin: "0 0 12px", fontSize: "17px", fontWeight: 600 }}>A. What was trained</h3>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "12px 18px", font: "13px/1.5 Arial, sans-serif" }}>
                      <div><span>Task</span><strong style={{ display: "block" }}>{trainingProblemType || "Not returned"}</strong></div>
                      <div><span>Target</span><strong style={{ display: "block" }}>{trainingTargetColumn || (isClustering ? "Not used (clustering)" : "No target column")}</strong></div>
                      <div><span>Dataset size</span><strong style={{ display: "block" }}>{trainingData.original_rows ?? trainingResult.report?.dataset?.rows ?? uploadResult?.analysis?.dataset_info?.rows ?? "Not returned"} rows × {trainingData.original_columns ?? uploadResult?.analysis?.dataset_info?.columns ?? "Not returned"} columns</strong></div>
                      <div><span>Input features</span><strong style={{ display: "block" }}>{trainingFeatureNames?.length ?? trainingResult.report?.dataset?.features ?? selectedFeatureColumns.length}</strong></div>
                      {(automlTraining.model?.model_id || trainingResult.report?.model_id) && (
                        <div><span>Saved model ID</span><strong style={{ display: "block", overflowWrap: "anywhere" }}>{automlTraining.model?.model_id || trainingResult.report.model_id}</strong></div>
                      )}
                      {automlTraining.training_time_seconds != null && Number.isFinite(Number(automlTraining.training_time_seconds)) && (
                        <div><span>Training time</span><strong style={{ display: "block" }}>{formatDuration(Number(automlTraining.training_time_seconds))}</strong></div>
                      )}
                    </div>
                  </section>

                  <section aria-label="Data preparation" style={{ padding: "18px 0", borderBottom: "1px solid #edf2ee" }}>
                    <h3 style={{ margin: "0 0 8px", fontSize: "17px", fontWeight: 600 }}>B. Data preparation</h3>
                    <p style={{ margin: "0 0 10px", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                      {isClustering
                        ? "This clustering workflow has no target. Numeric and parseable date features are median-imputed and robust-scaled; nominal categories use one-hot encoding. Constant, identifier-like, target, and existing label columns are excluded with recorded reasons."
                        : "The target is kept separate from input features. Numeric missing values use training-set means, categorical missing values use training-set modes, and categorical inputs are encoded for the model."}
                    </p>
                    {trainingFeatureNames && (
                      <p style={{ margin: "6px 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        <strong>Input features used:</strong> {trainingFeatureNames.length ? trainingFeatureNames.join(", ") : "None returned"}
                      </p>
                    )}
                    {removedColumnEntries.length > 0 && (
                      <div style={{ marginTop: "8px", font: "13px/1.5 Arial, sans-serif" }}>
                        <strong>Columns not used for training</strong>
                        <ul style={{ margin: "4px 0 0", paddingLeft: "20px", color: "#526158" }}>
                          {removedColumnEntries.map(({ column, reason }, index) => (
                            <li key={`${column}-${reason}-${index}`}>
                              {column}: {reason || getRemovedColumnReason({ [reason]: trainingPreprocessing.removed_columns?.[reason] }, column) || "excluded during preprocessing"}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {(Array.isArray(numericColumns) || Array.isArray(categoricalColumns)) && (
                      <div style={{ marginTop: "8px", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        {Array.isArray(numericColumns) && <div><strong>Numeric columns prepared:</strong> {numericColumns.length ? numericColumns.join(", ") : "None"}</div>}
                        {Array.isArray(categoricalColumns) && <div><strong>Categorical columns prepared:</strong> {categoricalColumns.length ? categoricalColumns.join(", ") : "None"}</div>}
                        {Array.isArray(dateColumns) && <div><strong>Date columns parsed:</strong> {dateColumns.length ? dateColumns.join(", ") : "None"}</div>}
                      </div>
                    )}
                    {encoderEntries.length > 0 && (
                      <p style={{ margin: "8px 0 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        <strong>Encoders:</strong> {encoderEntries.map(([column, encoder]) => `${column} (${encoder.type})`).join(", ")}
                      </p>
                    )}
                    {isClassificationTask && Array.isArray(encodedTargets) && encodedTargets.length > 0 && (
                      <p style={{ margin: "8px 0 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        <strong>Target classes:</strong> {encodedTargets.join(", ")}
                      </p>
                    )}
                  </section>

                  <section aria-label="Dataset split" style={{ padding: "18px 0", borderBottom: "1px solid #edf2ee" }}>
                    <h3 style={{ margin: "0 0 10px", fontSize: "17px", fontWeight: 600 }}>C. Dataset split</h3>
                    {isClustering ? (
                      <p style={{ margin: 0, color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        This clustering path uses the prepared data without separate training, validation, and test partitions.
                      </p>
                    ) : (
                      <>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "10px 16px", font: "13px Arial, sans-serif" }}>
                          <div>Training rows: <strong>{trainingData.training_rows ?? "Not returned"}</strong></div>
                          <div>Validation rows: <strong>{trainingData.validation_rows ?? "Not returned"}</strong></div>
                          <div>Test rows: <strong>{trainingData.test_rows ?? "Not returned"}</strong></div>
                        </div>
                        <p style={{ margin: "10px 0 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                          Training data teaches candidate models. Validation data is used to compare and select a model. Test data stays separate until final evaluation.
                        </p>
                      </>
                    )}
                  </section>

                  <section aria-label="Models evaluated" style={{ padding: "18px 0", borderBottom: "1px solid #edf2ee" }}>
                    <h3 style={{ margin: "0 0 10px", fontSize: "17px", fontWeight: 600 }}>D. Models evaluated</h3>
                    <p style={{ margin: "0 0 8px", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>Candidate models trained:</p>
                    {candidates.length > 0 ? (
                      <ul style={{ margin: 0, paddingLeft: "20px", color: "#26372d", font: "13px/1.6 Arial, sans-serif" }}>
                        {candidates.map((name) => <li key={name}>{name}</li>)}
                      </ul>
                    ) : (
                      <p style={{ margin: 0, color: "#66756b", font: "13px Arial, sans-serif" }}>Candidate names were not returned.</p>
                    )}
                    {evaluationResults.length > 0 && (
                      <>
                        <h4 style={{ margin: "14px 0 8px", fontSize: "14px" }}>Evaluation results returned</h4>
                        <div style={{ overflowX: "auto" }}>
                          <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", font: "13px/1.5 Arial, sans-serif" }}>
                            <thead><tr><th style={{ padding: "7px", borderBottom: "1px solid #dbe5de" }}>Model</th><th style={{ padding: "7px", borderBottom: "1px solid #dbe5de" }}>{isClustering ? "Candidate sample metrics returned" : "Final test metrics returned"}</th></tr></thead>
                            <tbody>
                              {evaluationResults.map((result, index) => {
                                const metricEntries = Object.entries(result.metrics || {}).filter(([, value]) => typeof value === "number" && Number.isFinite(value));
                                return (
                                  <tr key={`${result.model_name || "model"}-${index}`}>
                                    <th scope="row" style={{ padding: "7px", borderBottom: "1px solid #edf2ee" }}>{result.model_name || "Not returned"}</th>
                                    <td style={{ padding: "7px", borderBottom: "1px solid #edf2ee" }}>
                                      {metricEntries.length > 0
                                        ? metricEntries.map(([metric, value]) => (
                                          <span key={metric} style={{ display: "inline-block", marginRight: "12px" }}>
                                            {formatMetricLabel(metric)}: {formatPredictionMetric({ value, format: ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1_score"].includes(metric) ? "percentage" : "number" })}
                                          </span>
                                        ))
                                        : "No numeric evaluation metrics returned"}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                        {isClustering && Object.keys(clusteringEvaluation).length > 0 && (
                          <div style={{ marginTop: "12px", color: "#526158", font: "13px/1.6 Arial, sans-serif" }}>
                            <strong>Selected model evaluation:</strong>{" "}
                            {[
                              ["Silhouette score", clusteringMetrics.silhouette_score],
                              ["Calinski-Harabasz", clusteringMetrics.calinski_harabasz_score],
                              ["Davies-Bouldin", clusteringMetrics.davies_bouldin_score],
                            ].filter(([, value]) => typeof value === "number" && Number.isFinite(value))
                              .map(([label, value]) => `${label}: ${formatPredictionMetric({ value, format: "number" })}`)
                              .join("; ")}
                            <div>{clusteringEvaluation.quality_assessment}</div>
                            {clusterDistribution.length > 0 && (
                              <div>
                                <strong>Cluster sizes:</strong>{" "}
                                {clusterDistribution.map((cluster) => (
                                  `${cluster.cluster}: ${cluster.rows} (${formatPredictionMetric({ value: cluster.share, format: "percentage" })})`
                                )).join("; ")}
                                {Number(clusteringEvaluation.small_cluster_count) > 0
                                  ? `; ${clusteringEvaluation.small_cluster_count} small cluster(s) are flagged for review`
                                  : ""}
                              </div>
                            )}
                            {clusteringEvaluation.selection_strategy && (
                              <div>{clusteringEvaluation.selection_strategy}</div>
                            )}
                          </div>
                        )}
                        {validationDiagnostics.length === 0 && evaluationResults.length < candidates.length && (
                          <p style={{ margin: "8px 0 0", color: "#66756b", font: "12px/1.5 Arial, sans-serif" }}>
                            Candidate-by-candidate validation scores were not included in this response; only the listed evaluation results are shown.
                          </p>
                        )}
                      </>
                    )}
                    {validationDiagnostics.length > 0 && (
                      <details aria-label="Validation results by candidate" style={{ marginTop: "14px" }}>
                        <summary style={{ cursor: "pointer", color: "#1e6041", font: "14px Arial, sans-serif" }}>
                          Validation results by candidate
                        </summary>
                        <p style={{ margin: "6px 0", color: "#526158", font: "12px/1.5 Arial, sans-serif" }}>
                          Training and validation scores are diagnostic comparisons. Final holdout/test metrics are reported separately below.
                        </p>
                        <div style={{ marginTop: "8px", overflowX: "auto" }}>
                          <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", font: "12px/1.5 Arial, sans-serif" }}>
                            <thead>
                              <tr>
                                {["Candidate", "Status", "Validation selection score", "Training metrics", "Validation metrics", "Class details"].map((label) => (
                                  <th key={label} style={{ padding: "7px", borderBottom: "1px solid #dbe5de", whiteSpace: "nowrap" }}>{label}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {validationDiagnostics.map((candidate, index) => {
                                const validationMetrics = candidate.validation_metrics || {
                                  f1_score: candidate.weighted_f1,
                                  balanced_accuracy: candidate.balanced_accuracy,
                                  macro_f1_score: candidate.macro_f1,
                                };
                                const hasClassDetails = Boolean(
                                  renderClassDetails("Training", candidate.training_metrics) ||
                                  renderClassDetails("Validation", validationMetrics, candidate),
                                );
                                return (
                                  <tr key={`${candidate.candidate_name || "candidate"}-${index}`}>
                                    <th scope="row" style={{ padding: "7px", borderBottom: "1px solid #edf2ee", whiteSpace: "nowrap" }}>{candidate.candidate_name || "Not returned"}</th>
                                    <td style={{ padding: "7px", borderBottom: "1px solid #edf2ee" }}>
                                      {String(candidate.status || "unknown").replaceAll("_", " ")}
                                      {candidate.error_reason && <div style={{ color: "#7a3d32", maxWidth: "260px", whiteSpace: "normal" }}>{candidate.error_reason}</div>}
                                      {candidate.training_metrics_error && <div style={{ color: "#7a3d32", maxWidth: "260px", whiteSpace: "normal" }}>Training metrics: {candidate.training_metrics_error}</div>}
                                    </td>
                                    <td style={{ padding: "7px", borderBottom: "1px solid #edf2ee", whiteSpace: "nowrap" }}>
                                      {formatMetricLabel(candidate.selection_metric)}: {formatPredictionMetric({ value: candidate.selection_value, format: ["accuracy", "f1_score", "balanced_accuracy"].includes(candidate.selection_metric) ? "percentage" : "number" })}
                                    </td>
                                    <td style={{ padding: "7px", borderBottom: "1px solid #edf2ee" }}>
                                      {diagnosticMetricSummary(candidate.training_metrics)}
                                    </td>
                                    <td style={{ padding: "7px", borderBottom: "1px solid #edf2ee" }}>
                                      {diagnosticMetricSummary(validationMetrics)}
                                    </td>
                                    <td style={{ padding: "7px", borderBottom: "1px solid #edf2ee" }}>
                                      {hasClassDetails ? (
                                        <details>
                                          <summary style={{ cursor: "pointer" }}>Class details</summary>
                                          {renderClassDetails("Training", candidate.training_metrics)}
                                          {renderClassDetails("Validation", validationMetrics, candidate)}
                                        </details>
                                      ) : "Not available"}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </details>
                    )}
                  </section>

                  <section aria-label="Hyperparameter tuning result" style={{ padding: "18px 0", borderBottom: "1px solid #edf2ee" }}>
                    <h3 style={{ margin: "0 0 8px", fontSize: "17px", fontWeight: 600 }}>E. Hyperparameter tuning</h3>
                    {tuningDecision ? (
                      <>
                        <p style={{ margin: 0, color: "#26372d", font: "14px/1.5 Arial, sans-serif" }}>
                          {tuningDecision.strategy === "none" || tuningDecision.trialBudget === 0
                            ? "Automatic tuning was not applied for this run."
                            : `Tuning strategy: ${tuningStrategyLabels[tuningDecision.strategy] || tuningDecision.strategy}.`}
                        </p>
                        {tuningDecision.trialBudget > 0 && (
                          <p style={{ margin: "5px 0 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                            Trial budget: {tuningDecision.trialBudget} per model.
                          </p>
                        )}
                        {tuningDecision.reason && <p style={{ margin: "5px 0 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>Why: {tuningDecision.reason}</p>}
                      </>
                    ) : tuningStrategy !== "auto" ? (
                      <p style={{ margin: 0, color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        {tuningStrategyLabels[tuningStrategy] || tuningStrategy} was requested. Execution details were not returned for this strategy.
                      </p>
                    ) : (
                      <p style={{ margin: 0, color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        Automatic optimizer decision details are unavailable from this backend response.
                      </p>
                    )}
                    {tuningMetadata && (
                      <details style={{ marginTop: "8px" }}>
                        <summary style={{ cursor: "pointer", color: "#1e6041", font: "13px Arial, sans-serif" }}>Advanced details</summary>
                        <div style={{ marginTop: "6px", color: "#526158", font: "12px/1.5 Arial, sans-serif" }}>
                          {tuningMetadata.requested_mode && <div>Requested mode: {tuningMetadata.requested_mode}</div>}
                          {tuningMetadata.tuning_duration_seconds != null && <div>Tuning duration: {formatDuration(Number(tuningMetadata.tuning_duration_seconds))}</div>}
                          {tuningMetadata.models && Object.entries(tuningMetadata.models).map(([modelName, result]) => (
                            <div key={modelName}>{modelName}: {result.status || "status not returned"}{result.trials_run != null ? ` (${result.trials_run} trials)` : ""}</div>
                          ))}
                        </div>
                      </details>
                    )}
                  </section>

                  <section aria-label="Best model" style={{ paddingTop: "18px" }}>
                    <h3 style={{ margin: "0 0 10px", fontSize: "17px", fontWeight: 600 }}>F. Best model</h3>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px 18px", font: "13px/1.5 Arial, sans-serif" }}>
                      <div><span>Best model</span><strong style={{ display: "block" }}>{bestModel.name || automlTraining.model?.name || "Not returned"}</strong></div>
                      <div><span>Selection metric</span><strong style={{ display: "block" }}>{formatMetricLabel(selectionMetric)}</strong></div>
                      {selectionScore != null && Number.isFinite(Number(selectionScore)) && (
                        <div>
                          <span>{isClustering ? "Clustering selection score" : "Model selection score (validation)"}</span>
                          <strong style={{ display: "block" }}>{formatPredictionMetric({ value: selectionScore, format: ["accuracy", "f1_score", "balanced_accuracy"].includes(selectionMetric) ? "percentage" : "number" })}</strong>
                        </div>
                      )}
                    </div>
                    {readableSelectionReason && (
                      <p style={{ margin: "10px 0 0", color: "#526158", font: "13px/1.5 Arial, sans-serif" }}>
                        <strong>Selection reason:</strong> {readableSelectionReason}
                      </p>
                    )}
                    {!isClustering && (
                      <p style={{ margin: "8px 0 0", color: "#66756b", font: "12px/1.5 Arial, sans-serif" }}>
                        The model selection score is based on validation data. Final holdout/test performance is reported separately below.
                      </p>
                    )}
                  </section>
                </>
              );
            })()}
          </section>
        )}

        {trainingState === "success" && trainingResult?.explanation && (
          <section
            aria-label="AI-Generated Explanation"
            style={{ ...panelStyle, marginTop: "20px", padding: "24px 28px" }}
          >
            <h2 style={{ margin: "0 0 10px", fontSize: "20px", fontWeight: 500 }}>
              AI-Generated Explanation
            </h2>
            <p style={{ margin: 0, color: "#526158", font: "15px/1.7 Arial, sans-serif", whiteSpace: "pre-wrap" }}>
              {trainingResult.explanation}
            </p>
            <p style={{ margin: "10px 0 0", color: "#8a988e", font: "12px/1.5 Arial, sans-serif" }}>
              This explanation is based on the actual training results returned by TextToAutoML.
            </p>
          </section>
        )}

        {trainingState === "success" && trainingResult && (
          <section
            aria-label={`${evaluationSummary.problemType} Evaluation`}
            style={{ ...panelStyle, marginTop: "20px", padding: "28px" }}
          >
            <h2 style={{ margin: "0 0 18px", fontSize: "22px", fontWeight: 500 }}>
              {evaluationSummary.problemType === "classification"
                ? "Classification Evaluation"
                : evaluationSummary.problemType === "regression"
                  ? "Regression Evaluation"
                  : evaluationSummary.problemType === "clustering"
                    ? "Clustering Evaluation"
                    : "Model Evaluation"}
            </h2>
            {evaluationSummary.problemType === "classification" && (
              <p style={{ margin: "0 0 18px", color: "#526158", font: "13px/1.6 Arial, sans-serif" }}>
                Final holdout/test performance for the selected model. Accuracy is the share of holdout records classified correctly. Weighted F1 balances precision and recall across classes, weighted by each class's support. Balanced accuracy averages class recall equally. Macro precision, macro recall, and macro F1 average their respective class metrics equally.
              </p>
            )}
            {evaluationSummary.problemType === "regression" && (
              <p style={{ margin: "0 0 18px", color: "#526158", font: "13px/1.6 Arial, sans-serif" }}>
                Final holdout/test performance for the selected model. RMSE and MAE measure prediction error in target units; lower values mean smaller errors. R² compares the model with a mean-value baseline; higher values indicate a better fit to these held-out records.
              </p>
            )}
            {evaluationSummary.problemType === "clustering" && (
              <p style={{ margin: "0 0 18px", color: "#526158", font: "13px/1.6 Arial, sans-serif" }}>
                Silhouette score summarizes how well-separated the clusters are in the evaluated data; higher values indicate more distinct clusters. This clustering workflow does not have a separate holdout/test partition.
              </p>
            )}
            {(() => {
              const definitions = {
                classification: [
                  { label: "Holdout accuracy", key: "accuracy", format: "percentage" },
                  { label: "Holdout weighted F1", key: "f1_score", format: "percentage" },
                  { label: "Balanced accuracy", key: "balanced_accuracy", format: "percentage" },
                  { label: "Macro precision", key: "macro_precision", format: "percentage" },
                  { label: "Macro recall", key: "macro_recall", format: "percentage" },
                  { label: "Macro F1", key: "macro_f1_score", format: "percentage" },
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
              const metrics = definitions[evaluationSummary.problemType] || [];

              return (
                <>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                      gap: "14px 20px",
                      marginBottom: evaluationSummary.problemType === "classification" ? "22px" : 0,
                    }}
                  >
                    {metrics.map((metric) => (
                      <div key={metric.key} style={{ borderLeft: "2px solid #7b9a82", paddingLeft: "10px" }}>
                        <span style={{ display: "block", color: "#66756b", font: "12px Arial, sans-serif" }}>
                          {metric.label}
                        </span>
                        <strong style={{ display: "block", marginTop: "4px", font: "600 17px Arial, sans-serif" }}>
                          {formatPredictionMetric({
                            ...metric,
                            value: evaluationSummary.metrics[metric.key],
                          })}
                        </strong>
                      </div>
                    ))}
                  </div>

                  {evaluationSummary.problemType === "classification" && (
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                        gap: "24px",
                        font: "13px/1.5 Arial, sans-serif",
                      }}
                    >
                      {evaluationSummary.confusionMatrix && (
                        <div>
                          <h3 style={{ margin: "0 0 8px", fontSize: "15px", fontWeight: 600 }}>
                            Confusion matrix
                          </h3>
                          <p style={{ margin: "0 0 8px", color: "#66756b", fontSize: "12px" }}>
                            Rows: actual classes; columns: predicted classes.
                          </p>
                          <div style={{ marginBottom: "10px", color: "#526158", font: "12px/1.5 Arial, sans-serif" }}>
                            <strong>How to read this:</strong> Rows represent the actual class and columns represent the model's prediction. Diagonal values are correct predictions. Values outside the diagonal show where the model confused one class with another.
                          </div>
                          <div style={{ overflowX: "auto" }}>
                            <table style={{ borderCollapse: "collapse", minWidth: "100%", textAlign: "right" }}>
                              <thead>
                                <tr>
                                  <th style={{ padding: "6px 8px", textAlign: "left" }}>Actual / predicted</th>
                                  {evaluationSummary.confusionMatrixLabels.map((label) => (
                                    <th key={label} style={{ padding: "6px 8px" }}>{label}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {evaluationSummary.confusionMatrix.map((row, rowIndex) => (
                                  <tr key={evaluationSummary.confusionMatrixLabels[rowIndex]}>
                                    <th scope="row" style={{ padding: "6px 8px", textAlign: "left" }}>
                                      {evaluationSummary.confusionMatrixLabels[rowIndex]}
                                    </th>
                                    {row.map((count, columnIndex) => (
                                      <td key={evaluationSummary.confusionMatrixLabels[columnIndex]} style={{ padding: "6px 8px" }}>
                                        {count}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}

                      {evaluationSummary.perClassMetrics.length > 0 && (
                        <div>
                          <h3 style={{ margin: "0 0 8px", fontSize: "15px", fontWeight: 600 }}>
                            Per-class metrics
                          </h3>
                          <p style={{ margin: "0 0 8px", color: "#66756b", fontSize: "12px" }}>
                            This shows how well the model performs for each individual class.
                            {perClassF1Summary && (
                              <> Highest F1: {perClassF1Summary.highest}. Lowest F1: {perClassF1Summary.lowest}.</>
                            )}
                          </p>
                          <div style={{ overflowX: "auto" }}>
                            <table style={{ borderCollapse: "collapse", minWidth: "100%", textAlign: "right" }}>
                              <thead>
                                <tr>
                                  <th style={{ padding: "6px 8px", textAlign: "left" }}>Class</th>
                                  <th style={{ padding: "6px 8px" }}>Precision</th>
                                  <th style={{ padding: "6px 8px" }}>Recall</th>
                                  <th style={{ padding: "6px 8px" }}>F1</th>
                                  <th style={{ padding: "6px 8px" }}>Support</th>
                                </tr>
                              </thead>
                              <tbody>
                                {evaluationSummary.perClassMetrics.map((item) => (
                                  <tr key={item.label}>
                                    <th scope="row" style={{ padding: "6px 8px", textAlign: "left" }}>{item.label}</th>
                                    <td style={{ padding: "6px 8px" }}>{formatPredictionMetric({ value: item.precision, format: "percentage" })}</td>
                                    <td style={{ padding: "6px 8px" }}>{formatPredictionMetric({ value: item.recall, format: "percentage" })}</td>
                                    <td style={{ padding: "6px 8px" }}>{formatPredictionMetric({ value: item.f1_score, format: "percentage" })}</td>
                                    <td style={{ padding: "6px 8px" }}>{item.support ?? "Not available"}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </>
              );
            })()}
          </section>
        )}

        {trainingState === "success" && trainingResult && (
          <section
            aria-label="Prediction"
            style={{ ...panelStyle, marginTop: "20px", padding: "28px" }}
          >
            <div style={{ marginBottom: "18px" }}>
              <p
                style={{
                  margin: "0 0 8px",
                  color: "#8a988e",
                  fontFamily: "'Courier New', monospace",
                  fontSize: "11px",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}
              >
                Next step
              </p>
              <h2 style={{ margin: 0, fontSize: "22px", fontWeight: 500 }}>
                {predictionConfig.problemType === "classification"
                  ? "Predict a class"
                  : predictionConfig.problemType === "regression"
                    ? "Predict a value"
                    : predictionConfig.problemType === "clustering"
                      ? "Assign a cluster"
                      : "Make a prediction"}
              </h2>
              {trainedTabularModelId && (
                <p
                  role="status"
                  style={{
                    margin: "12px 0 0",
                    padding: "10px 12px",
                    border: "1px solid #cfe0d4",
                    borderRadius: "6px",
                    background: "#f8fbf8",
                    color: "#526158",
                    font: "13px/1.5 Arial, sans-serif",
                    overflowWrap: "anywhere",
                  }}
                >
                  Saved model ID: <strong>{trainedTabularModelId}</strong>
                </p>
              )}
              <div style={{ display: "flex", gap: "8px", marginTop: "16px", flexWrap: "wrap" }}>
                {[{ value: "sample", label: "Use first row" }, { value: "new", label: "Enter values" }].map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => {
                      setPredictionMode(option.value);
                      setPredictionState("idle");
                      setPredictionError("");
                      setPredictionResult(null);
                    }}
                    aria-pressed={predictionMode === option.value}
                    style={{
                      padding: "9px 14px",
                      border: `1px solid ${predictionMode === option.value ? "#1e6041" : "#cbd8cf"}`,
                      borderRadius: "6px",
                      background: predictionMode === option.value ? "#e3f0e6" : "#ffffff",
                      color: "#1e6041",
                      cursor: "pointer",
                      font: "600 13px Arial, sans-serif",
                    }}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              <p style={{ margin: "10px 0 0", color: "#66756b", font: "14px/1.5 Arial, sans-serif" }}>
                {predictionMode === "sample"
                  ? "Use the first row from the uploaded dataset."
                  : `${predictionConfig.problemType === "clustering" ? "Clustering" : "Model"} feature values are optional.`}
              </p>
            </div>

            {getPredictionFeatures().length > 0 ? (
              <>
                {predictionMode === "new" && (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                      gap: "14px 18px",
                      paddingTop: "18px",
                      borderTop: "1px solid #edf2ee",
                    }}
                  >
                    {getPredictionFeatures().map((column) => {
                      const dtype = String(getFeatureType(column)).toLowerCase();
                      const inputType =
                        dtype.includes("int") || dtype.includes("float") || dtype.includes("double") || dtype.includes("number")
                          ? "number"
                          : "text";
                      return (
                        <label key={column} style={{ font: "13px/1.4 Arial, sans-serif", color: "#526158" }}>
                          <span style={{ display: "block", marginBottom: "6px", fontWeight: 600 }}>
                            {column}
                          </span>
                          <input
                            type={inputType}
                            step={inputType === "number" ? "any" : undefined}
                            value={predictionInputs[column] ?? ""}
                            onChange={(event) => updatePredictionInput(column, event.target.value)}
                            style={{
                              width: "100%",
                              boxSizing: "border-box",
                              padding: "10px 11px",
                              border: "1px solid #cbd8cf",
                              borderRadius: "6px",
                              background: "#fbfcfa",
                              font: "14px Arial, sans-serif",
                            }}
                          />
                        </label>
                      );
                    })}
                  </div>
                )}

                <div style={{ marginTop: "20px" }}>
                  <button
                    type="button"
                    onClick={handlePredict}
                    disabled={predictionState === "loading"}
                    style={{
                      padding: "11px 18px",
                      border: 0,
                      borderRadius: "6px",
                      background: predictionState === "loading" ? "#7f9a89" : "#1e6041",
                      color: "#ffffff",
                      cursor: predictionState === "loading" ? "default" : "pointer",
                      font: "600 14px Arial, sans-serif",
                    }}
                  >
                    {predictionState === "loading" ? "Predicting..." : "Get prediction"}
                  </button>
                </div>

                {predictionState === "error" && (
                  <p style={{ margin: "12px 0 0", color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>
                    {predictionError}
                  </p>
                )}

                {predictionState === "success" && predictionResult && (
                  <div
                    style={{
                      marginTop: "20px",
                      padding: "18px",
                      border: "1px solid #cfe0d4",
                      borderRadius: "8px",
                      background: "#f8fbf8",
                      font: "14px/1.5 Arial, sans-serif",
                    }}
                  >
                      <span style={{ color: "#66756b", fontSize: "12px" }}>
                        {predictionConfig.problemType === "classification"
                          ? "Predicted class"
                          : predictionConfig.problemType === "regression"
                            ? "Predicted value"
                            : "Cluster ID"}
                      </span>
                    <div style={{ marginTop: "4px", fontSize: "28px", fontWeight: 600, color: "#1e6041" }}>
                        {predictionConfig.problemType === "classification"
                          ? String(predictionResult.prediction_labels?.[0] ?? predictionResult.predictions?.[0] ?? "-")
                          : String(predictionResult.predictions?.[0] ?? "-")}
                    </div>
                      {predictionConfig.problemType === "classification" && predictionResult.probabilities?.[0] && (
                      <div style={{ marginTop: "10px", color: "#526158" }}>
                        Probabilities: {predictionResult.probabilities[0].map((value, index) => (
                          <span key={index} style={{ marginRight: "12px" }}>
                            {predictionResult.probability_labels?.[index] || `Class ${index}`}: {(Number(value) * 100).toFixed(1)}%
                          </span>
                        ))}
                      </div>
                    )}
                    <div style={{ marginTop: "10px", color: "#8a988e", fontSize: "12px" }}>
                      <span style={{ display: "block" }}>Model used</span>
                      <strong style={{ color: "#526158" }}>
                        {predictionResult.model_name || trainingResult.automl_training?.model?.name || "Saved model"}
                        {predictionResult.model_id ? ` · ${predictionResult.model_id}` : ""}
                      </strong>
                    </div>
                    <p style={{ margin: "10px 0 0", color: "#66756b", fontSize: "12px" }}>
                      The model's output is based on the input values provided.
                    </p>
                  </div>
                )}
              </>
            ) : (
              <p style={{ marginTop: "18px", color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>
                The trained model did not return its required input feature metadata.
              </p>
            )}
          </section>
        )}
        <section
          aria-label="Test Your Model"
          style={{ ...panelStyle, marginTop: "20px", padding: "28px" }}
        >
          <div style={{ marginBottom: "18px" }}>
            <p
              style={{
                margin: "0 0 8px",
                color: "#8a988e",
                fontFamily: "'Courier New', monospace",
                fontSize: "11px",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
              }}
            >
              Saved tabular model
            </p>
            <h2 style={{ margin: 0, fontSize: "22px", fontWeight: 500 }}>
              Test Your Model
            </h2>
            <p style={{ margin: "8px 0 0", color: "#66756b", font: "14px/1.5 Arial, sans-serif" }}>
              Enter a Model ID and upload new CSV or XLSX rows. This loads the saved model and fitted preprocessing only; it does not train or refit anything. Required feature columns must be present, extra columns are ignored, and missing cell values use the saved training-time imputations.
            </p>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "14px 18px",
              alignItems: "end",
            }}
          >
            <label htmlFor="saved-model-id" style={{ font: "13px/1.4 Arial, sans-serif", color: "#526158" }}>
              <span style={{ display: "block", marginBottom: "6px", fontWeight: 600 }}>Model ID</span>
              <input
                id="saved-model-id"
                type="text"
                value={savedModelIdInput}
                onChange={(event) => {
                  setSavedModelIdInput(event.target.value);
                  setSavedModelPredictionState("idle");
                  setSavedModelPredictionError("");
                  setSavedModelPredictionResult(null);
                }}
                placeholder="automl_..."
                autoComplete="off"
                style={{
                  width: "100%",
                  boxSizing: "border-box",
                  padding: "10px 11px",
                  border: "1px solid #cbd8cf",
                  borderRadius: "6px",
                  background: "#fbfcfa",
                  font: "14px Arial, sans-serif",
                }}
              />
            </label>
            <label htmlFor="saved-model-file" style={{ font: "13px/1.4 Arial, sans-serif", color: "#526158" }}>
              <span style={{ display: "block", marginBottom: "6px", fontWeight: 600 }}>New prediction data</span>
              <input
                id="saved-model-file"
                type="file"
                accept=".csv,.xlsx"
                onChange={(event) => {
                  setSavedModelFile(event.target.files?.[0] || null);
                  setSavedModelPredictionState("idle");
                  setSavedModelPredictionError("");
                  setSavedModelPredictionResult(null);
                }}
                style={{ width: "100%", font: "13px Arial, sans-serif" }}
              />
            </label>
          </div>

          <button
            type="button"
            onClick={handleSavedModelFilePredict}
            disabled={
              savedModelPredictionState === "loading" ||
              !savedModelIdInput.trim() ||
              !savedModelFile
            }
            style={{
              marginTop: "16px",
              padding: "11px 18px",
              border: 0,
              borderRadius: "6px",
              background: savedModelPredictionState === "loading" ? "#7f9a89" : "#1e6041",
              color: "#ffffff",
              cursor: savedModelPredictionState === "loading" ? "default" : "pointer",
              font: "600 14px Arial, sans-serif",
            }}
          >
            {savedModelPredictionState === "loading" ? "Testing saved model..." : "Test saved model"}
          </button>

          {savedModelPredictionState === "error" && (
            <p role="alert" style={{ margin: "12px 0 0", color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>
              {savedModelPredictionError}
            </p>
          )}

          {savedModelPredictionState === "success" && savedModelPredictionResult && (
            <div style={{ marginTop: "20px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
                <p style={{ margin: 0, color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>
                  {savedModelPredictionResult.prediction_count} rows processed with{" "}
                  <strong>{savedModelPredictionResult.model_name || "saved model"}</strong>
                  {" · "}{savedModelPredictionResult.model_id}
                </p>
                <button
                  type="button"
                  onClick={downloadSavedModelPredictions}
                  style={{
                    padding: "9px 13px",
                    border: "1px solid #1e6041",
                    borderRadius: "6px",
                    background: "#ffffff",
                    color: "#1e6041",
                    cursor: "pointer",
                    font: "600 13px Arial, sans-serif",
                  }}
                >
                  Download results CSV
                </button>
              </div>
              <div style={{ marginTop: "12px", overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", font: "13px/1.5 Arial, sans-serif" }}>
                  <thead>
                    <tr>
                      <th style={{ padding: "8px", borderBottom: "1px solid #dbe5de" }}>Row</th>
                      <th style={{ padding: "8px", borderBottom: "1px solid #dbe5de" }}>
                        {String(savedModelPredictionResult.problem_type || "").toLowerCase().includes("cluster")
                          ? "Cluster assignment"
                          : String(savedModelPredictionResult.problem_type || "").toLowerCase().includes("regression")
                            ? "Predicted value"
                            : "Predicted class"}
                      </th>
                      {savedModelPredictionResult.probabilities && (
                        <th style={{ padding: "8px", borderBottom: "1px solid #dbe5de" }}>Class probabilities</th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {savedModelPredictionResult.predictions.map((prediction, index) => (
                      <tr key={index}>
                        <td style={{ padding: "8px", borderBottom: "1px solid #edf2ee" }}>{index + 1}</td>
                        <td style={{ padding: "8px", borderBottom: "1px solid #edf2ee", fontWeight: 600 }}>
                          {savedModelPredictionResult.prediction_labels?.[index] ?? String(prediction)}
                        </td>
                        {savedModelPredictionResult.probabilities && (
                          <td style={{ padding: "8px", borderBottom: "1px solid #edf2ee" }}>
                            {savedModelPredictionResult.probabilities[index]?.map((value, probabilityIndex) => (
                              <span key={probabilityIndex} style={{ display: "inline-block", marginRight: "12px" }}>
                                {savedModelPredictionResult.probability_labels?.[probabilityIndex] || `Class ${probabilityIndex + 1}`}: {(Number(value) * 100).toFixed(1)}%
                              </span>
                            ))}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
          </div>
        ) : (
          <section
            aria-label="Image workflow"
            style={{ ...panelStyle, padding: "28px" }}
          >
            <div style={{ marginBottom: "20px" }}>
              <p
                style={{
                  margin: "0 0 8px",
                  color: "#8a988e",
                  fontFamily: "'Courier New', monospace",
                  fontSize: "11px",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}
              >
                Image classification
              </p>
              <h2 style={{ margin: 0, fontSize: "26px", fontWeight: 500 }}>
                Add an image dataset
              </h2>
            </div>

            <label
              htmlFor="image-task-input"
              style={{ display: "block", marginBottom: "10px", color: "#526158", font: "14px Arial, sans-serif" }}
            >
              What would you like to do with images?
            </label>
            <textarea
              id="image-task-input"
              value={task}
              onChange={handleImageRequestChange}
              placeholder="e.g. Train a CNN on these images"
              rows={3}
              style={{
                width: "100%",
                boxSizing: "border-box",
                resize: "vertical",
                marginBottom: "18px",
                padding: "14px",
                border: "1px solid #cbd8ce",
                borderRadius: "6px",
                background: "#fbfcfa",
                color: "#17221d",
                font: "16px/1.5 Arial, sans-serif",
                outlineColor: "#1e6041",
              }}
            />
            <p style={{ margin: "0 0 12px", color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>
              Add an image dataset ZIP for training, or a single image for prediction.
            </p>
            <pre
              style={{
                margin: "0 0 18px",
                padding: "12px 14px",
                overflowX: "auto",
                border: "1px solid #dbe5de",
                borderRadius: "6px",
                background: "#f8fbf8",
                color: "#526158",
                font: "13px/1.5 'Courier New', monospace",
              }}
            >{`dataset.zip
  cats/
    cat1.jpg
    cat2.jpg
  dogs/
    dog1.jpg
    dog2.jpg`}</pre>

            <label
              htmlFor="image-dataset-upload"
              style={{
                display: "block",
                maxWidth: "520px",
                padding: "16px",
                border: "1px dashed #aebfb3",
                borderRadius: "8px",
                background: "#f8fbf8",
                cursor: "pointer",
                font: "14px/1.5 Arial, sans-serif",
              }}
            >
              <strong style={{ display: "block", color: "#294534" }}>
                {imageZipFile?.name || "Choose a ZIP dataset"}
              </strong>
              <span style={{ display: "block", marginTop: "5px", color: "#7b8a80" }}>
                {imageTrainingState === "loading" ? "Training image classifier..." : "Selecting a ZIP will not start training"}
              </span>
              <input
                id="image-dataset-upload"
                aria-label="Image dataset ZIP"
                type="file"
                accept=".zip,application/zip"
                onChange={handleImageDatasetChange}
                style={{ display: "none" }}
              />
            </label>
            {imageZipValidationError && (
              <p role="alert" style={{ margin: "8px 0 0", color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>
                {imageZipValidationError}
              </p>
            )}
            {imageZipFile && (
              <p role="status" style={{ margin: "8px 0 0", color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>
                Selected ZIP: {imageZipFile.name}. Archive contents are checked before training starts.
              </p>
            )}

            <label htmlFor="image-prediction-upload" style={{ display: "block", margin: "18px 0 8px", color: "#526158", font: "14px Arial, sans-serif" }}>
              Single image for prediction
            </label>
            <input
              id="image-prediction-upload"
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.webp"
              onChange={(event) => {
                imagePredictionRequestRef.current += 1;
                imagePredictionInFlightRef.current = null;
                setImagePredictionFile(event.target.files?.[0] || null);
                setImagePredictionState("idle");
                setImagePredictionError("");
                setImagePredictionResult(null);
                clearImageRouting();
              }}
            />

            <button
              type="button"
              onClick={handleImageRoute}
              disabled={imageRoutingState === "loading" || !task.trim()}
              style={{ display: "block", marginTop: "18px", padding: "11px 18px", border: 0, borderRadius: "6px", background: imageRoutingState === "loading" ? "#7f9a89" : "#1e6041", color: "#ffffff", cursor: imageRoutingState === "loading" ? "wait" : "pointer", font: "600 14px Arial, sans-serif" }}
            >
              {imageRoutingState === "loading" ? "Checking request..." : "Understand image request"}
            </button>

            {imageRoutingState === "error" && (
              <p role="alert" style={{ margin: "12px 0 0", color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>
                {imageRoutingError}
              </p>
            )}

            {imageRoutingState === "success" && imageRoutingResult && (
              <section aria-label="Image routing result" style={{ marginTop: "20px", padding: "16px", border: "1px solid #dbe5de", borderRadius: "6px", background: "#f8fbf8", font: "14px/1.5 Arial, sans-serif" }}>
                <div><span style={{ color: "#66756b" }}>Requested action</span><strong style={{ display: "block", overflowWrap: "anywhere" }}>{imageRoutedText}</strong></div>
                <div style={{ marginTop: "10px" }}><span style={{ color: "#66756b" }}>Detected intent</span><strong style={{ display: "block" }}>{imageIntentLabels[imageRoutingResult.intent] || "Needs Clarification"}</strong></div>
                <div style={{ marginTop: "10px" }}>
                  <span style={{ color: "#66756b" }}>Status</span>
                  <strong style={{ display: "block" }}>
                    {imageRoutingResult.intent === "IMAGE_DATASET_ANALYSIS"
                      ? imageAnalysisLoading
                        ? "Analyzing dataset..."
                        : imageAnalysisResult
                          ? "Analysis complete"
                          : !imageZipFile
                            ? "Image dataset ZIP required"
                            : imageAnalysisError
                              ? "Analysis could not be completed"
                              : "Analysis ready"
                      : imageRoutingResult.ready
                        ? "Ready"
                        : imageRoutingResult.reason}
                  </strong>
                </div>
                {imageRoutingResult.ready && <p style={{ margin: "8px 0 0", color: "#526158" }}>{imageRoutingResult.reason}</p>}
                {imageRoutingResult.required_context.length > 0 && (
                  <p style={{ margin: "8px 0 0", color: "#526158" }}>
                    Required: {imageRoutingResult.required_context.map((item) => imageContextLabels[item] || item).join(", ")}.
                  </p>
                )}
                {imageZipFile &&
                  ((imageRoutingResult.intent === "IMAGE_TRAINING_REQUEST" && imageRoutingResult.ready) || imagePredictionFile) && (
                    <div style={{ marginTop: "14px", paddingTop: "12px", borderTop: "1px solid #dbe5de" }}>
                      <p style={{ margin: 0, color: "#526158" }}>
                        Training dataset: <strong>{imageZipFile.name}</strong>
                      </p>
                      {imagePredictionFile && (
                        <p style={{ margin: "6px 0 0", color: "#526158" }}>
                          Both inputs are selected. Choose Train Image Model or use the separate prediction action below.
                        </p>
                      )}
                      <button
                        type="button"
                        onClick={startImageTraining}
                        disabled={imageTrainingState === "loading" || imageRoutedText !== task.trim()}
                        style={{ display: "block", marginTop: "12px", padding: "11px 18px", border: 0, borderRadius: "6px", background: imageTrainingState === "loading" ? "#7f9a89" : "#1e6041", color: "#ffffff", cursor: imageTrainingState === "loading" ? "wait" : "pointer", font: "600 14px Arial, sans-serif" }}
                      >
                        {imageTrainingState === "loading" ? "Training image classifier..." : "Train Image Model"}
                      </button>
                    </div>
                  )}
                {imageTrainingError && imageTrainingState === "error" && (
                  <p role="alert" style={{ margin: "10px 0 0", color: "#a13d35" }}>{imageTrainingError}</p>
                )}
                {imageRoutingResult.intent === "IMAGE_TRAINING_REPORT" && imageRoutingResult.ready && imageTrainingResult && (
                  <p style={{ margin: "10px 0 0", color: "#526158" }}>The current image training result is displayed below.</p>
                )}
              </section>
            )}

            {imageRoutingState === "success" && imageRoutingResult?.intent === "IMAGE_DATASET_ANALYSIS" && (
              <section
                aria-label="Image Dataset Analysis"
                style={{ marginTop: "20px", padding: "20px", border: "1px solid #dbe5de", borderRadius: "6px", background: "#ffffff" }}
              >
                <h3 style={{ margin: "0 0 16px", fontSize: "20px", fontWeight: 600 }}>
                  Image Dataset Analysis
                </h3>
                {imageAnalysisLoading && (
                  <p role="status" style={{ margin: 0, color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>
                    Analyzing the selected image dataset...
                  </p>
                )}
                {imageAnalysisError && (
                  <p role="alert" style={{ margin: 0, color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>
                    {imageAnalysisError}
                  </p>
                )}
                {imageAnalysisResult && (
                  <div style={{ font: "14px/1.5 Arial, sans-serif" }}>
                    <h4 style={{ margin: "18px 0 8px", fontSize: "16px" }}>Dataset Overview</h4>
                    <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "10px 18px", margin: 0 }}>
                      <div><dt>Total Images</dt><dd style={{ margin: 0, fontWeight: 600 }}>{displayAnalysisNumber(imageAnalysisResult.dataset?.image_count, 0)}</dd></div>
                      <div><dt>Number of Classes</dt><dd style={{ margin: 0, fontWeight: 600 }}>{displayAnalysisNumber(imageAnalysisResult.dataset?.class_count, 0)}</dd></div>
                      <div><dt>Archive Size</dt><dd style={{ margin: 0, fontWeight: 600 }}>{displayAnalysisBytes(imageAnalysisResult.dataset?.archive_size_bytes)}</dd></div>
                      <div><dt>Extracted Size</dt><dd style={{ margin: 0, fontWeight: 600 }}>{displayAnalysisBytes(imageAnalysisResult.dataset?.extracted_size_bytes)}</dd></div>
                    </dl>

                    <h4 style={{ margin: "20px 0 8px", fontSize: "16px" }}>Class Distribution</h4>
                    {imageAnalysisClasses.length > 0 ? (
                      <div style={{ display: "grid", gap: "6px" }}>
                        {imageAnalysisClasses.map((className) => {
                          const distribution = imageAnalysisResult.classes?.distribution?.[className];
                          const count = imageAnalysisResult.classes?.images_per_class?.[className];
                          return (
                            <div key={className}>
                              {className} — {displayAnalysisNumber(count, 0)} images — {displayAnalysisNumber(distribution?.percentage)}%
                            </div>
                          );
                        })}
                      </div>
                    ) : <p style={{ margin: 0 }}>Not available</p>}

                    <h4 style={{ margin: "20px 0 8px", fontSize: "16px" }}>Dataset Splits</h4>
                    {imageAnalysisResult.splits?.source === "not_provided" ? (
                      <p style={{ margin: 0 }}>No predefined train/validation/test split was provided.</p>
                    ) : imageAnalysisResult.splits?.source === "provided" ? (
                      <>
                        <p style={{ margin: "0 0 8px" }}>Train, validation, and test splits were provided.</p>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "8px 16px" }}>
                          {["train", "validation", "test"].map((splitName) => (
                            <div key={splitName}>
                              <span>{splitName[0].toUpperCase() + splitName.slice(1)}</span>
                              <strong style={{ display: "block" }}>{displayAnalysisNumber(imageAnalysisResult.splits?.counts?.[splitName], 0)}</strong>
                            </div>
                          ))}
                        </div>
                        <p style={{ margin: "12px 0 6px", fontWeight: 600 }}>Per-class counts</p>
                        {imageAnalysisPerClassSplits.length > 0 ? (
                          <div style={{ display: "grid", gap: "6px" }}>
                            {imageAnalysisClasses.map((className) => (
                              <div key={className}>
                                {className}: {["train", "validation", "test"].map((splitName) =>
                                  `${splitName} ${displayAnalysisNumber(imageAnalysisResult.splits?.per_class_counts?.[className]?.[splitName], 0)}`,
                                ).join(" · ")}
                              </div>
                            ))}
                          </div>
                        ) : <p style={{ margin: 0 }}>Not available</p>}
                      </>
                    ) : <p style={{ margin: 0 }}>Split information not available</p>}

                    <h4 style={{ margin: "20px 0 8px", fontSize: "16px" }}>Image Properties</h4>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "8px 18px" }}>
                      <div>Width: min {displayAnalysisNumber(imageAnalysisProperties.width?.min, 0)} px, max {displayAnalysisNumber(imageAnalysisProperties.width?.max, 0)} px, average {displayAnalysisNumber(imageAnalysisProperties.width?.average)} px</div>
                      <div>Height: min {displayAnalysisNumber(imageAnalysisProperties.height?.min, 0)} px, max {displayAnalysisNumber(imageAnalysisProperties.height?.max, 0)} px, average {displayAnalysisNumber(imageAnalysisProperties.height?.average)} px</div>
                      <div>Aspect ratio: min {displayAnalysisNumber(imageAnalysisProperties.aspect_ratio?.min)}, max {displayAnalysisNumber(imageAnalysisProperties.aspect_ratio?.max)}, average {displayAnalysisNumber(imageAnalysisProperties.aspect_ratio?.average)}</div>
                    </div>
                    {[ ["File Extensions", imageAnalysisProperties.extensions], ["Decoded Formats", imageAnalysisProperties.formats], ["Source Image Modes", imageAnalysisProperties.modes], ["Source Channel Counts", imageAnalysisProperties.channels] ].map(([label, values]) => {
                      const entries = analysisEntries(values);
                      return (
                        <div key={label} style={{ marginTop: "10px" }}>
                          <strong>{label}</strong>
                          {entries.length > 0
                            ? <div>{entries.map(([name, count]) => `${name}: ${displayAnalysisNumber(count, 0)}`).join(" · ")}</div>
                            : <div>Not available</div>}
                        </div>
                      );
                    })}
                    {imageAnalysisProperties.channel_note && (
                      <p style={{ margin: "6px 0 0", color: "#66756b" }}>{imageAnalysisProperties.channel_note}</p>
                    )}

                    <h4 style={{ margin: "20px 0 8px", fontSize: "16px" }}>Quality Metrics</h4>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "8px 18px" }}>
                      <div>Dimensions consistent: {typeof imageQualityMetrics.dimension_consistency?.consistent === "boolean"
                        ? imageQualityMetrics.dimension_consistency.consistent ? "Yes" : "No"
                        : "Not available"}</div>
                      <div>Unique dimensions: {displayAnalysisNumber(imageQualityMetrics.dimension_consistency?.unique_dimensions, 0)}</div>
                      <div>Minimum pixel area: {displayAnalysisNumber(imageQualityMetrics.resolution?.min_pixel_area, 0)}</div>
                      <div>Maximum pixel area: {displayAnalysisNumber(imageQualityMetrics.resolution?.max_pixel_area, 0)}</div>
                      <div>Average pixel area: {displayAnalysisNumber(imageQualityMetrics.resolution?.average_pixel_area)}</div>
                      <div>Median pixel area: {displayAnalysisNumber(imageQualityMetrics.resolution?.median_pixel_area)}</div>
                      <div>Minimum images per class: {displayAnalysisNumber(imageQualityMetrics.class_size?.minimum_images_per_class, 0)}</div>
                      <div>Class imbalance ratio: {displayAnalysisNumber(imageQualityMetrics.class_balance?.imbalance_ratio)}</div>
                    </div>

                    <h5 style={{ margin: "16px 0 6px", fontSize: "14px" }}>Exact Duplicate Check</h5>
                    <p style={{ margin: 0, color: "#66756b" }}>
                      Checks exact byte-for-byte matches using SHA-256. Re-encoded copies may not match.
                    </p>
                    <div style={{ marginTop: "8px" }}>
                      Duplicate groups: {displayAnalysisNumber(imageDuplicateMetrics.duplicate_group_count, 0)}
                      {" · "}Duplicate files: {displayAnalysisNumber(imageDuplicateMetrics.duplicate_file_count, 0)}
                    </div>
                    {imageDuplicateMetrics.duplicate_group_count === 0 ? (
                      <p style={{ margin: "6px 0 0" }}>No exact duplicates were detected.</p>
                    ) : imageDuplicateGroups.length > 0 ? (
                      <div style={{ display: "grid", gap: "8px", marginTop: "8px" }}>
                        {imageDuplicateGroups.map((group, index) => (
                          <div key={group.hash || index} style={{ padding: "8px 10px", border: "1px solid #dbe5de", borderRadius: "4px" }}>
                            <div>Group {index + 1}: {displayAnalysisNumber(group.file_count, 0)} files</div>
                            {typeof group.hash === "string" && <code style={{ display: "block", overflowWrap: "anywhere", color: "#66756b" }}>{group.hash}</code>}
                            {Array.isArray(group.files) && group.files.length > 0
                              ? <div style={{ marginTop: "4px" }}>{group.files.map((path, pathIndex) => <div key={`${path}-${pathIndex}`} style={{ overflowWrap: "anywhere" }}>{path}</div>)}</div>
                              : <div style={{ marginTop: "4px" }}>Path samples not available</div>}
                            {group.files_truncated === true && <p style={{ margin: "4px 0 0", color: "#66756b" }}>Path list is a bounded sample.</p>}
                          </div>
                        ))}
                        {imageDuplicateMetrics.groups_truncated === true && (
                          <p style={{ margin: 0, color: "#66756b" }}>Some duplicate groups are omitted from this bounded sample; totals include them.</p>
                        )}
                      </div>
                    ) : (
                      <p style={{ margin: "6px 0 0" }}>Duplicate group details not available.</p>
                    )}

                    <h5 style={{ margin: "16px 0 6px", fontSize: "14px" }}>Cross-Split Duplicate Check</h5>
                    {imageAnalysisResult.splits?.source === "not_provided" || imageCrossSplitMetrics.checked === false ? (
                      <p style={{ margin: 0 }}>Not applicable / not checked because the dataset has no predefined splits.</p>
                    ) : imageCrossSplitMetrics.checked === true ? (
                      <>
                        <p style={{ margin: 0 }}>
                          Exact duplicate overlap groups: {displayAnalysisNumber(imageCrossSplitMetrics.match_count, 0)}
                        </p>
                        {imageCrossSplitMetrics.match_count === 0
                          ? <p style={{ margin: "4px 0 0", color: "#66756b" }}>No exact-byte overlap was detected; this does not rule out other forms of leakage.</p>
                          : imageCrossSplitMatches.length > 0
                            ? <div style={{ display: "grid", gap: "6px", marginTop: "8px" }}>
                              {imageCrossSplitMatches.map((match, index) => (
                                <div key={match.hash || index} style={{ overflowWrap: "anywhere" }}>
                                  {match.hash && <code>{match.hash} </code>}
                                  {displayAnalysisNumber(match.file_count, 0)} files across {Array.isArray(match.splits) ? match.splits.join(", ") : "splits not available"}
                                </div>
                              ))}
                              {imageCrossSplitMetrics.matches_truncated === true && (
                                <p style={{ margin: 0, color: "#66756b" }}>Some overlap groups are omitted from this bounded sample; totals include them.</p>
                              )}
                              <p style={{ margin: 0, color: "#66756b" }}>This is a possible exact-duplicate split overlap, not confirmation of model leakage.</p>
                            </div>
                            : <p style={{ margin: "4px 0 0" }}>Overlap match details not available.</p>}
                      </>
                    ) : (
                      <p style={{ margin: 0 }}>Not available</p>
                    )}

                    <h4 style={{ margin: "20px 0 8px", fontSize: "16px" }}>Validation</h4>
                    <p style={{ margin: 0 }}>
                      Validation: {imageAnalysisResult.quality?.validation === "passed"
                        ? "Passed"
                        : imageAnalysisResult.quality?.validation || "Not available"}
                    </p>
                    <p style={{ margin: "6px 0 4px", fontWeight: 600 }}>Findings</p>
                    {imageAnalysisFindings === null
                      ? <p style={{ margin: 0 }}>Not available</p>
                      : imageAnalysisFindings.length > 0
                        ? <ul style={{ margin: 0, paddingLeft: "20px" }}>{imageAnalysisFindings.map((finding, index) => <li key={index}>{typeof finding === "string" ? finding : JSON.stringify(finding)}</li>)}</ul>
                        : <p style={{ margin: 0 }}>No findings reported.</p>}

                    <h4 style={{ margin: "20px 0 8px", fontSize: "16px" }}>CNN Suitability</h4>
                    <p style={{ margin: 0 }}>
                      {imageAnalysisResult.cnn_suitability?.status === "not_assessed"
                        ? "Not assessed yet"
                        : imageAnalysisResult.cnn_suitability?.status || "Not available"}
                    </p>
                    {Array.isArray(imageAnalysisResult.cnn_suitability?.reasons) && imageAnalysisResult.cnn_suitability.reasons.length > 0 && (
                      <ul style={{ margin: "6px 0 0", paddingLeft: "20px" }}>
                        {imageAnalysisResult.cnn_suitability.reasons.map((reason, index) => <li key={index}>{reason}</li>)}
                      </ul>
                    )}
                  </div>
                )}
              </section>
            )}

            {imageTrainingState === "error" && (
              <p role="alert" style={{ margin: "12px 0 0", color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>
                {imageTrainingError}
              </p>
            )}

            {imageTrainingState === "success" && imageTrainingResult && (
              <section
                aria-label="Image training result"
                style={{ marginTop: "22px", paddingTop: "18px", borderTop: "1px solid #edf2ee" }}
              >
                <h3 style={{ margin: "0 0 14px", fontSize: "18px", fontWeight: 600 }}>
                  Training result
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px 18px", font: "13px/1.5 Arial, sans-serif" }}>
                  {imageTrainingResult.task && <div><span>Task</span><strong style={{ display: "block" }}>{imageTrainingResult.task}</strong></div>}
                  {imageTrainingResult.model && <div><span>Model</span><strong style={{ display: "block" }}>{imageTrainingResult.model}</strong></div>}
                  {Array.isArray(imageTrainingResult.classes) && <div><span>Classes</span><strong style={{ display: "block" }}>{imageTrainingResult.classes.join(", ")}</strong></div>}
                  {imageTrainingResult.metrics?.accuracy != null && Number.isFinite(Number(imageTrainingResult.metrics.accuracy)) && (
                    <div><span>Accuracy</span><strong style={{ display: "block" }}>{(Number(imageTrainingResult.metrics.accuracy) * 100).toFixed(1)}%</strong></div>
                  )}
                  {imageTrainingResult.samples?.train != null && <div><span>Training samples</span><strong style={{ display: "block" }}>{imageTrainingResult.samples.train}</strong></div>}
                  {imageTrainingResult.samples?.validation != null && <div><span>Validation samples</span><strong style={{ display: "block" }}>{imageTrainingResult.samples.validation}</strong></div>}
                  {imageTrainingResult.samples?.test != null && <div><span>Test samples</span><strong style={{ display: "block" }}>{imageTrainingResult.samples.test}</strong></div>}
                  {imageTrainingResult.training_time_seconds != null && Number.isFinite(Number(imageTrainingResult.training_time_seconds)) && (
                    <div><span>Training time</span><strong style={{ display: "block" }}>{formatDuration(Number(imageTrainingResult.training_time_seconds))}</strong></div>
                  )}
                  {imageTrainingResult.model_id && <div><span>Model ID</span><strong style={{ display: "block", overflowWrap: "anywhere" }}>{imageTrainingResult.model_id}</strong></div>}
                </div>
              </section>
            )}

            {imageModelSource === "none" && (
              <section aria-label="Use existing image model" style={{ marginTop: "24px", paddingTop: "18px", borderTop: "1px solid #edf2ee" }}>
                <h3 style={{ margin: "0 0 10px", fontSize: "18px", fontWeight: 600 }}>Use an existing image model</h3>
                <label htmlFor="existing-image-model-id" style={{ display: "block", marginBottom: "6px", color: "#526158", font: "14px Arial, sans-serif" }}>
                  Trained image model ID
                </label>
                <input
                  id="existing-image-model-id"
                  type="text"
                  value={existingImageModelIdInput}
                  onChange={(event) => {
                    setExistingImageModelIdInput(event.target.value);
                    setExistingImageModelIdError("");
                  }}
                  placeholder="Enter an existing model ID"
                  style={{ display: "block", width: "100%", maxWidth: "480px", boxSizing: "border-box", padding: "10px 12px", border: "1px solid #cbd8ce", borderRadius: "5px", font: "14px Arial, sans-serif" }}
                />
                <button
                  type="button"
                  onClick={useExistingImageModel}
                  style={{ display: "block", marginTop: "10px", padding: "9px 14px", border: "1px solid #8aa294", borderRadius: "5px", background: "#ffffff", color: "#294534", cursor: "pointer", font: "600 13px Arial, sans-serif" }}
                >
                  Use Existing Model
                </button>
                {existingImageModelIdError && <p role="alert" style={{ margin: "8px 0 0", color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>{existingImageModelIdError}</p>}
              </section>
            )}

            {imageModelId.trim() && ["trained", "existing"].includes(imageModelSource) && (
            <section aria-label="Image prediction" style={{ marginTop: "24px", paddingTop: "18px", borderTop: "1px solid #edf2ee" }}>
                <h3 style={{ margin: "0 0 12px", fontSize: "18px", fontWeight: 600 }}>Test your model</h3>
                <p role="status" style={{ margin: "0 0 10px", color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>
                  {imageModelSource === "trained"
                    ? imageTrainingState === "success"
                      ? "Training succeeded. This model is ready for testing."
                      : "Previously trained model selected. The current training attempt has not replaced it."
                    : "Using the selected existing model. The backend will validate it during prediction."}
                </p>
                <label htmlFor="image-model-id" style={{ display: "block", marginBottom: "6px", color: "#526158", font: "14px Arial, sans-serif" }}>
                  Trained image model ID
                </label>
                <input
                  id="image-model-id"
                  type="text"
                  value={imageModelId}
                  disabled={imagePredictionState === "loading"}
                  onChange={(event) => {
                    imagePredictionRequestRef.current += 1;
                    imagePredictionInFlightRef.current = null;
                    setImageModelId(event.target.value);
                    setImageModelSource(event.target.value.trim() ? "existing" : "none");
                    setImagePredictionState("idle");
                    setImagePredictionError("");
                    setImagePredictionResult(null);
                    clearImageRouting();
                  }}
                  placeholder="Enter an existing model ID"
                  style={{ display: "block", width: "100%", maxWidth: "480px", boxSizing: "border-box", padding: "10px 12px", border: "1px solid #cbd8ce", borderRadius: "5px", font: "14px Arial, sans-serif" }}
                />
                {!imageModelId.trim() && (
                  <p role="status" style={{ margin: "8px 0 0", color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>
                    Prediction requires a trained image model. Enter an existing model ID or train the selected ZIP first.
                  </p>
                )}
                <p style={{ margin: "8px 0 0", color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>
                  {imagePredictionFile ? `Selected image: ${imagePredictionFile.name}` : "Choose a single image above to predict."}
                </p>
                <button
                  type="button"
                  onClick={handleImagePredict}
                  disabled={imagePredictionState === "loading"}
                  style={{ display: "block", marginTop: "12px", padding: "11px 18px", border: 0, borderRadius: "6px", background: imagePredictionState === "loading" ? "#7f9a89" : "#1e6041", color: "#ffffff", cursor: imagePredictionState === "loading" ? "wait" : "pointer", font: "600 14px Arial, sans-serif" }}
                >
                  {imagePredictionState === "loading" ? "Predicting..." : "Predict Image"}
                </button>
                {imagePredictionState === "loading" && <p role="status" style={{ margin: "10px 0 0", color: "#526158", font: "14px/1.5 Arial, sans-serif" }}>Running image prediction...</p>}
                {imagePredictionState === "error" && <p role="alert" style={{ margin: "12px 0 0", color: "#a13d35", font: "14px/1.5 Arial, sans-serif" }}>{imagePredictionError}</p>}
                {imagePredictionState === "success" && imagePredictionResult && (
                  <div aria-label="Image prediction result" style={{ marginTop: "18px", padding: "16px", border: "1px solid #cfe0d4", borderRadius: "6px", background: "#f8fbf8", font: "14px/1.5 Arial, sans-serif" }}>
                    <span style={{ color: "#66756b", fontSize: "12px" }}>Predicted class</span>
                    <div style={{ marginTop: "3px", color: "#1e6041", fontSize: "24px", fontWeight: 600 }}>{imagePredictionResult.prediction?.class ?? "Not available"}</div>
                    {imagePredictionResult.prediction?.confidence != null && Number.isFinite(Number(imagePredictionResult.prediction.confidence)) && (
                      <div style={{ marginTop: "8px", color: "#526158" }}>Confidence: {(Number(imagePredictionResult.prediction.confidence) * 100).toFixed(1)}%</div>
                    )}
                    {imagePredictionResult.probabilities && typeof imagePredictionResult.probabilities === "object" && (
                      <div style={{ marginTop: "8px", color: "#526158" }}>
                        Probabilities: {Object.entries(imagePredictionResult.probabilities).map(([className, probability]) => (
                          <span key={className} style={{ display: "inline-block", marginRight: "12px" }}>
                            {className}: {(Number(probability) * 100).toFixed(1)}%
                          </span>
                        ))}
                      </div>
                    )}
                    {imagePredictionResult.explanation && (
                      <div aria-label="Image explanation" style={{ marginTop: "12px" }}>
                        <strong>Explanation</strong>
                        <p style={{ margin: "4px 0 0", color: "#526158" }}>{imagePredictionResult.explanation}</p>
                      </div>
                    )}
                  </div>
                )}
              </section>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
