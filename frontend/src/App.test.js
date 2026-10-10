import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  getNlpTrainingOutcome,
  getUploadValidationError,
  IMAGE_DATASET_RESET,
  resetImageDatasetWorkflow,
  resetTabularDatasetWorkflow,
  shouldRunTargetProxyDiagnostic,
  TABULAR_DATASET_RESET,
} from "./workflowLifecycle.js";

const appSource = await readFile(new URL("./App.jsx", import.meta.url), "utf8");

function makeStateSetters(state, resetValues) {
  return Object.fromEntries(
    Object.keys(resetValues).map((key) => [key, (value) => { state[key] = value; }]),
  );
}

test("tabular dataset reset clears derived workflow state and preserves shared upload/request", () => {
  const state = {
    ...TABULAR_DATASET_RESET,
    fileName: "sales.csv",
    uploadState: "success",
    task: "Predict sales",
    analyzeState: "success",
    analyzeError: "old analysis error",
    analyzeResult: { ready_for_training: true },
    trainingConfirmed: true,
    groupColumn: "region",
    trainingState: "success",
    trainingError: "old training error",
    trainingResult: { automl_training: { model: { model_id: "stale" } } },
    excludedFeatureColumns: ["legacy_feature"],
    diagnosticState: "success",
    diagnosticError: "old diagnostic error",
    diagnosticResult: { warnings: [] },
    diagnosticGroupColumn: "region",
    diagnosticRetry: 5,
    predictionState: "success",
    predictionError: "old prediction error",
    predictionMode: "new",
    predictionInputs: { feature: "old" },
    predictionResult: { predictions: [123] },
  };

  resetTabularDatasetWorkflow(makeStateSetters(state, TABULAR_DATASET_RESET));

  for (const [key, value] of Object.entries(TABULAR_DATASET_RESET)) {
    assert.deepEqual(state[key], value, `${key} should be reset`);
  }
  assert.equal(state.fileName, "sales.csv");
  assert.equal(state.uploadState, "success");
  assert.equal(state.task, "Predict sales");
});

test("image dataset reset removes the old training model and prediction state", () => {
  const state = {
    ...IMAGE_DATASET_RESET,
    imageTrainingState: "success",
    imageTrainingError: "old error",
    imageTrainingResult: { model_id: "old-model" },
    imageModelId: "old-model",
    imagePredictionFile: { name: "old.jpg" },
    imagePredictionState: "success",
    imagePredictionError: "old prediction error",
    imagePredictionResult: { prediction: { class: "cat" } },
    imageZipFile: { name: "old.zip" },
  };

  resetImageDatasetWorkflow(makeStateSetters(state, IMAGE_DATASET_RESET));

  for (const [key, value] of Object.entries(IMAGE_DATASET_RESET)) {
    assert.deepEqual(state[key], value, `${key} should be reset`);
  }
  assert.equal(state.imageZipFile.name, "old.zip");
});

test("upload readiness requires explicit successful dataset validation", () => {
  assert.equal(
    getUploadValidationError({ validation: { valid: true } }),
    null,
  );
  assert.match(
    getUploadValidationError({
      message: "Dataset validation failed.",
      validation: { valid: false, errors: ["Dataset is empty."] },
    }),
    /Dataset validation failed\. Dataset is empty\./,
  );
  assert.match(getUploadValidationError({}), /did not confirm/);
});

test("training outcome only retains a result for completed AutoML runs", () => {
  const successPayload = {
    ready_for_training: true,
    automl_training: { status: "success" },
  };
  const completedPayload = {
    ready_for_training: true,
    automl_training: { status: "Completed" },
  };
  const failedPayload = {
    ready_for_training: true,
    automl_training: { status: "failed", error: "All candidate models failed." },
  };

  assert.deepEqual(getNlpTrainingOutcome(successPayload), {
    successful: true,
    message: "",
    result: successPayload,
  });
  assert.equal(getNlpTrainingOutcome(completedPayload).successful, true);
  assert.deepEqual(getNlpTrainingOutcome(failedPayload), {
    successful: false,
    message: "All candidate models failed.",
    result: null,
  });
  assert.equal(
    getNlpTrainingOutcome({ ready_for_training: false }).successful,
    false,
  );
});

test("target-proxy diagnostic is only eligible in active tabular mode", () => {
  const context = {
    diagnosticApplicable: true,
    fileName: "sales.csv",
    task: "Predict sales",
  };

  assert.equal(shouldRunTargetProxyDiagnostic({ ...context, datasetType: "tabular" }), true);
  assert.equal(shouldRunTargetProxyDiagnostic({ ...context, datasetType: "image" }), false);
});

test("task understanding uses the request, resolved task, target rationale, and confidence", () => {
  assert.match(appSource, /Understanding Your Task/);
  assert.match(appSource, /task\.trim\(\) \|\| analyzeResult\.user_text/);
  assert.match(appSource, /analyzeResult\.dataset_resolution\?\.problem_type/);
  assert.match(appSource, /resolvedTargetColumn/);
  assert.match(appSource, /analyzeResult\.dataset_resolution\.reason/);
  assert.match(appSource, /analyzeResult\.intent\?\.confidence/);
  assert.match(appSource, /analyzeResult\.dataset_resolution\?\.confidence/);
});

test("dataset understanding uses only returned dataset and training metadata", () => {
  assert.match(appSource, /aria-label="Understanding Your Dataset"/);
  assert.match(appSource, /analysis\?\.dataset_info\?\.rows/);
  assert.match(appSource, /analysis\?\.dataset_info\?\.columns/);
  assert.match(appSource, /analysis\?\.quality\?\.missing_values/);
  assert.match(appSource, /analysis\.quality\.duplicate_rows/);
  assert.match(appSource, /Column Details/);
  assert.match(appSource, /analysis\?\.columns/);
  assert.match(appSource, /analysis\?\.data_types/);
  assert.match(appSource, /trainingPreprocessing\.raw_feature_names/);
  assert.match(appSource, /trainingPreprocessing\.removed_columns/);
  assert.match(appSource, /trainingPreprocessing\.imputed_columns/);
  assert.match(appSource, /trainingPreprocessing\.target_classes/);
  assert.match(appSource, /No semantic meaning is inferred/);
  assert.match(appSource, /Dataset snapshot pending clarification/);
  assert.match(appSource, /Column roles are not assigned until the task and target are resolved/);
});

test("training workflow explains supervised and clustering paths separately", () => {
  assert.match(appSource, /How Your Model Will Be Trained/);
  assert.match(appSource, /trainingData\.training_rows/);
  assert.match(appSource, /trainingData\.validation_rows/);
  assert.match(appSource, /trainingData\.test_rows/);
  assert.match(appSource, /Validation data is used to compare and select a model/);
  assert.match(appSource, /Clustering is unsupervised/);
  assert.match(appSource, /or create separate validation and test partitions/);
});

test("classification evaluation metrics render outside the prediction section", () => {
  const evaluationStart = appSource.indexOf("Classification Evaluation");
  const predictionStart = appSource.indexOf('aria-label="Prediction"');
  assert.ok(evaluationStart >= 0);
  assert.ok(predictionStart > evaluationStart);

  const evaluationMarkup = appSource.slice(evaluationStart, predictionStart);
  const predictionMarkup = appSource.slice(predictionStart);
  for (const label of [
    "Holdout accuracy",
    "Holdout weighted F1",
    "Balanced accuracy",
    "Macro precision",
    "Macro recall",
    "Macro F1",
    "Confusion matrix",
    "Per-class metrics",
  ]) {
    assert.ok(evaluationMarkup.includes(label), `Missing evaluation label: ${label}`);
  }
  assert.doesNotMatch(predictionMarkup, /Holdout accuracy|Balanced accuracy|Confusion matrix/);
  assert.match(predictionMarkup, /Predicted class/);
  assert.match(predictionMarkup, /Probabilities:/);
});

test("training form binds the optional group dropdown to uploaded columns", () => {
  assert.match(appSource, /const \[groupColumn, setGroupColumn\] = useState\(""\)/);
  assert.match(appSource, /getGroupColumnOptions\(\s*uploadedColumns/);
  assert.match(appSource, /<option value="">None<\/option>/);
  assert.match(appSource, /onChange=\{\(event\) => setGroupColumn\(event\.target\.value\)\}/);
  assert.match(appSource, /Group-aware splitting keeps records from the same entity together across train, validation, and test sets\./);
  const trainingHandler = appSource.slice(
    appSource.indexOf("async function confirmTraining()"),
    appSource.indexOf("function getPredictionFeatures()"),
  );
  assert.match(trainingHandler, /fetch\("\/nlp\/train"/);
  assert.match(trainingHandler, /JSON\.stringify\(\s*buildNlpTrainingRequest/);
  assert.match(trainingHandler, /groupColumn,/);
});

test("feature selection excludes target and group and is sent with training", () => {
  assert.match(appSource, /getFeatureColumnOptions\(\s*uploadedColumns,\s*resolvedTargetColumn,\s*groupColumn/);
  assert.match(appSource, /Features included in the model/);
  assert.match(appSource, /checked=\{!excludedFeatureColumns\.includes\(column\)\}/);
  assert.match(appSource, /selectedFeatureColumns\.length === 0/);
  assert.match(appSource, /Select at least one feature to train a model/);
  const trainingHandler = appSource.slice(
    appSource.indexOf("async function confirmTraining()"),
    appSource.indexOf("function getPredictionFeatures()"),
  );
  assert.match(trainingHandler, /featureColumns: selectedFeatureColumns/);
});

test("target-proxy diagnostic refreshes by group and protects training from stale results", () => {
  assert.match(appSource, /fetch\("\/nlp\/diagnose-target-proxies"/);
  assert.match(appSource, /diagnosticRetry,[\s\S]*fileName,[\s\S]*groupColumn,[\s\S]*task,/);
  assert.match(appSource, /tracker\.isLatest\(requestId\)/);
  assert.match(appSource, /controller\.abort\(\)/);
  assert.match(appSource, /diagnosticGroupColumn === groupColumn/);
  assert.match(appSource, /Statistical association does not prove leakage or prediction-time unavailability/);
  assert.match(appSource, /diagnosticApplicable && !diagnosticReady/);
  assert.match(appSource, /warning\.feature/);
  assert.match(appSource, /warning\.detection_type/);
  assert.match(appSource, /warning\.evidence/);
  assert.match(appSource, /warning\.evidence\?\.method/);
  assert.match(appSource, /warning\.evidence\?\.effect_size_name/);
  assert.match(appSource, /warning\.evidence\.q_value < 0\.001/);
  assert.match(appSource, /not repeated within each CV fold/);
  assert.match(appSource, /Grouped tests use one summary per group/);
  const trainingButton = appSource.slice(
    appSource.indexOf("disabled={\n                      trainingState"),
    appSource.indexOf("style={{", appSource.indexOf("disabled={\n                      trainingState")),
  );
  assert.doesNotMatch(trainingButton, /diagnosticResult\?\.warnings/);
});

test("automatic tuning is the default and manual choices are in Advanced settings", () => {
  assert.match(appSource, /const \[tuningStrategy, setTuningStrategy\] = useState\("auto"\)/);
  assert.match(appSource, /Automatic selection/);
  assert.match(appSource, /TextToAutoML selects a suitable tuning strategy and trial budget/);
  const advancedSettingsStart = appSource.indexOf('aria-label="Advanced Settings"');
  const advancedSettingsEnd = appSource.indexOf('aria-label="Advanced Diagnostics"');
  const advancedSettings = appSource.slice(
    advancedSettingsStart,
    advancedSettingsEnd,
  );
  assert.match(advancedSettings, /Advanced Settings/);
  assert.match(advancedSettings, /value="grid">Grid Search/);
  assert.match(advancedSettings, /value="bayesian">Bayesian Optimization \(Optuna TPE\)/);
  assert.doesNotMatch(advancedSettings, /value="random"|value="none"/);
  assert.match(advancedSettings, /tuningStrategy === "bayesian" && manualTuningStrategy === "bayesian"/);
  assert.match(advancedSettings, /min="1"/);
  assert.match(advancedSettings, /max="30"/);
  assert.match(advancedSettings, /Use automatic selection/);
  assert.match(advancedSettings, /groupColumn/);
  assert.match(advancedSettings, /selectedFeatureColumns\.length/);
});

test("diagnostic details stay collapsed under Advanced Diagnostics", () => {
  assert.match(appSource, /<details aria-label="Advanced Diagnostics"/);
  assert.match(appSource, /View detailed diagnostics/);
  assert.match(appSource, /exploratory checks for unusual relationships/);
  assert.match(appSource, /warning\.evidence\.q_value/);
  assert.match(appSource, /JSON\.stringify\(\{/);
});

test("automatic result display uses backend decision metadata and handles older responses", () => {
  assert.match(appSource, /getTuningDecision\(trainingResult\)/);
  assert.match(appSource, /Tuning strategy:/);
  assert.match(appSource, /Automatic tuning was not applied for this run/);
  assert.match(appSource, /tuningDecision\.trialBudget/);
  assert.match(appSource, /tuningDecision\.reason/);
  assert.match(appSource, /Automatic optimizer decision details are unavailable from this backend response/);
  assert.match(appSource, /tuningStrategyLabels\[tuningDecision\.strategy\]/);
  assert.doesNotMatch(appSource, /Automatic selection: Bayesian Optimization \(Optuna TPE\)/);
});

test("AI explanation is displayed separately from factual training results", () => {
  assert.match(appSource, /aria-label="AI-Generated Explanation"/);
  assert.match(appSource, /trainingResult\.explanation/);
  assert.match(appSource, /This explanation is based on the actual training results returned by TextToAutoML/);
  assert.match(appSource, /whiteSpace: "pre-wrap"/);
  assert.doesNotMatch(appSource, /trainingResult\.report\?\.summary/);
  assert.match(appSource, /evaluationSummary\.metrics/);
});

test("training report separates model selection score from final holdout results", () => {
  assert.match(appSource, /aria-label="Model Training Report"/);
  assert.match(appSource, /bestModel\.validation_metrics\?\./);
  assert.match(appSource, /Model selection score \(validation\)/);
  assert.match(appSource, /Final holdout\/test performance/);
  assert.match(appSource, /Candidate-by-candidate validation scores were not included/);
  assert.match(appSource, /evaluationResults\.map/);
  assert.match(appSource, /bestModel\.reason/);
  assert.match(appSource, /automlTraining\.training_time_seconds/);
  assert.match(appSource, /\["accuracy", "f1_score", "balanced_accuracy"\]\.includes\(selectionMetric\)/);
  assert.match(appSource, /automlTraining\.validation_diagnostics/);
  assert.match(appSource, /Validation results by candidate/);
  assert.match(appSource, /Training metrics/);
  assert.match(appSource, /Validation metrics/);
  assert.match(appSource, /Final holdout\/test metrics are reported separately below/);
  assert.match(appSource, /candidate\.training_metrics/);
  assert.match(appSource, /candidate\.validation_metrics/);
  assert.match(appSource, /diagnosticMetricSummary/);
  assert.match(appSource, /Not available/);
  assert.match(appSource, /candidate\.balanced_accuracy/);
  assert.match(appSource, /per_class_metrics/);
  assert.match(appSource, /candidate\.error_reason/);
  assert.match(appSource, /validationDiagnostics\.length === 0 && evaluationResults\.length < candidates\.length/);
});

test("evaluation help is task-specific and per-class summary comes from returned F1 values", () => {
  assert.match(appSource, /Weighted F1 balances precision and recall/);
  assert.match(appSource, /Balanced accuracy averages class recall equally/);
  assert.match(appSource, /Macro precision, macro recall, and macro F1 average/);
  assert.match(appSource, /RMSE and MAE measure prediction error in target units/);
  assert.match(appSource, /Silhouette score summarizes how well-separated the clusters are/);
  assert.match(appSource, /Diagonal values are correct predictions/);
  assert.match(appSource, /Values outside the diagonal show where the model confused/);
  assert.match(appSource, /summarizePerClassF1\(evaluationSummary\.perClassMetrics\)/);
  assert.match(appSource, /Highest F1:/);
  assert.match(appSource, /Lowest F1:/);
});

test("prediction result names the model and avoids causal feature claims", () => {
  assert.match(appSource, /Model used/);
  assert.match(appSource, /predictionResult\.model_name \|\| trainingResult\.automl_training\?\.model\?\.name/);
  assert.match(appSource, /The model's output is based on the input values provided/);
});

test("optimizer settings are sent only with training and preserve training states", () => {
  const trainingHandler = appSource.slice(
    appSource.indexOf("async function confirmTraining()"),
    appSource.indexOf("function getPredictionFeatures()"),
  );
  assert.match(trainingHandler, /setTrainingState\("loading"\)/);
  assert.match(trainingHandler, /setTrainingState\("success"\)/);
  assert.match(trainingHandler, /setTrainingState\("error"\)/);
  assert.match(trainingHandler, /tuningStrategy,/);
  assert.match(trainingHandler, /tuningTrials:/);

  const predictionHandler = appSource.slice(
    appSource.indexOf("async function handlePredict()"),
    appSource.indexOf("\n  return (", appSource.indexOf("async function handlePredict()")),
  );
  assert.doesNotMatch(predictionHandler, /tuningStrategy|tuningTrials|tuning_strategy|tuning_trials/);
});

test("dataset type selector defaults to tabular and exposes both choices", () => {
  assert.match(appSource, /const \[datasetType, setDatasetType\] = useState\("tabular"\)/);
  assert.match(appSource, /What type of dataset do you have\?/);
  assert.match(appSource, /title: "Tabular Dataset", detail: "CSV \/ XLSX files"/);
  assert.match(appSource, /title: "Image Dataset", detail: "ZIP containing folders of images"/);
  assert.match(appSource, /aria-pressed=\{datasetType === option\.value\}/);
  assert.match(appSource, /datasetType === "tabular" \?/);
});

test("tabular choice retains CSV and XLSX upload without image endpoint calls", () => {
  const tabularWorkflow = appSource.slice(
    appSource.indexOf('<div data-workflow="tabular">'),
    appSource.indexOf("</div>\n        ) : (", appSource.indexOf('<div data-workflow="tabular">')),
  );
  const tabularUploadHandler = appSource.slice(
    appSource.indexOf("async function handleFileChange(event)"),
    appSource.indexOf("async function handleImageDatasetChange(event)"),
  );

  assert.match(tabularWorkflow, /accept="\.csv,\.xlsx"/);
  assert.match(tabularWorkflow, /handleFileChange/);
  assert.match(tabularWorkflow, /handleAnalyze/);
  assert.match(tabularWorkflow, /handlePredict/);
  assert.match(tabularUploadHandler, /fetch\("\/upload\/"/);
  assert.doesNotMatch(tabularUploadHandler, /\/image\//);
});

test("image ZIP selection stores the file without starting training", () => {
  const imageDatasetHandler = appSource.slice(
    appSource.indexOf("async function handleImageDatasetChange(event)"),
    appSource.indexOf("function clearImageRouting()"),
  );

  assert.match(appSource, /Add an image dataset ZIP for training/);
  assert.match(appSource, /accept="\.zip,application\/zip"/);
  assert.match(appSource, /cats\//);
  assert.match(appSource, /dogs\//);
  assert.match(imageDatasetHandler, /setImageZipFile\(file\)/);
  assert.doesNotMatch(imageDatasetHandler, /trainImageModel|predictImageModel/);
  assert.match(appSource, /Selecting a ZIP will not start training/);
});

test("image requests route through the shared endpoint with current context", () => {
  const routingHandler = appSource.slice(
    appSource.indexOf("async function handleImageRoute()"),
    appSource.indexOf("async function startImageTraining()"),
  );

  assert.match(appSource, /value=\{task\}[\s\S]*onChange=\{handleImageRequestChange\}/);
  assert.match(routingHandler, /routeImageRequest\(\{/);
  assert.match(routingHandler, /asset_type: assetType/);
  assert.match(routingHandler, /image_dataset_available: Boolean\(imageZipFile\)/);
  assert.match(routingHandler, /image_available: Boolean\(imagePredictionFile\)/);
  assert.match(routingHandler, /image_model_id: imageModelId \|\| null/);
  assert.match(routingHandler, /image_training_result_available: Boolean\(imageTrainingResult\)/);
  assert.doesNotMatch(routingHandler, /trainImageModel|predictImageModel/);
  assert.match(appSource, /aria-label="Image routing result"/);
  assert.match(appSource, /imageIntentLabels\[imageRoutingResult\.intent\]/);
  assert.match(appSource, /imageRoutingResult\.reason/);
  assert.match(appSource, /imageRoutingResult\.required_context/);
});

test("training begins only from a fresh ready training decision and explicit button", () => {
  const trainingHandler = appSource.slice(
    appSource.indexOf("async function startImageTraining()"),
    appSource.indexOf("async function handleImagePredict()"),
  );

  assert.match(trainingHandler, /imageRoutingResult\?\.intent !== "IMAGE_TRAINING_REQUEST"/);
  assert.match(trainingHandler, /!imageRoutingResult\.ready/);
  assert.match(trainingHandler, /imageRoutedText !== task\.trim\(\)/);
  assert.match(trainingHandler, /!imageZipFile/);
  assert.match(trainingHandler, /trainImageModel\(imageZipFile\)/);
  assert.match(appSource, /onClick=\{startImageTraining\}/);
  assert.match(appSource, /Start Image Training/);
});

test("deferred and ambiguous image routes expose no training or prediction action", () => {
  const routingHandler = appSource.slice(
    appSource.indexOf("async function handleImageRoute()"),
    appSource.indexOf("async function startImageTraining()"),
  );
  const actionMarkup = appSource.slice(
    appSource.indexOf('aria-label="Image routing result"'),
    appSource.indexOf('{imageTrainingState === "error"', appSource.indexOf('aria-label="Image routing result"')),
  );

  assert.doesNotMatch(routingHandler, /trainImageModel|predictImageModel/);
  assert.match(actionMarkup, /intent === "IMAGE_TRAINING_REQUEST" && imageRoutingResult\.ready/);
  assert.match(actionMarkup, /intent === "IMAGE_PREDICTION" && imageRoutingResult\.ready/);
  assert.doesNotMatch(actionMarkup, /IMAGE_DATASET_ANALYSIS[\s\S]*trainImageModel/);
  assert.doesNotMatch(actionMarkup, /IMAGE_MODEL_EXPLANATION[\s\S]*predictImageModel/);
  assert.match(appSource, /UNKNOWN_AMBIGUOUS: "Needs Clarification"/);
});

test("image prediction requires a routed intent, model ID, and selected single image", () => {
  const predictionHandler = appSource.slice(
    appSource.indexOf("async function handleImagePredict()"),
    appSource.indexOf("function useExample()"),
  );

  assert.match(appSource, /id="image-prediction-upload"/);
  assert.match(appSource, /onClick=\{handleImagePredict\}/);
  assert.match(predictionHandler, /imageRoutingResult\?\.intent !== "IMAGE_PREDICTION"/);
  assert.match(predictionHandler, /!imageRoutingResult\.ready/);
  assert.match(predictionHandler, /!imageModelId \|\| !imagePredictionFile/);
  assert.match(predictionHandler, /predictImageModel\(imageModelId, imagePredictionFile\)/);
});

test("image model ID state is reused for routing and prediction", () => {
  assert.equal((appSource.match(/const \[imageModelId, setImageModelId\]/g) || []).length, 1);
  assert.match(appSource, /image_model_id: imageModelId \|\| null/);
  assert.match(appSource, /setImageModelId\(payload\.model_id \|\| ""\)/);
});

test("routing errors are rendered as alerts without parsing assumptions", () => {
  assert.match(appSource, /imageRoutingError/);
  assert.match(appSource, /imageRoutingState === "error"/);
  assert.match(appSource, /routeImageRequest/);
  assert.match(appSource, /error\.message \|\| "Image request routing failed\. Please try again\."/);
});

test("IMAGE_DATASET_ANALYSIS calls the analysis endpoint only after routing and ZIP validation", () => {
  const routeHandler = appSource.slice(
    appSource.indexOf("async function handleImageRoute()"),
    appSource.indexOf("async function startImageTraining()"),
  );
  const missingZipCheck = routeHandler.indexOf("if (!imageZipFile)");
  const analysisCall = routeHandler.indexOf("analyzeImageDataset(imageZipFile)");

  assert.match(routeHandler, /decision\.intent === "IMAGE_DATASET_ANALYSIS"/);
  assert.ok(missingZipCheck >= 0 && missingZipCheck < analysisCall);
  assert.match(routeHandler, /setImageAnalysisError\("Select an image dataset ZIP before analyzing it\."\)/);
  assert.match(routeHandler, /setImageAnalysisLoading\(true\)/);
  assert.match(routeHandler, /finally[\s\S]*setImageAnalysisLoading\(false\)/);
  assert.doesNotMatch(routeHandler, /trainImageModel|predictImageModel/);
});

test("successful image analysis displays backend total, classes, and percentages", () => {
  assert.match(appSource, /aria-label="Image Dataset Analysis"/);
  assert.match(appSource, /Total Images/);
  assert.match(appSource, /imageAnalysisResult\.dataset\?\.image_count/);
  assert.match(appSource, /Number of Classes/);
  assert.match(appSource, /imageAnalysisResult\.dataset\?\.class_count/);
  assert.match(appSource, /Class Distribution/);
  assert.match(appSource, /imageAnalysisResult\.classes\?\.images_per_class\?\.\[className\]/);
  assert.match(appSource, /distribution\?\.percentage/);
  assert.doesNotMatch(appSource, /percentage\s*=.*image_count/);
});

test("provided split counts render while unsplit datasets show no fabricated counts", () => {
  assert.match(appSource, /splits\?\.source === "provided"/);
  assert.match(appSource, /splits\?\.source === "not_provided"/);
  assert.match(appSource, /No predefined train\/validation\/test split was provided\./);
  assert.match(appSource, /imageAnalysisResult\.splits\?\.counts\?\.\[splitName\]/);
  assert.match(appSource, /imageAnalysisResult\.splits\?\.per_class_counts/);
  assert.doesNotMatch(appSource, /split_dataset\(/);
});

test("image properties, actual validation findings, and unassessed suitability are displayed", () => {
  assert.match(appSource, /Image Properties/);
  assert.match(appSource, /imageAnalysisProperties\.width\?\.min/);
  assert.match(appSource, /imageAnalysisProperties\.height\?\.average/);
  assert.match(appSource, /imageAnalysisProperties\.aspect_ratio\?\.max/);
  assert.match(appSource, /File Extensions/);
  assert.match(appSource, /Decoded Formats/);
  assert.match(appSource, /Source Image Modes/);
  assert.match(appSource, /Source Channel Counts/);
  assert.match(appSource, /imageAnalysisResult\.quality\?\.validation/);
  assert.match(appSource, /imageAnalysisFindings\.map/);
  assert.match(appSource, /status === "not_assessed"[\s\S]*Not assessed yet/);
  assert.doesNotMatch(appSource, /blur score|brightness score|duplicate count|leakage score|noise score/i);
});

test("quality metrics render backend values without recomputing them", () => {
  const analysisReport = appSource.slice(
    appSource.indexOf('aria-label="Image Dataset Analysis"'),
    appSource.indexOf('{imageTrainingState === "error"', appSource.indexOf('aria-label="Image Dataset Analysis"')),
  );

  assert.match(analysisReport, /Quality Metrics/);
  assert.match(analysisReport, /dimension_consistency\?\.consistent/);
  assert.match(analysisReport, /dimension_consistency\?\.unique_dimensions/);
  assert.match(analysisReport, /resolution\?\.min_pixel_area/);
  assert.match(analysisReport, /resolution\?\.max_pixel_area/);
  assert.match(analysisReport, /resolution\?\.average_pixel_area/);
  assert.match(analysisReport, /resolution\?\.median_pixel_area/);
  assert.match(analysisReport, /class_size\?\.minimum_images_per_class/);
  assert.match(analysisReport, /class_balance\?\.imbalance_ratio/);
  assert.match(analysisReport, /typeof imageQualityMetrics\.dimension_consistency\?\.consistent === "boolean"/);
  assert.doesNotMatch(analysisReport, /largest_class_count\s*\/\s*smallest_class_count/);
});

test("exact duplicate report explains SHA-256 bytes and displays bounded group samples", () => {
  const analysisReport = appSource.slice(
    appSource.indexOf('aria-label="Image Dataset Analysis"'),
    appSource.indexOf('{imageTrainingState === "error"', appSource.indexOf('aria-label="Image Dataset Analysis"')),
  );

  assert.match(analysisReport, /Exact Duplicate Check/);
  assert.match(analysisReport, /exact byte-for-byte matches using SHA-256/);
  assert.match(analysisReport, /Re-encoded copies may not match/);
  assert.match(analysisReport, /duplicate_group_count/);
  assert.match(analysisReport, /duplicate_file_count/);
  assert.match(analysisReport, /imageDuplicateGroups\.map/);
  assert.match(analysisReport, /group\.files\.map/);
  assert.match(analysisReport, /group\.files_truncated === true/);
  assert.match(analysisReport, /groups_truncated === true/);
  assert.match(analysisReport, /No exact duplicates were detected\./);
  assert.match(analysisReport, /Duplicate group details not available\./);
});

test("cross-split duplicate report displays provided overlap and backend match details", () => {
  const analysisReport = appSource.slice(
    appSource.indexOf('aria-label="Image Dataset Analysis"'),
    appSource.indexOf('{imageTrainingState === "error"', appSource.indexOf('aria-label="Image Dataset Analysis"')),
  );

  assert.match(analysisReport, /Cross-Split Duplicate Check/);
  assert.match(appSource, /imageQualityMetrics\.cross_split_duplicates/);
  assert.match(analysisReport, /imageCrossSplitMetrics\.checked === true/);
  assert.match(analysisReport, /imageCrossSplitMetrics\.match_count/);
  assert.match(analysisReport, /imageCrossSplitMatches\.map/);
  assert.match(analysisReport, /match\.splits\.join/);
  assert.match(analysisReport, /match\.file_count/);
  assert.match(analysisReport, /This is a possible exact-duplicate split overlap, not confirmation of model leakage\./);
  assert.match(analysisReport, /matches_truncated === true/);
});

test("unsplit dataset says cross-split checking is not applicable", () => {
  assert.match(appSource, /splits\?\.source === "not_provided" \|\| imageCrossSplitMetrics\.checked === false/);
  assert.match(appSource, /Not applicable \/ not checked because the dataset has no predefined splits\./);
});

test("missing quality metrics display as unavailable rather than zero", () => {
  const analysisReport = appSource.slice(
    appSource.indexOf('aria-label="Image Dataset Analysis"'),
    appSource.indexOf('{imageTrainingState === "error"', appSource.indexOf('aria-label="Image Dataset Analysis"')),
  );

  assert.match(appSource, /function displayAnalysisNumber\(value/);
  assert.match(appSource, /typeof value !== "number" \|\| !Number\.isFinite\(value\)/);
  assert.match(analysisReport, /displayAnalysisNumber\(imageQualityMetrics\.resolution\?\.median_pixel_area\)/);
  assert.match(analysisReport, /displayAnalysisNumber\(imageDuplicateMetrics\.duplicate_group_count, 0\)/);
  assert.match(analysisReport, /displayAnalysisNumber\(imageCrossSplitMetrics\.match_count, 0\)/);
  assert.match(analysisReport, /typeof imageQualityMetrics\.dimension_consistency\?\.consistent === "boolean"/);
  assert.match(analysisReport, /: "Not available"\}/);
});

test("quality report preserves overview, properties, validation findings, warnings, and suitability sections", () => {
  const analysisReport = appSource.slice(
    appSource.indexOf('aria-label="Image Dataset Analysis"'),
    appSource.indexOf('{imageTrainingState === "error"', appSource.indexOf('aria-label="Image Dataset Analysis"')),
  );

  for (const label of [
    "Dataset Overview",
    "Class Distribution",
    "Dataset Splits",
    "Image Properties",
    "Quality Metrics",
    "Validation",
    "CNN Suitability",
  ]) {
    assert.ok(analysisReport.includes(label), `Missing existing analysis section: ${label}`);
  }
  assert.match(analysisReport, /imageAnalysisFindings\.map/);
  assert.match(analysisReport, /imageAnalysisResult\.cnn_suitability\?\.status/);
});

test("null analysis values display as unavailable and never become zero", () => {
  assert.match(appSource, /function displayAnalysisNumber\(value/);
  assert.match(appSource, /typeof value !== "number" \|\| !Number\.isFinite\(value\)/);
  assert.match(appSource, /return "Not available"/);
  assert.match(appSource, /displayAnalysisBytes\(imageAnalysisResult\.dataset\?\.archive_size_bytes\)/);
  assert.match(appSource, /displayAnalysisBytes\(imageAnalysisResult\.dataset\?\.extracted_size_bytes\)/);
  assert.match(appSource, /displayAnalysisNumber\(imageAnalysisResult\.splits\?\.counts\?\.\[splitName\], 0\)/);
});

test("analysis errors render and stale results clear on ZIP, request, and mode changes", () => {
  assert.match(appSource, /imageAnalysisError/);
  assert.match(appSource, /imageAnalysisLoading &&/);
  assert.match(appSource, /catch \(error\)[\s\S]*setImageAnalysisError/);
  assert.match(appSource, /function clearImageAnalysis\(\)/);
  assert.match(appSource, /setImageAnalysisResult\(null\)/);
  assert.match(appSource, /setImageAnalysisLoading\(false\)/);

  const zipHandler = appSource.slice(
    appSource.indexOf("async function handleImageDatasetChange(event)"),
    appSource.indexOf("function clearImageAnalysis()"),
  );
  const requestHandler = appSource.slice(
    appSource.indexOf("function handleImageRequestChange(event)"),
    appSource.indexOf("async function handleImageRoute()"),
  );
  const modeHandler = appSource.slice(
    appSource.indexOf("function selectDatasetType(type)"),
    appSource.indexOf("async function handleFileChange(event)"),
  );
  assert.match(zipHandler, /clearImageAnalysis\(\)/);
  assert.match(requestHandler, /clearImageAnalysis\(\)/);
  assert.match(modeHandler, /clearImageAnalysis\(\)/);
});

test("image analysis route cannot invoke training or prediction", () => {
  const analysisPanel = appSource.slice(
    appSource.indexOf('aria-label="Image Dataset Analysis"'),
    appSource.indexOf('{imageTrainingState === "error"', appSource.indexOf('aria-label="Image Dataset Analysis"')),
  );
  assert.doesNotMatch(analysisPanel, /trainImageModel|predictImageModel|startImageTraining|handleImagePredict/);
  assert.match(appSource, /analyzeImageDataset\(imageZipFile\)/);
});

test("image training results render only returned fields", () => {
  assert.match(appSource, /aria-label="Image training result"/);
  for (const field of [
    "imageTrainingResult.task",
    "imageTrainingResult.model",
    "imageTrainingResult.classes",
    "imageTrainingResult.metrics?.accuracy",
    "imageTrainingResult.samples?.train",
    "imageTrainingResult.samples?.validation",
    "imageTrainingResult.samples?.test",
    "imageTrainingResult.training_time_seconds",
    "imageTrainingResult.model_id",
  ]) {
    assert.ok(appSource.includes(field), `Missing image training field: ${field}`);
  }
});

test("image prediction uses the image model and displays factual results", () => {
  const imagePredictionHandler = appSource.slice(
    appSource.indexOf("async function handleImagePredict()"),
    appSource.indexOf("function useExample()"),
  );

  assert.match(appSource, /accept="\.jpg,\.jpeg,\.png,\.bmp,\.webp"/);
  assert.match(imagePredictionHandler, /predictImageModel\(imageModelId, imagePredictionFile\)/);
  assert.doesNotMatch(imagePredictionHandler, /fetch\("\/predict\//);
  assert.match(appSource, /imagePredictionResult\.prediction\?\.class/);
  assert.match(appSource, /imagePredictionResult\.prediction\?\.confidence/);
  assert.match(appSource, /Object\.entries\(imagePredictionResult\.probabilities\)/);
  assert.match(appSource, /imagePredictionResult\.explanation/);
});