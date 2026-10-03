import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("./App.jsx", import.meta.url), "utf8");

function valueMarkupForLabel(label) {
  const start = appSource.indexOf(label);
  assert.notEqual(start, -1, `Missing Analyze Results label: ${label}`);
  const labelEnd = appSource.indexOf("</span>", start);
  const valueEnd = appSource.indexOf("</div>", labelEnd);
  return appSource.slice(labelEnd, valueEnd);
}

test("Analyze Results distinguishes raw classifier values from the resolved task", () => {
  const classifierPrediction = valueMarkupForLabel("Classifier prediction");
  const classifierConfidence = valueMarkupForLabel("Classifier confidence");
  const resolvedTask = valueMarkupForLabel("Resolved task");

  assert.match(classifierPrediction, /analyzeResult\.intent\?\.intent/);
  assert.match(classifierConfidence, /analyzeResult\.intent\?\.confidence/);
  assert.match(classifierConfidence, /Math\.round\(analyzeResult\.intent\.confidence \* 100\)/);
  assert.match(resolvedTask, /analyzeResult\.dataset_resolution\?\.problem_type/);
  assert.doesNotMatch(resolvedTask, /analyzeResult\.task\?\.problem_type/);
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

test("target-proxy diagnostic refreshes by group and protects training from stale results", () => {
  assert.match(appSource, /fetch\("\/nlp\/diagnose-target-proxies"/);
  assert.match(appSource, /diagnosticRetry,[\s\S]*fileName,[\s\S]*groupColumn,[\s\S]*task,/);
  assert.match(appSource, /tracker\.isLatest\(requestId\)/);
  assert.match(appSource, /controller\.abort\(\)/);
  assert.match(appSource, /diagnosticGroupColumn === groupColumn/);
  assert.match(appSource, /They do not prove leakage or that a feature is unavailable at prediction time/);
  assert.match(appSource, /diagnosticApplicable && !diagnosticReady/);
  assert.match(appSource, /warning\.feature/);
  assert.match(appSource, /warning\.detection_type/);
  assert.match(appSource, /warning\.evidence/);
});