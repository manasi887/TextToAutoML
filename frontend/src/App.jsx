import { useState } from "react";

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

export default function App() {
  const [fileName, setFileName] = useState("");
  const [task, setTask] = useState("");
  const [uploadState, setUploadState] = useState("idle");
  const [uploadError, setUploadError] = useState("");
  const [uploadResult, setUploadResult] = useState(null);
  const [analyzeState, setAnalyzeState] = useState("idle");
  const [analyzeError, setAnalyzeError] = useState("");
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [trainingConfirmed, setTrainingConfirmed] = useState(false);
  const [trainingState, setTrainingState] = useState("idle");
  const [trainingError, setTrainingError] = useState("");
  const [trainingResult, setTrainingResult] = useState(null);
  const [predictionState, setPredictionState] = useState("idle");
  const [predictionError, setPredictionError] = useState("");
  const [predictionMode, setPredictionMode] = useState("sample");
  const [predictionInputs, setPredictionInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setFileName(file.name);
    setUploadState("loading");
    setUploadError("");
    setUploadResult(null);
    setPredictionState("idle");
    setPredictionError("");
    setPredictionMode("sample");
    setPredictionInputs({});
    setPredictionResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/upload/", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || payload.message || "Upload failed.");
      }

      setUploadResult(payload);
      setUploadState("success");
    } catch (error) {
      setUploadState("error");
      setUploadError(error.message || "Upload failed. Please try again.");
    }
  }

  function useExample() {
    setTask("Predict customer churn");
  }

  async function handleAnalyze() {
    if (!fileName || !task.trim()) {
      setAnalyzeState("error");
      setAnalyzeError("Choose a dataset and describe the task before analyzing.");
      setAnalyzeResult(null);
      setTrainingConfirmed(false);
      setTrainingState("idle");
      setTrainingError("");
      setTrainingResult(null);
      return;
    }

    setAnalyzeState("loading");
    setAnalyzeError("");
    setAnalyzeResult(null);
    setTrainingConfirmed(false);
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
      setAnalyzeState("error");
      setAnalyzeError(error.message || "Analysis failed. Please try again.");
    }
  }

  async function confirmTraining() {
    setTrainingState("loading");
    setTrainingError("");
    setTrainingResult(null);

    try {
      const response = await fetch("/nlp/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filename: fileName, text: task.trim() }),
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || "Training failed.");
      }

      setTrainingResult(payload);
      setTrainingState("success");
      setTrainingConfirmed(true);
      setPredictionState("idle");
      setPredictionError("");
      setPredictionMode("sample");
      setPredictionResult(null);
      const rawFeatureNames = payload.automl_training?.preprocessing?.raw_feature_names || [];
      setPredictionInputs(
        Object.fromEntries(rawFeatureNames.map((column) => [column, ""]))
      );
    } catch (error) {
      setTrainingState("error");
      setTrainingError(error.message || "Training failed. Please try again.");
    }
  }

  function getPredictionFeatures() {
    return trainingResult?.automl_training?.preprocessing?.raw_feature_names || [];
  }

  function getFeatureType(column) {
    return uploadResult?.analysis?.data_types?.[column] || "object";
  }

  function updatePredictionInput(column, value) {
    setPredictionInputs((current) => ({ ...current, [column]: value }));
  }

  function buildPredictionRecord() {
    const features = getPredictionFeatures();
    const record = {};
    const targetColumn = trainingResult?.automl_training?.target_column || "";

    if (predictionMode === "sample") {
      const sample = uploadResult?.sample_rows?.[0] || {};
      for (const column of features) {
        if (column !== targetColumn && Object.prototype.hasOwnProperty.call(sample, column)) {
          record[column] = sample[column];
        }
      }
      return record;
    }

    for (const column of features) {
      const value = predictionInputs[column];
      if (value === undefined || value === "") {
        continue;
      }

      const dtype = String(getFeatureType(column)).toLowerCase();
      if (dtype.includes("bool")) {
        const normalized = String(value).trim().toLowerCase();
        if (!["true", "false"].includes(normalized)) {
          throw new Error(`'${column}' must be true or false.`);
        }
        record[column] = normalized === "true";
      } else if (
        dtype.includes("int") ||
        dtype.includes("float") ||
        dtype.includes("double") ||
        dtype.includes("number")
      ) {
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

  async function handlePredict() {
    const modelId = trainingResult?.automl_training?.model?.model_id || trainingResult?.report?.model_id;
    if (!modelId) {
      setPredictionState("error");
      setPredictionError("No saved model ID was returned by training.");
      return;
    }

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

      if (!response.ok) {
        throw new Error(payload.detail || "Prediction failed.");
      }

      setPredictionResult(payload);
      setPredictionState("success");
    } catch (error) {
      setPredictionState("error");
      setPredictionError(error.message || "Prediction failed. Please try again.");
    }
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
              onChange={(event) => setTask(event.target.value)}
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
            disabled={analyzeState === "loading"}
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
              Results
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
              {analyzeResult?.needs_clarification
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
              </div>
            ) : analyzeState === "success" && analyzeResult ? (
              <div style={{ font: "14px/1.5 Arial, sans-serif" }}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                    gap: "14px 20px",
                    paddingTop: "18px",
                    borderTop: "1px solid #edf2ee",
                  }}
                >
                  <div>
                  <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>
                    Intent
                  </span>
                  <strong>{analyzeResult.intent?.intent || "-"}</strong>
                  </div>
                  <div>
                  <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>
                    Problem type
                  </span>
                  <strong>
                    {analyzeResult.dataset_resolution?.problem_type || analyzeResult.task?.problem_type || "-"}
                  </strong>
                  </div>
                  <div>
                  <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>
                    Target
                  </span>
                  <strong>
                    {analyzeResult.dataset_resolution?.target_column || analyzeResult.target?.matched_column || "None"}
                  </strong>
                  </div>
                  <div>
                  <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>
                    Confidence
                  </span>
                  <strong>
                    {analyzeResult.intent?.confidence != null
                      ? `${Math.round(analyzeResult.intent.confidence * 100)}%`
                      : "-"}
                  </strong>
                  </div>
                  <div>
                  <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>
                    Clarification
                  </span>
                  <strong>
                    {analyzeResult.needs_clarification || analyzeResult.dataset_resolution?.needs_clarification
                      ? "Needed"
                      : "Not needed"}
                  </strong>
                  </div>
                {analyzeResult.training_time_estimate && (
                  <div>
                    <span style={{ display: "block", color: "#8a988e", fontSize: "12px" }}>
                      Estimated training time
                    </span>
                    <strong>
                      {formatDuration(analyzeResult.training_time_estimate.min_seconds)}
                      {" - "}
                      {formatDuration(analyzeResult.training_time_estimate.max_seconds)}
                    </strong>
                  </div>
                )}
                </div>
                <div style={{ marginTop: "22px" }}>
                  <button
                    type="button"
                    onClick={confirmTraining}
                    disabled={trainingState === "loading" || trainingConfirmed}
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
                  {trainingState === "success" && trainingResult && (
                    <p style={{ margin: "10px 0 0", color: "#1e6041" }}>
                      {trainingResult.report?.summary ||
                        trainingResult.automl_training?.status ||
                        "Training completed successfully."}
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
                Make a prediction
              </h2>
              <div style={{ display: "flex", gap: "8px", marginTop: "16px", flexWrap: "wrap" }}>
                {[{ value: "sample", label: "Predict Sample" }, { value: "new", label: "New Customer" }].map((option) => (
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
                  : "All fields are optional."}
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
                    <span style={{ color: "#66756b", fontSize: "12px" }}>Prediction</span>
                    <div style={{ marginTop: "4px", fontSize: "28px", fontWeight: 600, color: "#1e6041" }}>
                      {String(predictionResult.prediction_labels?.[0] ?? predictionResult.predictions?.[0] ?? "-")}
                    </div>
                    {predictionResult.probabilities?.[0] && (
                      <div style={{ marginTop: "10px", color: "#526158" }}>
                        Probabilities: {predictionResult.probabilities[0].map((value, index) => (
                          <span key={index} style={{ marginRight: "12px" }}>
                            {predictionResult.probability_labels?.[index] || `Class ${index}`}: {(Number(value) * 100).toFixed(1)}%
                          </span>
                        ))}
                      </div>
                    )}
                    <div style={{ marginTop: "10px", color: "#8a988e", fontSize: "12px" }}>
                      {predictionResult.model_name || "Saved model"} · {predictionResult.model_id || ""}
                    </div>
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
      </main>
    </div>
  );
}
