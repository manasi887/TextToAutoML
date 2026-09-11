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

export default function App() {
  const [fileName, setFileName] = useState("");
  const [task, setTask] = useState("");
  const [uploadState, setUploadState] = useState("idle");
  const [uploadError, setUploadError] = useState("");
  const [uploadResult, setUploadResult] = useState(null);
  const [analyzeState, setAnalyzeState] = useState("idle");
  const [analyzeError, setAnalyzeError] = useState("");
  const [analyzeResult, setAnalyzeResult] = useState(null);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setFileName(file.name);
    setUploadState("loading");
    setUploadError("");
    setUploadResult(null);

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
      return;
    }

    setAnalyzeState("loading");
    setAnalyzeError("");
    setAnalyzeResult(null);

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

      setAnalyzeResult(payload);
      setAnalyzeState("success");
    } catch (error) {
      setAnalyzeState("error");
      setAnalyzeError(error.message || "Analysis failed. Please try again.");
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
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                  gap: "14px 20px",
                  paddingTop: "18px",
                  borderTop: "1px solid #edf2ee",
                  font: "14px/1.5 Arial, sans-serif",
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
      </main>
    </div>
  );
}
