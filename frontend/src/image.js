async function readResponse(response, fallbackMessage) {
  const responseText = await response.text();
  let payload = null;
  let invalidJson = false;

  if (responseText.trim()) {
    try {
      payload = JSON.parse(responseText);
    } catch {
      invalidJson = true;
    }
  }

  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : detail?.message || payload?.message || `${fallbackMessage} (HTTP ${response.status}).`,
    );
  }

  if (!responseText.trim() || invalidJson) {
    throw new Error(`${fallbackMessage} The server returned an empty or invalid response.`);
  }

  return payload;
}

const IMAGE_ROUTE_INTENTS = new Set([
  "IMAGE_TRAINING_REQUEST",
  "IMAGE_PREDICTION",
  "IMAGE_TRAINING_REPORT",
  "IMAGE_DATASET_ANALYSIS",
  "IMAGE_MODEL_EXPLANATION",
  "UNKNOWN_AMBIGUOUS",
]);

export async function routeImageRequest(request, fetchImpl = globalThis.fetch) {
  const response = await fetchImpl("/nlp/route", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  const payload = await readResponse(response, "Image request routing failed.");
  const validDecision =
    payload !== null &&
    typeof payload === "object" &&
    !Array.isArray(payload) &&
    ["image", "tabular", "unknown"].includes(payload.modality) &&
    IMAGE_ROUTE_INTENTS.has(payload.intent) &&
    typeof payload.ready === "boolean" &&
    typeof payload.clarification_required === "boolean" &&
    typeof payload.reason === "string" &&
    Array.isArray(payload.required_context) &&
    payload.required_context.every((item) => typeof item === "string");

  if (!validDecision) {
    throw new Error("Image request routing failed. The server returned an invalid routing response.");
  }
  return payload;
}

export async function trainImageModel(file, fetchImpl = globalThis.fetch) {
  const formData = new FormData();
  formData.append("file", file, file.name);
  const response = await fetchImpl("/image/train", {
    method: "POST",
    body: formData,
  });
  return readResponse(response, "Image training failed.");
}

export async function analyzeImageDataset(file, fetchImpl = globalThis.fetch) {
  if (!file) {
    throw new Error("Choose an image dataset ZIP before analyzing.");
  }
  const formData = new FormData();
  formData.append("file", file, file.name);
  const response = await fetchImpl("/image/analyze", {
    method: "POST",
    body: formData,
  });
  const payload = await readResponse(response, "Image dataset analysis failed.");
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Image dataset analysis failed. The server returned an invalid analysis response.");
  }
  return payload;
}

export async function predictImageModel(modelId, file, fetchImpl = globalThis.fetch) {
  const formData = new FormData();
  formData.append("model_id", modelId);
  formData.append("file", file, file.name);
  const response = await fetchImpl("/image/predict", {
    method: "POST",
    body: formData,
  });
  return readResponse(response, "Image prediction failed.");
}