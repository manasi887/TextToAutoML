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

export function predictionIntentOverride(request) {
  const text = typeof request?.text === "string" ? request.text : "";
  const imageContext =
    request?.asset_type === "image" ||
    request?.asset_type === "image_dataset" ||
    request?.image_available ||
    request?.image_dataset_available;
  const unsupportedObjectAnalysis =
    /\bobject\s+detection\b|\b(?:detect|count|locate|localize|segment)\s+(?:all\s+)?objects?\b|\bidentify\s+(?:all\s+)?objects?\b/i.test(text);
  const topKRequest =
    /\btop[-\s]?(?:k|[2-9]|\d{2,}|two|three|four|five|six|seven|eight|nine)\b/i.test(text);
  const predicts = /\b(?:predict|classify|recognize|identify)\b/i.test(text);
  const negatesTraining =
    /\b(?:do\s+not|don't|dont|never|not)\s+(?:train|retrain|build|create)\b|\bno\s+(?:new\s+)?(?:training|model)\b/i.test(text);
  const affirmativeTraining =
    /\b(?:train|retrain|build|create|fit)\b/i.test(text) &&
    !negatesTraining;
  const explicitlyPrediction =
    negatesTraining ||
    /\b(?:existing|already\s+trained|saved)\s+(?:trained\s+)?(?:image\s+)?model\b|\bmodel\s+(?:id|with\s+id)\b/i.test(text) ||
    /\bclassify\s+(?:this|that|the|selected|uploaded|single)\s+(?:image|photo|picture)\b/i.test(text) ||
    !/\b(?:train|retrain|build|create|fit)\b/i.test(text);

  if (
    !imageContext ||
    unsupportedObjectAnalysis ||
    topKRequest ||
    !predicts ||
    !explicitlyPrediction ||
    affirmativeTraining
  ) {
    return null;
  }

  const requiredContext = [];
  if (!(request.image_available || request.asset_type === "image")) {
    requiredContext.push("image");
  }
  const modelId = request.image_model_id?.trim();
  if (!modelId) {
    requiredContext.push("image_model_id");
  } else if (!/^[A-Za-z0-9_]+$/.test(modelId)) {
    requiredContext.push("valid_image_model_id");
  }
  return {
    modality: "image",
    intent: "IMAGE_PREDICTION",
    ready: requiredContext.length === 0,
    clarification_required: false,
    reason: requiredContext.includes("valid_image_model_id")
      ? "Image model IDs may contain only letters, numbers, and underscores."
      : requiredContext.length === 2
        ? "A single image and a trained image model are required for image prediction."
        : requiredContext.includes("image")
          ? "A single image is required for image prediction."
          : requiredContext.length
            ? "A trained image model is required for image prediction."
            : "Ready to route to image prediction.",
    required_context: requiredContext,
  };
}

export async function routeImageRequest(request, fetchImpl = globalThis.fetch) {
  const predictionOverride = predictionIntentOverride(request);
  if (predictionOverride) return predictionOverride;

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
  if (typeof modelId !== "string" || !modelId.trim()) {
    throw new Error("Enter a trained image model ID before predicting.");
  }
  if (!file) {
    throw new Error("Choose a single image before predicting.");
  }
  const formData = new FormData();
  formData.append("model_id", modelId.trim());
  formData.append("file", file, file.name);
  const response = await fetchImpl("/image/predict", {
    method: "POST",
    body: formData,
  });
  return readResponse(response, "Image prediction failed.");
}