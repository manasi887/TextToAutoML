import assert from "node:assert/strict";
import test from "node:test";

import {
  analyzeImageDataset,
  predictImageModel,
  predictionIntentOverride,
  routeImageRequest,
  trainImageModel,
} from "./image.js";
import viteConfig, { resolveApiBase } from "../vite.config.js";

function namedBlob(name, type) {
  const file = new Blob(["image data"], { type });
  Object.defineProperty(file, "name", { value: name });
  return file;
}

function mockResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => payload,
  };
}

test("image training posts only the ZIP file to the image training endpoint", async () => {
  const payload = { task: "Image Classification", model_id: "img_123" };
  let request;
  const result = await trainImageModel(
    namedBlob("dataset.zip", "application/zip"),
    async (url, options) => {
      request = { url, options };
      return mockResponse(JSON.stringify(payload));
    },
  );

  assert.equal(request.url, "/image/train");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.body.get("file").name, "dataset.zip");
  assert.equal([...request.options.body.keys()].length, 1);
  assert.deepEqual(result, payload);
});

test("image prediction posts the model ID and image to the image prediction endpoint", async () => {
  const payload = {
    prediction: { class: "cats", confidence: 0.94 },
    probabilities: { cats: 0.94, dogs: 0.06 },
  };
  let request;
  const result = await predictImageModel(
    "img_123",
    namedBlob("cat.jpg", "image/jpeg"),
    async (url, options) => {
      request = { url, options };
      return mockResponse(JSON.stringify(payload));
    },
  );

  assert.equal(request.url, "/image/predict");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.body.get("model_id"), "img_123");
  assert.equal(request.options.body.get("file").name, "cat.jpg");
  assert.deepEqual(result, payload);
});

test("image prediction uses the selected file and never calls the training endpoint", async () => {
  const image = namedBlob("cat.4497.jpg", "image/jpeg");
  const calls = [];
  const result = await predictImageModel(
    "img_20261011_231604_5c125d",
    image,
    async (url, options) => {
      calls.push(url);
      assert.equal(options.body.get("model_id"), "img_20261011_231604_5c125d");
      assert.equal(options.body.get("file").name, "cat.4497.jpg");
      return mockResponse(JSON.stringify({ prediction: { class: "cat", confidence: 0.96 } }));
    },
  );

  assert.deepEqual(calls, ["/image/predict"]);
  assert.equal(result.prediction.class, "cat");
  assert.ok(!calls.includes("/image/train"));
});

test("image prediction validates missing model ID and image before requesting", async () => {
  let called = false;
  const fetchImpl = async () => {
    called = true;
    throw new Error("Should not call API without valid inputs.");
  };

  await assert.rejects(predictImageModel("", namedBlob("cat.jpg", "image/jpeg"), fetchImpl), /trained image model ID/);
  await assert.rejects(predictImageModel("img_existing", null, fetchImpl), /Choose a single image/);
  assert.equal(called, false);
});

test("image prediction displays backend errors as readable exceptions", async () => {
  await assert.rejects(
    predictImageModel("img_missing", namedBlob("cat.jpg", "image/jpeg"), async () =>
      mockResponse(JSON.stringify({ detail: "Image model manifest not found." }), 404),
    ),
    /Image model manifest not found\./,
  );
  await assert.rejects(
    predictImageModel("img_existing", namedBlob("broken.jpg", "image/jpeg"), async () =>
      mockResponse(JSON.stringify({ detail: "Corrupt or unreadable image file." }), 400),
    ),
    /Corrupt or unreadable image file\./,
  );
});

test("image dataset analysis posts the selected ZIP as multipart form data", async () => {
  const payload = { dataset: { image_count: 4, class_count: 2 } };
  let request;
  const result = await analyzeImageDataset(
    namedBlob("dataset.zip", "application/zip"),
    async (url, options) => {
      request = { url, options };
      return mockResponse(JSON.stringify(payload));
    },
  );

  assert.equal(request.url, "/image/analyze");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.body.get("file").name, "dataset.zip");
  assert.deepEqual([...request.options.body.keys()], ["file"]);
  assert.deepEqual(result, payload);
});

test("image dataset analysis rejects a missing ZIP before making a request", async () => {
  let called = false;

  await assert.rejects(
    analyzeImageDataset(null, async () => {
      called = true;
    }),
    /Choose an image dataset ZIP/,
  );
  assert.equal(called, false);
});

test("image dataset analysis rejects malformed successful responses", async () => {
  await assert.rejects(
    analyzeImageDataset(namedBlob("dataset.zip", "application/zip"), async () =>
      mockResponse("[]"),
    ),
    /invalid analysis response/,
  );
});

test("image dataset analysis uses shared HTTP and response parsing errors", async () => {
  await assert.rejects(
    analyzeImageDataset(namedBlob("dataset.zip", "application/zip"), async () =>
      mockResponse(JSON.stringify({ detail: "Corrupt image." }), 400),
    ),
    /Corrupt image\./,
  );
  await assert.rejects(
    analyzeImageDataset(namedBlob("dataset.zip", "application/zip"), async () =>
      mockResponse("", 200),
    ),
    /empty or invalid response/,
  );
});

test("image training requests route through the shared NLP endpoint with context", async () => {
  const requestBody = {
    text: "Train a CNN on these images",
    asset_type: "image_dataset",
    image_dataset_available: true,
    image_available: false,
    image_model_id: null,
    image_training_result_available: false,
  };
  const payload = {
    modality: "image",
    intent: "IMAGE_TRAINING_REQUEST",
    ready: true,
    clarification_required: false,
    reason: "Ready to train.",
    required_context: [],
  };
  let request;

  const result = await routeImageRequest(requestBody, async (url, options) => {
    request = { url, options };
    return mockResponse(JSON.stringify(payload));
  });

  assert.equal(request.url, "/nlp/route");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(request.options.body), requestBody);
  assert.deepEqual(result, payload);
});

test("image prediction requests route with the existing model and single-image context", async () => {
  const requestBody = {
    text: "Classify this image",
    asset_type: "image",
    image_dataset_available: false,
    image_available: true,
    image_model_id: "img_123",
    image_training_result_available: true,
  };

  const result = await routeImageRequest(requestBody, async (url, options) => {
    assert.equal(url, "/nlp/route");
    assert.deepEqual(JSON.parse(options.body), requestBody);
    return mockResponse(JSON.stringify({
      modality: "image",
      intent: "IMAGE_PREDICTION",
      ready: true,
      clarification_required: false,
      reason: "Ready to predict.",
      required_context: [],
    }));
  });

  assert.equal(result.intent, "IMAGE_PREDICTION");
  assert.equal(result.ready, true);
});

test("prediction prompt with an existing model ID is deterministically classified as prediction", async () => {
  const request = {
    text: "Use my existing trained image classification model with ID img_20261011_231604_5c125d to predict whether the uploaded image contains a cat or a dog.",
    asset_type: "image",
    image_available: true,
    image_model_id: "img_20261011_231604_5c125d",
  };
  let called = false;

  const decision = await routeImageRequest(request, async () => {
    called = true;
    throw new Error("Model-based routing should not override explicit prediction language.");
  });

  assert.equal(decision.intent, "IMAGE_PREDICTION");
  assert.equal(decision.ready, true);
  assert.equal(called, false);
});

test("Do not train a new model is a prediction override", () => {
  const decision = predictionIntentOverride({
    text: "Predict this image with my existing model. Do not train a new model.",
    asset_type: "image",
    image_available: true,
    image_model_id: "img_existing",
  });

  assert.equal(decision.intent, "IMAGE_PREDICTION");
  assert.equal(decision.ready, true);
});

test("a ZIP-backed image training request still uses the training route", async () => {
  const request = {
    text: "Train a CNN using this image dataset ZIP.",
    asset_type: "image_dataset",
    image_dataset_available: true,
    image_available: false,
    image_model_id: null,
  };
  let called = false;

  const decision = await routeImageRequest(request, async (url) => {
    called = true;
    assert.equal(url, "/nlp/route");
    return mockResponse(JSON.stringify({
      modality: "image",
      intent: "IMAGE_TRAINING_REQUEST",
      ready: true,
      clarification_required: false,
      reason: "Ready to train.",
      required_context: [],
    }));
  });

  assert.equal(called, true);
  assert.equal(decision.intent, "IMAGE_TRAINING_REQUEST");
});

test("affirmative training with both selected files is not overridden by prediction keywords", () => {
  const decision = predictionIntentOverride({
    text: "Train a new model using this ZIP to classify this image and predict its class.",
    asset_type: "image_dataset",
    image_dataset_available: true,
    image_available: true,
    image_model_id: "img_existing",
  });

  assert.equal(decision, null);
});

test("routing context preserves ZIP and single-image availability together", async () => {
  const request = {
    text: "Train a CNN on this image dataset.",
    asset_type: "image_dataset",
    image_dataset_available: true,
    image_available: true,
    image_model_id: "img_existing",
  };

  const result = await routeImageRequest(request, async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body), request);
    return mockResponse(JSON.stringify({
      modality: "image",
      intent: "IMAGE_TRAINING_REQUEST",
      ready: true,
      clarification_required: false,
      reason: "Ready to train.",
      required_context: [],
    }));
  });

  assert.equal(result.intent, "IMAGE_TRAINING_REQUEST");
});

test("routing rejects malformed successful responses", async () => {
  await assert.rejects(
    routeImageRequest({ text: "Analyze this" }, async () => mockResponse("null")),
    /invalid routing response/,
  );
});

test("routing reports non-JSON HTTP errors safely", async () => {
  await assert.rejects(
    routeImageRequest({ text: "Analyze this" }, async () =>
      mockResponse("<html>Unavailable</html>", 502),
    ),
    /Image request routing failed\. \(HTTP 502\)\./,
  );
});

test("routing propagates network failures as readable errors", async () => {
  await assert.rejects(
    routeImageRequest({ text: "Analyze this" }, async () => {
      throw new Error("Network unavailable.");
    }),
    /Network unavailable\./,
  );
});

test("image request errors use the factual API detail", async () => {
  await assert.rejects(
    trainImageModel(namedBlob("dataset.zip", "application/zip"), async () => ({
      ok: false,
      status: 400,
      text: async () => JSON.stringify({ detail: "Invalid image archive." }),
    })),
    /Invalid image archive\./,
  );
});

test("empty image API response produces a readable error", async () => {
  await assert.rejects(
    trainImageModel(namedBlob("dataset.zip", "application/zip"), async () =>
      mockResponse("", 502),
    ),
    /Image training failed\. \(HTTP 502\)\./,
  );
});

test("non-JSON image API error produces a readable error", async () => {
  await assert.rejects(
    trainImageModel(namedBlob("dataset.zip", "application/zip"), async () =>
      mockResponse("<html><body>Bad gateway</body></html>", 502),
    ),
    /Image training failed\. \(HTTP 502\)\./,
  );
});

test("successful empty response is reported without a JSON parser exception", async () => {
  await assert.rejects(
    trainImageModel(namedBlob("dataset.zip", "application/zip"), async () =>
      mockResponse("", 200),
    ),
    /Image training failed\. The server returned an empty or invalid response\./,
  );
});

test("Vite proxy uses VITE_API_BASE and falls back to port 8000", async () => {
  const originalApiBase = process.env.VITE_API_BASE;
  try {
    delete process.env.VITE_API_BASE;
    const fallbackConfig = await viteConfig({ mode: "test" });
    const fallback = "http://127.0.0.1:8000";
    assert.equal(resolveApiBase({}), fallback);
    for (const route of ["/upload", "/nlp", "/train", "/predict", "/image"]) {
      assert.equal(fallbackConfig.server.proxy[route], fallback);
    }

    process.env.VITE_API_BASE = "http://backend:8000";
    const environmentConfig = await viteConfig({ mode: "test" });
    assert.equal(resolveApiBase({ VITE_API_BASE: "http://backend:8000" }), "http://backend:8000");
    for (const route of ["/upload", "/nlp", "/train", "/predict", "/image"]) {
      assert.equal(environmentConfig.server.proxy[route], "http://backend:8000");
    }
  } finally {
    if (originalApiBase === undefined) delete process.env.VITE_API_BASE;
    else process.env.VITE_API_BASE = originalApiBase;
  }
});