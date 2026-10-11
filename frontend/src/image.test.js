import assert from "node:assert/strict";
import test from "node:test";

import {
  analyzeImageDataset,
  predictImageModel,
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