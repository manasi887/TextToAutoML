import assert from "node:assert/strict";
import test from "node:test";

import {
  buildNlpTrainingRequest,
  createLatestRequestTracker,
  getGroupColumnOptions,
} from "./training.js";

test("group selector options come from uploaded columns and exclude the target", () => {
  assert.deepEqual(
    getGroupColumnOptions(["student_id", "hours", "completed"], "completed"),
    ["student_id", "hours"],
  );
  assert.deepEqual(getGroupColumnOptions(null, "completed"), []);
});

test("training request sends the selected group column", () => {
  assert.deepEqual(
    buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
      groupColumn: "student_id",
    }),
    {
      filename: "courses.csv",
      text: "Predict completion",
      group_column: "student_id",
    },
  );
});

test("training request sends null when no group is selected", () => {
  assert.deepEqual(
    buildNlpTrainingRequest({
      filename: "courses.csv",
      text: "Predict completion",
      groupColumn: "",
    }),
    {
      filename: "courses.csv",
      text: "Predict completion",
      group_column: null,
    },
  );
});

test("request tracker ignores responses after a newer request starts", () => {
  const tracker = createLatestRequestTracker();
  const olderRequest = tracker.begin();
  const newerRequest = tracker.begin();

  assert.equal(tracker.isLatest(olderRequest), false);
  assert.equal(tracker.isLatest(newerRequest), true);

  tracker.invalidate(newerRequest);
  assert.equal(tracker.isLatest(newerRequest), false);
});