export function getGroupColumnOptions(columns, targetColumn) {
  if (!Array.isArray(columns)) return [];
  return columns.filter(
    (column) => typeof column === "string" && column !== targetColumn,
  );
}

export function buildNlpTrainingRequest({ filename, text, groupColumn }) {
  return {
    filename,
    text,
    group_column: groupColumn || null,
  };
}

export function createLatestRequestTracker() {
  let latestRequest = 0;
  return {
    begin() {
      latestRequest += 1;
      return latestRequest;
    },
    isLatest(requestId) {
      return requestId === latestRequest;
    },
    invalidate(requestId) {
      if (requestId === latestRequest) latestRequest += 1;
    },
  };
}