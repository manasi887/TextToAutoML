import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";

globalThis.React = React;

import("./App.jsx").then(({ default: App }) => {
  createRoot(document.getElementById("root")).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});