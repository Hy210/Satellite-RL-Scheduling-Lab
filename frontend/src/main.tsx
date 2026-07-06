import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./styles.css";

function App() {
  // 현재는 개발 기반 검증용 화면이며 실제 도메인 화면은 구현 계획 후반에 추가한다.
  return (
    <main>
      <h1>NSICPS RL Scheduling</h1>
      <p>Frontend foundation is ready. Domain screens will be added in a later phase.</p>
    </main>
  );
}

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element was not found");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
