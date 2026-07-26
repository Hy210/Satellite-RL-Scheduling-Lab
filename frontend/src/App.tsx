import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { ScenarioDetailPage } from "./pages/ScenarioDetailPage";
import { ScenarioListPage } from "./pages/ScenarioListPage";
import { DashboardPage } from "./pages/DashboardPage";
import { ResultDetailPage, ResultsPage } from "./pages/ResultsPage";
import { TrainingDetailPage, TrainingPage } from "./pages/TrainingPage";
import { ReplayPage } from "./pages/ReplayPage";
import { ComparisonPage } from "./pages/ComparisonPage";
import { EvaluationStatusPage } from "./pages/EvaluationStatusPage";

function navClassName({ isActive }: { isActive: boolean }) {
  return isActive ? "app-nav__link app-nav__link--active" : "app-nav__link";
}

/** 단계별 화면을 같은 앱 shell 안에서 확장하기 위한 최상위 라우팅 구성이다. */
export function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <NavLink className="brand" to="/">NSICPS <span>RL Scheduling</span></NavLink>
        <span className="app-header__mode">읽기 전용 탐색</span>
      </header>
      <div className="app-body">
        <nav className="app-nav" aria-label="주요 메뉴">
          <NavLink className={navClassName} to="/">대시보드</NavLink>
          <NavLink className={navClassName} to="/scenarios">시나리오</NavLink>
          <NavLink className={navClassName} to="/training">학습</NavLink>
          <NavLink className={navClassName} to="/results">결과</NavLink>
          <NavLink className={navClassName} to="/comparisons">비교</NavLink>
        </nav>
        <main className="app-content">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/scenarios" element={<ScenarioListPage />} />
            <Route path="/scenarios/:scenarioId" element={<ScenarioDetailPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/results/:runId" element={<ResultDetailPage />} />
            <Route path="/results/:runId/replay" element={<ReplayPage />} />
            <Route path="/evaluations/:runId" element={<EvaluationStatusPage />} />
            <Route path="/comparisons" element={<ComparisonPage />} />
            <Route path="/training" element={<TrainingPage />} />
            <Route path="/training/:runId" element={<TrainingDetailPage />} />
            <Route path="*" element={<Navigate replace to="/" />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
