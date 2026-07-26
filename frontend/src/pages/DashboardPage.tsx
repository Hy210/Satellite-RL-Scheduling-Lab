import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listEvaluationRuns, listTrainingRuns, type EvaluationRun, type TrainingRun } from "../api/runs";
import { listScenarios, type ScenarioSummary } from "../api/scenarios";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";

type DashboardData = { scenarios: ScenarioSummary[]; training: TrainingRun[]; evaluations: EvaluationRun[] };

function RunStatusBadge({ status }: { status: string }) {
  return <span className={`run-status run-status--${status}`}>{status}</span>;
}

/** 저장된 metadata만 사용해 현재 프로젝트의 탐색 시작점을 제공하는 대시보드다. */
export function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const reload = useCallback(() => setReloadToken((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController(); setData(null); setError(null);
    void Promise.all([listScenarios(controller.signal), listTrainingRuns(controller.signal), listEvaluationRuns(controller.signal)])
      .then(([scenarios, training, evaluations]) => setData({ scenarios: scenarios.slice(0, 5), training: training.items, evaluations: evaluations.items }))
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason : new Error("알 수 없는 오류가 발생했습니다."));
      });
    return () => controller.abort();
  }, [reloadToken]);

  return <section className="page-section" aria-labelledby="dashboard-title"><div className="page-heading"><div><p className="eyebrow">OVERVIEW</p><h1 id="dashboard-title">대시보드</h1><p>저장된 시나리오와 실행 metadata를 읽기 전용으로 확인합니다.</p></div><button onClick={reload}>새로고침</button></div>{error ? <ErrorState error={error} onRetry={reload} /> : null}{!data && !error ? <LoadingState /> : null}{data ? <div className="dashboard-grid"><section className="dashboard-card"><div className="card-heading"><h2>최근 시나리오</h2><Link className="text-link" to="/scenarios">전체 보기</Link></div>{data.scenarios.length === 0 ? <EmptyState>저장된 시나리오가 없습니다.</EmptyState> : <ul className="metadata-list">{data.scenarios.map((item) => <li key={item.scenario_id}><Link to={`/scenarios/${item.scenario_id}`}>{item.name}</Link><span>Seed {item.seed}</span></li>)}</ul>}</section><section className="dashboard-card"><div className="card-heading"><h2>최근 학습 실행</h2></div>{data.training.length === 0 ? <EmptyState>학습 실행이 없습니다.</EmptyState> : <ul className="metadata-list">{data.training.map((item) => <li key={item.run_id}><div><code>{item.algorithm}</code><span>{item.total_timesteps.toLocaleString()} timesteps</span></div><RunStatusBadge status={item.status} /></li>)}</ul>}</section><section className="dashboard-card dashboard-card--wide"><div className="card-heading"><h2>최근 평가 실행</h2><Link className="text-link" to="/results">전체 보기</Link></div>{data.evaluations.length === 0 ? <EmptyState>평가 실행이 없습니다.</EmptyState> : <ul className="metadata-list">{data.evaluations.map((item) => <li key={item.run_id}><div>{item.status === "completed" ? <Link to={`/results/${item.run_id}`}>{item.policy_name}</Link> : <code>{item.policy_name}</code>}<span>{item.scenario_id} · Seed {item.seed}</span></div><RunStatusBadge status={item.status} /></li>)}</ul>}</section></div> : null}</section>;
}
