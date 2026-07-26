import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listScenarios, type ScenarioSummary } from "../api/scenarios";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("ko-KR");
}

/** 준비된 scenario를 상세 탐색 화면으로 연결하는 읽기 전용 목록 페이지다. */
export function ScenarioListPage() {
  const [items, setItems] = useState<ScenarioSummary[] | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const reload = useCallback(() => setReloadToken((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    setItems(null);
    setError(null);
    void listScenarios(controller.signal)
      .then(setItems)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason : new Error("알 수 없는 오류가 발생했습니다."));
      });
    return () => controller.abort();
  }, [reloadToken]);

  return (
    <section className="page-section" aria-labelledby="scenario-list-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">READ ONLY</p>
          <h1 id="scenario-list-title">시나리오</h1>
          <p>저장된 시나리오를 선택해 주문, strip, 촬영 기회와 검증 결과를 확인합니다.</p>
        </div>
        <button type="button" onClick={reload}>새로고침</button>
      </div>

      {error ? <ErrorState error={error} onRetry={reload} /> : null}
      {items === null && !error ? <LoadingState /> : null}
      {items?.length === 0 ? (
        <EmptyState>아직 저장된 시나리오가 없습니다.</EmptyState>
      ) : null}
      {items && items.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">이름</th>
                <th scope="col">Seed</th>
                <th scope="col">Scenario ID</th>
                <th scope="col">수정 시각</th>
                <th scope="col"><span className="sr-only">상세 보기</span></th>
              </tr>
            </thead>
            <tbody>
              {items.map((scenario) => (
                <tr key={scenario.scenario_id}>
                  <td>{scenario.name}</td>
                  <td>{scenario.seed}</td>
                  <td><code>{scenario.scenario_id}</code></td>
                  <td>{formatDateTime(scenario.updated_at)}</td>
                  <td>
                    <Link className="text-link" to={`/scenarios/${scenario.scenario_id}`}>열기</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
