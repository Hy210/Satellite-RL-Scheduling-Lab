import { useCallback, useEffect, useState } from "react";
import { Link, useSearchParams, useParams } from "react-router-dom";

import {
  getScenario, listOpportunities, listOrders, listStrips, validateScenario,
  type OpportunityKind, type PageResponse, type Priority, type Scenario,
  type ScenarioValidation,
} from "../api/scenarios";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { ScenarioMap } from "../components/ScenarioMap";

type Tab = "overview" | "map" | "orders" | "strips" | "opportunities" | "validation";
const tabs: Array<[Tab, string]> = [["overview", "개요"], ["map", "지도"], ["orders", "주문"], ["strips", "Strip"], ["opportunities", "촬영 기회"], ["validation", "검증"]];

function seconds(value: number) { return `${value.toLocaleString("ko-KR")}초`; }
function degrees(value: number) { return `${value.toFixed(1)}°`; }
function range(value: { minimum: number; maximum: number }) { return `${degrees(value.minimum)} ~ ${degrees(value.maximum)}`; }

function useLoaded<Value>(load: (signal: AbortSignal) => Promise<Value>, dependencies: unknown[]) {
  const [value, setValue] = useState<Value | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const reload = useCallback(() => setReloadToken((current) => current + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setValue(null); setError(null);
    void load(controller.signal).then(setValue).catch((reason: unknown) => {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      setError(reason instanceof Error ? reason : new Error("알 수 없는 오류가 발생했습니다."));
    });
    return () => controller.abort();
  // API 요청의 입력과 재시도 token만 변경 시 다시 읽는다.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, reloadToken]);
  return { value, error, reload };
}

function PageControls<Item>({ page, onOffset }: { page: PageResponse<Item>; onOffset: (offset: number) => void }) {
  const first = page.total === 0 ? 0 : page.offset + 1;
  const last = Math.min(page.offset + page.items.length, page.total);
  return <div className="page-controls"><span>{first}–{last} / {page.total}</span><button disabled={page.offset === 0} onClick={() => onOffset(Math.max(0, page.offset - page.limit))}>이전</button><button disabled={page.offset + page.limit >= page.total} onClick={() => onOffset(page.offset + page.limit)}>다음</button></div>;
}

function Overview({ scenario }: { scenario: Scenario }) {
  return <div className="detail-grid">
    <section className="detail-card"><h2>위성 자세</h2><dl><dt>위성 ID</dt><dd>{scenario.satellite.satellite_id}</dd><dt>Roll / Tilt 한계</dt><dd>{degrees(scenario.satellite.roll_limit_deg)} / {degrees(scenario.satellite.tilt_limit_deg)}</dd><dt>Off-nadir 한계</dt><dd>{degrees(scenario.satellite.combined_off_nadir_limit_deg)}</dd><dt>기동 속도</dt><dd>Roll {scenario.satellite.roll_rate_deg_per_sec}°/s · Tilt {scenario.satellite.tilt_rate_deg_per_sec}°/s</dd><dt>안정화</dt><dd>{seconds(scenario.satellite.settling_time_sec)}</dd></dl></section>
    <section className="detail-card"><h2>환경</h2><dl><dt>Episode 길이</dt><dd>{seconds(scenario.environment.duration_sec)}</dd><dt>촬영 시간</dt><dd>{seconds(scenario.environment.imaging_duration_sec)}</dd><dt>최소 간격</dt><dd>{seconds(scenario.environment.minimum_interval_sec)}</dd><dt>최대 strip / 후보</dt><dd>{scenario.environment.max_strips.toLocaleString()} / {scenario.environment.max_candidates}</dd></dl></section>
    <section className="detail-card"><h2>보상</h2><dl><dt>Angle bonus</dt><dd>{scenario.reward.angle_bonus_weight}</dd><dt>Missed penalty</dt><dd>{scenario.reward.missed_penalty_weight}</dd></dl></section>
    <section className="detail-card"><h2>Pass</h2><div className="compact-list">{scenario.passes.map((item) => <div key={item.pass_id}><code>{item.pass_id}</code><span>{seconds(item.start_time_sec)} ~ {seconds(item.end_time_sec)}</span></div>)}</div></section>
  </div>;
}

/** 선택한 pass와 strip만 상세 레이어로 그려 대규모 시나리오 탐색 비용을 제한한다. */
function MapTab({ scenario, selectedPassId, selectedStripId, setQuery }: {
  scenario: Scenario; selectedPassId?: string; selectedStripId?: string;
  setQuery: (values: Record<string, string | undefined>) => void;
}) {
  return <div className="tab-stack"><div className="filter-row"><label>Pass <select value={selectedPassId ?? ""} onChange={(event) => setQuery({ passId: event.target.value || undefined, stripId: undefined })}><option value="">선택 안 함</option>{scenario.passes.map((item) => <option key={item.pass_id} value={item.pass_id}>{item.pass_id}</option>)}</select></label>{selectedStripId ? <button onClick={() => setQuery({ stripId: undefined })}>Strip 선택 해제</button> : null}</div><p className="map-note">주문 윤곽은 항상 표시합니다. pass를 선택하면 해당 ground track, footprint와 교차 strip을, strip을 선택하면 해당 polygon을 강조합니다.</p><ScenarioMap scenario={scenario} selectedPassId={selectedPassId} selectedStripId={selectedStripId} onSelectStrip={(stripId) => setQuery({ stripId })} /></div>;
}

function OrdersTab({ scenarioId, priority, setQuery }: { scenarioId: string; priority?: Priority; setQuery: (values: Record<string, string | undefined>) => void }) {
  const offset = Number(new URLSearchParams(window.location.search).get("offset") ?? "0");
  const loaded = useLoaded((signal) => listOrders(scenarioId, { offset, priority }, signal), [scenarioId, offset, priority]);
  if (loaded.error) return <ErrorState error={loaded.error} onRetry={loaded.reload} />;
  if (!loaded.value) return <LoadingState label="주문 목록을 불러오는 중입니다." />;
  return <div className="tab-stack"><label className="filter-label">우선순위 <select value={priority ?? ""} onChange={(event) => setQuery({ priority: event.target.value || undefined, offset: undefined })}><option value="">전체</option><option value="red">red</option><option value="blue">blue</option><option value="background">background</option></select></label>{loaded.value.items.length === 0 ? <EmptyState>조건에 맞는 주문이 없습니다.</EmptyState> : <div className="table-wrap"><table><thead><tr><th>이름</th><th>우선순위</th><th>요청 기간</th><th>허용 Roll / Tilt</th><th>Strip</th><th>기회</th><th /></tr></thead><tbody>{loaded.value.items.map((item) => <tr key={item.order_id}><td>{item.name}<br /><code>{item.order_id}</code></td><td><span className={`priority priority--${item.priority}`}>{item.priority}</span></td><td>{seconds(item.request_start_sec)} ~ {seconds(item.request_end_sec)}</td><td>{range(item.allowed_roll_deg)} / {range(item.allowed_tilt_deg)}</td><td>{item.strip_count}</td><td>{item.opportunity_count}</td><td><button onClick={() => setQuery({ tab: "strips", orderId: item.order_id, priority: undefined, offset: undefined })}>Strip 보기</button></td></tr>)}</tbody></table></div>}<PageControls page={loaded.value} onOffset={(next) => setQuery({ offset: String(next) })} /></div>;
}

function StripsTab({ scenarioId, orderId, setQuery }: { scenarioId: string; orderId?: string; setQuery: (values: Record<string, string | undefined>) => void }) {
  const offset = Number(new URLSearchParams(window.location.search).get("offset") ?? "0");
  const loaded = useLoaded((signal) => listStrips(scenarioId, { offset, orderId }, signal), [scenarioId, offset, orderId]);
  if (loaded.error) return <ErrorState error={loaded.error} onRetry={loaded.reload} />;
  if (!loaded.value) return <LoadingState label="Strip 목록을 불러오는 중입니다." />;
  return <div className="tab-stack">{orderId ? <p className="filter-note">선택 주문: <code>{orderId}</code> <button onClick={() => setQuery({ orderId: undefined, offset: undefined })}>필터 해제</button></p> : null}{loaded.value.items.length === 0 ? <EmptyState>조건에 맞는 strip이 없습니다.</EmptyState> : <div className="table-wrap"><table><thead><tr><th>Strip ID</th><th>Order ID</th><th>순서</th><th>촬영 기회</th><th /></tr></thead><tbody>{loaded.value.items.map((item) => <tr key={item.strip_id}><td><code>{item.strip_id}</code></td><td><code>{item.order_id}</code></td><td>{item.sequence}</td><td>{item.opportunity_count}</td><td><button onClick={() => setQuery({ tab: "opportunities", stripId: item.strip_id, orderId: item.order_id, offset: undefined })}>기회 보기</button></td></tr>)}</tbody></table></div>}<PageControls page={loaded.value} onOffset={(next) => setQuery({ offset: String(next) })} /></div>;
}

function OpportunitiesTab({ scenarioId, scenario, searchParams, setQuery }: { scenarioId: string; scenario: Scenario; searchParams: URLSearchParams; setQuery: (values: Record<string, string | undefined>) => void }) {
  const offset = Number(searchParams.get("offset") ?? "0"); const orderId = searchParams.get("orderId") ?? undefined; const stripId = searchParams.get("stripId") ?? undefined; const passId = searchParams.get("passId") ?? undefined; const kind = (searchParams.get("kind") as OpportunityKind | null) ?? undefined;
  const loaded = useLoaded((signal) => listOpportunities(scenarioId, { offset, orderId, stripId, passId, kind }, signal), [scenarioId, offset, orderId, stripId, passId, kind]);
  if (loaded.error) return <ErrorState error={loaded.error} onRetry={loaded.reload} />;
  if (!loaded.value) return <LoadingState label="촬영 기회를 불러오는 중입니다." />;
  return <div className="tab-stack"><div className="filter-row"><label>Pass <select value={passId ?? ""} onChange={(event) => setQuery({ passId: event.target.value || undefined, offset: undefined })}><option value="">전체</option>{scenario.passes.map((item) => <option key={item.pass_id} value={item.pass_id}>{item.pass_id}</option>)}</select></label><label>Kind <select value={kind ?? ""} onChange={(event) => setQuery({ kind: event.target.value || undefined, offset: undefined })}><option value="">전체</option><option value="early">early</option><option value="min_off_nadir">min_off_nadir</option><option value="late">late</option></select></label>{orderId || stripId ? <button onClick={() => setQuery({ orderId: undefined, stripId: undefined, offset: undefined })}>연결 필터 해제</button> : null}</div>{loaded.value.items.length === 0 ? <EmptyState>조건에 맞는 촬영 기회가 없습니다.</EmptyState> : <div className="table-wrap"><table><thead><tr><th>기회 ID</th><th>Order / Strip</th><th>Pass / kind</th><th>촬영 시각</th><th>Window</th><th>Roll / Tilt</th><th>Off-nadir</th></tr></thead><tbody>{loaded.value.items.map((item) => <tr key={item.opportunity_id}><td><code>{item.opportunity_id}</code></td><td><code>{item.order_id}</code><br /><code>{item.strip_id}</code></td><td>{item.pass_id}<br />{item.kind}</td><td>{seconds(item.capture_time_sec)}</td><td>{seconds(item.window_start_sec)} ~ {seconds(item.window_end_sec)}</td><td>{degrees(item.required_roll_deg)} / {degrees(item.required_tilt_deg)}</td><td>{degrees(item.off_nadir_deg)}</td></tr>)}</tbody></table></div>}<PageControls page={loaded.value} onOffset={(next) => setQuery({ offset: String(next) })} /></div>;
}

function ValidationTab({ scenarioId }: { scenarioId: string }) {
  const loaded = useLoaded((signal) => validateScenario(scenarioId, signal), [scenarioId]);
  if (loaded.error) return <ErrorState error={loaded.error} onRetry={loaded.reload} />;
  if (!loaded.value) return <LoadingState label="시나리오 무결성을 검증하는 중입니다." />;
  const validation: ScenarioValidation = loaded.value;
  if (validation.valid) return <EmptyState>구조와 artifact 무결성 검사를 통과했습니다.</EmptyState>;
  return <div className="table-wrap"><table><thead><tr><th>코드</th><th>경로</th><th>설명</th></tr></thead><tbody>{validation.issues.map((issue, index) => <tr key={`${issue.code}-${index}`}><td><code>{issue.code}</code></td><td><code>{issue.location.join(".")}</code></td><td>{issue.message}</td></tr>)}</tbody></table></div>;
}

/** 상단 Scenario 요약과 URL 기반 탭 상태로 읽기 전용 탐색을 제공한다. */
export function ScenarioDetailPage() {
  const { scenarioId = "" } = useParams(); const [searchParams, setSearchParams] = useSearchParams();
  const requestedTab = searchParams.get("tab"); const tab: Tab = tabs.some(([key]) => key === requestedTab) ? requestedTab as Tab : "overview";
  const loaded = useLoaded((signal) => getScenario(scenarioId, signal), [scenarioId]);
  const setQuery = useCallback((changes: Record<string, string | undefined>) => { const next = new URLSearchParams(searchParams); Object.entries(changes).forEach(([key, value]) => value === undefined ? next.delete(key) : next.set(key, value)); setSearchParams(next); }, [searchParams, setSearchParams]);
  if (loaded.error) return <section className="page-section"><Link className="text-link" to="/scenarios">시나리오 목록으로</Link><ErrorState error={loaded.error} onRetry={loaded.reload} /></section>;
  if (!loaded.value) return <section className="page-section"><LoadingState label="시나리오 상세를 불러오는 중입니다." /></section>;
  const scenario = loaded.value;
  return <section className="page-section"><Link className="text-link" to="/scenarios">← 시나리오 목록</Link><div className="scenario-hero"><div><p className="eyebrow">SCENARIO · READ ONLY</p><h1>{scenario.name}</h1><p><code>{scenario.scenario_id}</code> · Seed {scenario.seed}</p></div><div className="stat-grid"><span><b>{scenario.orders.length}</b>주문</span><span><b>{scenario.strips.length}</b>Strip</span><span><b>{scenario.opportunities.length}</b>기회</span><span><b>{scenario.passes.length}</b>Pass</span></div></div><div className="tabs" role="tablist" aria-label="시나리오 상세 탭">{tabs.map(([key, label]) => <button key={key} role="tab" aria-selected={tab === key} className={tab === key ? "tab tab--active" : "tab"} onClick={() => setQuery({ tab: key, offset: undefined })}>{label}</button>)}</div>{tab === "overview" ? <Overview scenario={scenario} /> : null}{tab === "map" ? <MapTab scenario={scenario} selectedPassId={searchParams.get("passId") ?? undefined} selectedStripId={searchParams.get("stripId") ?? undefined} setQuery={setQuery} /> : null}{tab === "orders" ? <OrdersTab scenarioId={scenarioId} priority={(searchParams.get("priority") as Priority | null) ?? undefined} setQuery={setQuery} /> : null}{tab === "strips" ? <StripsTab scenarioId={scenarioId} orderId={searchParams.get("orderId") ?? undefined} setQuery={setQuery} /> : null}{tab === "opportunities" ? <OpportunitiesTab scenarioId={scenarioId} scenario={scenario} searchParams={searchParams} setQuery={setQuery} /> : null}{tab === "validation" ? <ValidationTab scenarioId={scenarioId} /> : null}</section>;
}
