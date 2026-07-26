import { useCallback, useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { getEvaluationResult, getEvaluationStep, getEvaluationTimeline, listCompletedEvaluationRuns, type EvaluationResult, type EvaluationRun, type ReplayStep, type TimelineCapture } from "../api/runs";
import { getOpportunityAttitudeTarget, getScenario } from "../api/scenarios";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { ScenarioMap } from "../components/ScenarioMap";

function useLoad<Value>(request: (signal: AbortSignal) => Promise<Value>, dependencies: unknown[]) {
  const [value, setValue] = useState<Value | null>(null); const [error, setError] = useState<Error | null>(null); const [token, setToken] = useState(0);
  const reload = useCallback(() => setToken((item) => item + 1), []);
  useEffect(() => { const controller = new AbortController(); setValue(null); setError(null); void request(controller.signal).then(setValue).catch((reason: unknown) => { if (reason instanceof DOMException && reason.name === "AbortError") return; setError(reason instanceof Error ? reason : new Error("알 수 없는 오류가 발생했습니다.")); }); return () => controller.abort(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, token]);
  return { value, error, reload };
}

function resultError(error: Error): string {
  if (!(error instanceof ApiError)) return error.message;
  return ({ evaluation_result_not_ready: "평가가 아직 완료되지 않았습니다.", evaluation_run_not_completed: "실패 또는 중지된 실행은 결과를 제공하지 않습니다.", evaluation_artifact_missing: "결과 artifact가 없습니다.", evaluation_artifact_invalid: "결과 artifact 검증에 실패했습니다." }[error.code] ?? error.message);
}

export function ResultsPage() {
  const loaded = useLoad(listCompletedEvaluationRuns, []);
  return <section className="page-section"><div className="page-heading"><div><p className="eyebrow">EVALUATION</p><h1>평가 결과</h1><p>완료된 평가 실행의 검증된 결과를 선택합니다.</p></div><button onClick={loaded.reload}>새로고침</button></div>{loaded.error ? <ErrorState error={loaded.error} onRetry={loaded.reload} /> : null}{!loaded.value && !loaded.error ? <LoadingState /> : null}{loaded.value?.items.length === 0 ? <EmptyState>완료된 평가 실행이 없습니다.</EmptyState> : null}{loaded.value?.items.length ? <div className="table-wrap"><table><thead><tr><th>정책</th><th>시나리오</th><th>Seed</th><th>Run ID</th><th /></tr></thead><tbody>{loaded.value.items.map((item: EvaluationRun) => <tr key={item.run_id}><td>{item.policy_name}</td><td><code>{item.scenario_id}</code></td><td>{item.seed}</td><td><code>{item.run_id}</code></td><td><Link className="text-link" to={`/results/${item.run_id}`}>결과 열기</Link></td></tr>)}</tbody></table></div> : null}</section>;
}

function clock(seconds: number) {
  const hours = Math.floor(seconds / 3600).toString().padStart(2, "0");
  const minutes = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
  const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${hours}:${minutes}:${remainder}`;
}

function signed(value: number) { return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`; }

/** schedule 항목의 step_index로 저장된 reward와 action-mask 근거를 지연 조회한다. */
function CaptureDetail({ runId, capture }: { runId: string; capture: TimelineCapture }) {
  const loaded = useLoad((signal) => getEvaluationStep(runId, capture.step_index, signal), [runId, capture.step_index]);
  if (loaded.error) return <ErrorState error={loaded.error} onRetry={loaded.reload} />;
  if (!loaded.value) return <LoadingState label="선택 촬영의 replay step을 불러오는 중입니다." />;
  const step: ReplayStep = loaded.value;
  const candidate = step.candidates.find((item) => item.opportunity_id === capture.opportunity_id);
  const invalidCount = step.candidates.filter((item) => !item.valid).length;
  return <section className="capture-detail"><h3>선택 촬영 상세</h3><dl><dt>시각</dt><dd>{clock(capture.capture_time_sec)}</dd><dt>자세 전 → 후</dt><dd>Roll {step.state_before.roll_deg.toFixed(1)}° → {step.state_after.roll_deg.toFixed(1)}° · Tilt {step.state_before.tilt_deg.toFixed(1)}° → {step.state_after.tilt_deg.toFixed(1)}°</dd><dt>보상 구성</dt><dd>기본 {signed(step.reward_breakdown.strip_base)} · 각도 {signed(step.reward_breakdown.angle_bonus)} · 미완료 {signed(step.reward_breakdown.missed_penalty)}</dd><dt>누적 return</dt><dd>{step.cumulative_return.toFixed(3)}</dd><dt>후보 상태</dt><dd>무효 후보 {invalidCount}개{candidate?.mask_reasons.length ? ` · 선택 후보 사유: ${candidate.mask_reasons.join(", ")}` : ""}</dd></dl></section>;
}

/** capture 선택을 지도·타임라인·URL query에서 공유하는 결과 시각화 영역이다. */
function EvaluationSchedule({ runId, scenarioId }: { runId: string; scenarioId: string }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const timeline = useLoad((signal) => getEvaluationTimeline(runId, signal), [runId]);
  const scenario = useLoad((signal) => getScenario(scenarioId, signal), [scenarioId]);
  const selectedCaptureId = searchParams.get("captureId") ?? undefined;
  const selected = timeline.value?.items.find((item) => item.opportunity_id === selectedCaptureId) ?? timeline.value?.items[0];
  const attitudeTarget = useLoad(
    (signal) => (selected ? getOpportunityAttitudeTarget(scenarioId, selected.opportunity_id, signal) : Promise.resolve(null)),
    [scenarioId, selected?.opportunity_id],
  );
  const selectCapture = (capture: TimelineCapture) => {
    const next = new URLSearchParams(searchParams); next.set("captureId", capture.opportunity_id); setSearchParams(next);
  };
  if (timeline.error || scenario.error) return <ErrorState error={timeline.error ?? scenario.error ?? new Error("결과 시각화를 불러오지 못했습니다.")} onRetry={() => { timeline.reload(); scenario.reload(); }} />;
  if (!timeline.value || !scenario.value) return <LoadingState label="촬영 타임라인과 지도를 불러오는 중입니다." />;
  if (timeline.value.items.length === 0) return <EmptyState>이 평가에서는 촬영된 strip이 없습니다.</EmptyState>;
  const evaluatedScenario = scenario.value;
  const completed = new Set(timeline.value.items.map((capture) => capture.strip_id));
  const orderStates = evaluatedScenario.orders.reduce((counts, order) => {
    const strips = evaluatedScenario.strips.filter((strip) => strip.order_id === order.order_id);
    const captured = strips.filter((strip) => completed.has(strip.strip_id)).length;
    const key = captured === 0 ? "missed" : captured === strips.length ? "completed" : "partial";
    counts[key] += 1; return counts;
  }, { completed: 0, partial: 0, missed: 0 });
  return <section className="schedule-layout"><div className="detail-card"><h2>24시간 촬영 타임라인</h2><ol className="timeline-list">{timeline.value.items.map((capture) => <li key={capture.opportunity_id}><button className={selected?.opportunity_id === capture.opportunity_id ? "timeline-item timeline-item--active" : "timeline-item"} onClick={() => selectCapture(capture)}><time>{clock(capture.capture_time_sec)}</time><span><code>{capture.strip_id}</code><small>{capture.pass_id} · Roll {capture.roll_deg.toFixed(1)}° · Tilt {capture.tilt_deg.toFixed(1)}°</small></span><b>{signed(capture.reward)}</b></button></li>)}</ol></div><div className="detail-card"><h2>지도 결과</h2><p className="map-legend">테두리: 우선순위 · 채움: 완료(초록) / 부분 완료 주문(주황) / 미촬영(회색) · 주문 {orderStates.completed} 완료, {orderStates.partial} 부분 완료, {orderStates.missed} 미촬영</p><ScenarioMap scenario={evaluatedScenario} selectedPassId={selected?.pass_id} selectedStripId={selected?.strip_id} completedStripIds={completed} attitudeTarget={attitudeTarget.value ?? undefined} captureTimeSec={selected?.capture_time_sec} onSelectStrip={(stripId) => { const first = timeline.value?.items.find((capture) => capture.strip_id === stripId); if (first) selectCapture(first); }} />{selected ? <CaptureDetail runId={runId} capture={selected} /> : null}</div></section>;
}

export function ResultDetailPage() {
  const { runId = "" } = useParams(); const loaded = useLoad((signal) => getEvaluationResult(runId, signal), [runId]);
  if (loaded.error) return <section className="page-section"><Link className="text-link" to="/results">← 평가 결과</Link><ErrorState error={new Error(resultError(loaded.error))} onRetry={loaded.reload} /></section>;
  if (!loaded.value) return <section className="page-section"><LoadingState label="평가 결과를 검증하며 불러오는 중입니다." /></section>;
  const result: EvaluationResult = loaded.value; const summary = result.summary;
  const cards = [["Total return", summary.total_return.toFixed(3)], ["Priority score", summary.priority_score.toFixed(3)], ["Captures", String(summary.captures)], ["Completed orders", String(summary.completed_orders)], ["Completed strips", String(summary.completed_strips)], ["Average off-nadir", `${summary.average_off_nadir_deg.toFixed(2)}°`]];
  return <section className="page-section"><Link className="text-link" to="/results">← 평가 결과</Link><div><p className="eyebrow">COMPLETED EVALUATION</p><h1>{summary.policy_name}</h1><p><code>{result.run.run_id}</code> · {summary.scenario_id} · Seed {summary.seed}</p></div><Link className="text-link" to={`/results/${result.run.run_id}/replay`}>Episode 재생 열기 →</Link><div className="metric-grid">{cards.map(([label, value]) => <div className="metric-card" key={label}><span>{label}</span><b>{value}</b></div>)}</div><section className="detail-card"><h2>Reward breakdown</h2><dl><dt>Priority score</dt><dd>{summary.priority_score.toFixed(3)}</dd><dt>Angle bonus</dt><dd>{summary.angle_bonus.toFixed(3)}</dd><dt>Missed penalty</dt><dd>{summary.missed_penalty.toFixed(3)}</dd><dt>Steps</dt><dd>{summary.steps.toLocaleString()}</dd></dl></section><EvaluationSchedule runId={result.run.run_id} scenarioId={summary.scenario_id} /></section>;
}
