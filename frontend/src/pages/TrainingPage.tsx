import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { listScenarios, type ScenarioSummary } from "../api/scenarios";
import { createCpSatEvaluationRun, createTrainingRun, getTrainingDetail, getTrainingMetrics, listEvaluationRuns, listTrainingRuns, requestTrainingStop, type TrainingConfig, type TrainingDetail, type TrainingMetric } from "../api/runs";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";

const defaults: TrainingConfig = { total_timesteps: 1_024, learning_seed: 17, evaluation_seed: 23, n_steps: 64, batch_size: 32, n_epochs: 2, learning_rate: 0.0003, gamma: 0.99, checkpoint_interval: 512, evaluation_interval: 512, deterministic_eval: true };
const activeStatuses = new Set(["queued", "running", "stop_requested"]);

export function useRequest<Value>(request: (signal: AbortSignal) => Promise<Value>, dependencies: unknown[], intervalMs?: number) {
  const [value, setValue] = useState<Value | null>(null); const [error, setError] = useState<Error | null>(null); const [token, setToken] = useState(0);
  const reload = useCallback(() => setToken((item) => item + 1), []);
  useEffect(() => { const controller = new AbortController(); const load = () => void request(controller.signal).then(setValue).catch((reason: unknown) => { if (!(reason instanceof DOMException && reason.name === "AbortError")) setError(reason instanceof Error ? reason : new Error("알 수 없는 오류가 발생했습니다.")); }); setError(null); load(); const timer = intervalMs ? window.setInterval(load, intervalMs) : undefined; return () => { controller.abort(); if (timer) window.clearInterval(timer); }; // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, token]);
  return { value, error, reload };
}

function inputNumber(config: TrainingConfig, setConfig: (config: TrainingConfig) => void, key: keyof TrainingConfig, label: string, step = 1) {
  return <label className="training-field" key={key}>{label}<input type="number" step={step} min={step > 0 ? step : undefined} value={config[key] as number} onChange={(event) => setConfig({ ...config, [key]: Number(event.target.value) })} /></label>;
}

/** 단일 local worker 제약을 화면에도 드러내는 학습 시작 설정 폼이다. */
export function TrainingPage() {
  const navigate = useNavigate(); const scenarios = useRequest(listScenarios, []); const runs = useRequest(listTrainingRuns, []); const evaluationRuns = useRequest(listEvaluationRuns, []);
  const [scenarioId, setScenarioId] = useState(""); const [config, setConfig] = useState(defaults); const [submitting, setSubmitting] = useState(false); const [cpSatSubmitting, setCpSatSubmitting] = useState(false); const [cpSatSeed, setCpSatSeed] = useState(23); const [cpSatTimeLimit, setCpSatTimeLimit] = useState(10); const [error, setError] = useState<Error | null>(null);
  useEffect(() => { if (!scenarioId && scenarios.value?.[0]) setScenarioId(scenarios.value[0].scenario_id); }, [scenarioId, scenarios.value]);
  const active = runs.value?.items.find((run) => activeStatuses.has(run.status)) ?? evaluationRuns.value?.items.find((run) => activeStatuses.has(run.status));
  const submit = async (event: React.FormEvent) => { event.preventDefault(); if (!scenarioId) return; setSubmitting(true); setError(null); try { const result = await createTrainingRun(scenarioId, config); navigate(`/training/${result.run.run_id}`); } catch (reason) { setError(reason instanceof Error ? reason : new Error("학습을 시작하지 못했습니다.")); } finally { setSubmitting(false); } };
  const startCpSat = async () => { if (!scenarioId) return; setCpSatSubmitting(true); setError(null); try { const result = await createCpSatEvaluationRun(scenarioId, cpSatSeed, cpSatTimeLimit); navigate(`/evaluations/${result.run.run_id}`); } catch (reason) { setError(reason instanceof Error ? reason : new Error("CP-SAT 평가를 시작하지 못했습니다.")); } finally { setCpSatSubmitting(false); } };
  return <section className="page-section"><div className="page-heading"><div><p className="eyebrow">MASKABLE PPO · CP-SAT</p><h1>학습 및 기준해 제어</h1><p>PPO 학습과 CP-SAT 기준해는 하나의 local worker 슬롯을 공유합니다.</p></div></div>{scenarios.error || runs.error || evaluationRuns.error || error ? <ErrorState error={error ?? scenarios.error ?? runs.error ?? evaluationRuns.error ?? new Error()} onRetry={() => { scenarios.reload(); runs.reload(); evaluationRuns.reload(); setError(null); }} /> : null}{!scenarios.value || !runs.value || !evaluationRuns.value ? <LoadingState /> : null}{active ? <section className="detail-card"><h2>진행 중인 실행</h2><p><code>{active.run_id}</code> · {active.status}</p>{"algorithm" in active ? <Link className="text-link" to={`/training/${active.run_id}`}>학습 상태 열기</Link> : <Link className="text-link" to={`/results/${active.run_id}`}>평가 상태 열기</Link>}</section> : null}{scenarios.value ? <><form className="training-form" onSubmit={submit}><section className="detail-card"><h2>PPO 학습 설정</h2><label className="training-field">시나리오<select value={scenarioId} onChange={(event) => setScenarioId(event.target.value)}>{scenarios.value.map((scenario: ScenarioSummary) => <option key={scenario.scenario_id} value={scenario.scenario_id}>{scenario.name} · Seed {scenario.seed}</option>)}</select></label><div className="training-fields">{inputNumber(config, setConfig, "total_timesteps", "총 timestep")}{inputNumber(config, setConfig, "learning_seed", "학습 seed")}{inputNumber(config, setConfig, "evaluation_seed", "평가 seed")}{inputNumber(config, setConfig, "n_steps", "Rollout step")}{inputNumber(config, setConfig, "batch_size", "Batch size")}</div></section><button disabled={submitting || Boolean(active) || !scenarioId} type="submit">{submitting ? "학습 시작 중…" : "학습 시작"}</button></form><section className="detail-card"><h2>CP-SAT 기준해</h2><div className="training-fields"><label className="training-field">평가 seed<input type="number" value={cpSatSeed} onChange={(event) => setCpSatSeed(Number(event.target.value))} /></label><label className="training-field">시간 제한(초)<input type="number" min="1" value={cpSatTimeLimit} onChange={(event) => setCpSatTimeLimit(Number(event.target.value))} /></label></div><button disabled={cpSatSubmitting || Boolean(active) || !scenarioId} onClick={startCpSat}>{cpSatSubmitting ? "CP-SAT 시작 중…" : "CP-SAT 기준해 실행"}</button></section></> : null}</section>;
}

function MetricChart({ metrics }: { metrics: TrainingMetric[] }) {
  if (!metrics.length) return <EmptyState>아직 기록된 평가 metric이 없습니다.</EmptyState>;
  const values = metrics.map((item) => item.evaluation.total_return); const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;
  const points = metrics.map((item, index) => `${(index / Math.max(1, metrics.length - 1)) * 100},${100 - ((item.evaluation.total_return - min) / range) * 100}`).join(" ");
  return <><svg className="metric-chart" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="평가 return 학습 곡선"><polyline points={points} fill="none" stroke="#78b6ff" strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg><p className="map-note">Return 범위 {min.toFixed(3)} ~ {max.toFixed(3)} · 최근 {metrics.at(-1)?.timesteps.toLocaleString()} timestep</p></>;
}

/** URL의 run ID를 기준으로 polling을 재개해 새로고침 뒤에도 상태를 복구한다. */
export function TrainingDetailPage() {
  const { runId = "" } = useParams(); const [stopError, setStopError] = useState<Error | null>(null); const [stopping, setStopping] = useState(false);
  const detail = useRequest((signal) => getTrainingDetail(runId, signal), [runId], 3_000);
  const active = detail.value ? activeStatuses.has(detail.value.run.status) : true;
  const metrics = useRequest((signal) => getTrainingMetrics(runId, signal), [runId], active ? 3_000 : undefined);
  const stop = async () => { setStopping(true); setStopError(null); try { await requestTrainingStop(runId); detail.reload(); } catch (reason) { setStopError(reason instanceof Error ? reason : new Error("중지 요청을 보낼 수 없습니다.")); } finally { setStopping(false); } };
  if (detail.error) return <section className="page-section"><Link className="text-link" to="/training">← 학습 제어</Link><ErrorState error={detail.error} onRetry={detail.reload} /></section>;
  if (!detail.value || !metrics.value) return <section className="page-section"><LoadingState label="학습 실행 상태를 복구하는 중입니다." /></section>;
  const run: TrainingDetail = detail.value; const latest = metrics.value.items.at(-1); const progress = latest ? Math.min(100, latest.timesteps / run.run.total_timesteps * 100) : 0;
  return <section className="page-section"><Link className="text-link" to="/training">← 학습 제어</Link><div className="page-heading"><div><p className="eyebrow">TRAINING RUN</p><h1>{run.run.algorithm}</h1><p><code>{run.run.run_id}</code> · {run.run.scenario_id}</p></div><span className={`run-status run-status--${run.run.status}`}>{run.run.status}</span></div>{stopError || metrics.error ? <ErrorState error={stopError ?? metrics.error ?? new Error()} onRetry={() => { detail.reload(); metrics.reload(); setStopError(null); }} /> : null}{run.run.error_message ? <div className="state-message state-message--error">{run.run.error_message}</div> : null}<div className="metric-grid"><div className="metric-card"><span>진행률</span><b>{progress.toFixed(1)}%</b></div><div className="metric-card"><span>최근 return</span><b>{latest?.evaluation.total_return.toFixed(3) ?? "-"}</b></div><div className="metric-card"><span>완료 strip / order</span><b>{latest ? `${latest.evaluation.completed_strips} / ${latest.evaluation.completed_orders}` : "-"}</b></div></div><section className="detail-card"><h2>제어 및 artifact</h2><p>Checkpoint {run.checkpoints.length}개 · 최종 모델 {run.final_model_available ? "생성됨" : "없음"} · 최종 평가 {run.final_evaluation_available ? "생성됨" : "없음"}</p>{active ? <button onClick={stop} disabled={stopping || run.run.status === "stop_requested"}>{run.run.status === "stop_requested" ? "중지 요청됨" : stopping ? "요청 중…" : "안전하게 중지 요청"}</button> : null}</section><section className="detail-card"><h2>학습 곡선</h2><MetricChart metrics={metrics.value.items} /></section><section className="detail-card"><h2>설정 snapshot</h2><dl><dt>Total timestep</dt><dd>{run.config.total_timesteps.toLocaleString()}</dd><dt>Seed</dt><dd>{run.config.learning_seed} / 평가 {run.config.evaluation_seed}</dd><dt>Rollout / batch</dt><dd>{run.config.n_steps} / {run.config.batch_size}</dd><dt>평가 / checkpoint</dt><dd>{run.config.evaluation_interval} / {run.config.checkpoint_interval}</dd></dl></section></section>;
}
