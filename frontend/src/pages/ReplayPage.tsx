import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getEvaluationEpisodes, getEvaluationStep, type EpisodeSummary, type ReplayStep } from "../api/runs";
import { getScenario } from "../api/scenarios";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { ScenarioMap } from "../components/ScenarioMap";

function useLoad<Value>(request: (signal: AbortSignal) => Promise<Value>, dependencies: unknown[]) {
  const [value, setValue] = useState<Value | null>(null); const [error, setError] = useState<Error | null>(null); const [token, setToken] = useState(0);
  const reload = useCallback(() => setToken((current) => current + 1), []);
  useEffect(() => { const controller = new AbortController(); setValue(null); setError(null); void request(controller.signal).then(setValue).catch((reason: unknown) => { if (!(reason instanceof DOMException && reason.name === "AbortError")) setError(reason instanceof Error ? reason : new Error("재생 정보를 읽지 못했습니다.")); }); return () => controller.abort(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, token]);
  return { value, error, reload };
}

function time(seconds: number) { return `${Math.floor(seconds / 3600).toString().padStart(2, "0")}:${Math.floor(seconds % 3600 / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`; }

/** 저장 replay의 step을 재계산 없이 하나씩 탐색하고 지도 선택과 연결한다. */
export function ReplayPage() {
  const { runId = "" } = useParams(); const episodes = useLoad((signal) => getEvaluationEpisodes(runId, signal), [runId]);
  const episode: EpisodeSummary | undefined = episodes.value?.items[0]; const [index, setIndex] = useState(0); const [playing, setPlaying] = useState(false); const [speed, setSpeed] = useState(1);
  useEffect(() => { setIndex(0); setPlaying(false); }, [episode?.steps]);
  useEffect(() => { if (!playing || !episode) return; const timer = window.setInterval(() => setIndex((current) => { if (current >= episode.steps - 1) { setPlaying(false); return current; } return current + 1; }), 1_000 / speed); return () => window.clearInterval(timer); }, [episode, playing, speed]);
  const step = useLoad((signal) => getEvaluationStep(runId, index, signal), [runId, index]);
  const scenario = useLoad((signal) => episode ? getScenario(episode.scenario_id, signal) : Promise.reject(new DOMException("대기 중", "AbortError")), [episode?.scenario_id]);
  if (episodes.error) return <section className="page-section"><Link className="text-link" to={`/results/${runId}`}>← 평가 결과</Link><ErrorState error={episodes.error} onRetry={episodes.reload} /></section>;
  if (step.error || scenario.error) return <section className="page-section"><ErrorState error={step.error ?? scenario.error ?? new Error()} onRetry={() => { step.reload(); scenario.reload(); }} /></section>;
  if (!episode || !scenario.value || !step.value) return <section className="page-section"><LoadingState label="episode 재생 정보를 불러오는 중입니다." /></section>;
  const replay: ReplayStep = step.value; const selected = replay.selected_opportunity_id ? scenario.value.opportunities.find((item) => item.opportunity_id === replay.selected_opportunity_id) : undefined;
  return <section className="page-section"><Link className="text-link" to={`/results/${runId}`}>← 평가 결과</Link><div><p className="eyebrow">EPISODE REPLAY</p><h1>{episode.policy_name}</h1><p>Step {index + 1} / {episode.steps} · 누적 return {replay.cumulative_return.toFixed(3)}</p></div><section className="replay-layout"><div className="detail-card"><div className="replay-controls"><button onClick={() => { setPlaying(false); setIndex(0); }}>처음</button><button onClick={() => { setPlaying(false); setIndex((current) => Math.max(0, current - 1)); }}>이전</button><button onClick={() => setPlaying((current) => !current)}>{playing ? "정지" : "재생"}</button><button onClick={() => setIndex((current) => Math.min(episode.steps - 1, current + 1))}>다음</button><button onClick={() => { setPlaying(false); setIndex(episode.steps - 1); }}>마지막</button><select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option></select></div><input className="replay-range" type="range" min="0" max={Math.max(0, episode.steps - 1)} value={index} onChange={(event) => { setPlaying(false); setIndex(Number(event.target.value)); }} /><dl><dt>시각</dt><dd>{time(replay.state_before.time_sec)} → {time(replay.state_after.time_sec)}</dd><dt>자세</dt><dd>Roll {replay.state_before.roll_deg.toFixed(1)}° → {replay.state_after.roll_deg.toFixed(1)}° · Tilt {replay.state_before.tilt_deg.toFixed(1)}° → {replay.state_after.tilt_deg.toFixed(1)}°</dd><dt>선택 action</dt><dd>{selected ? <code>{selected.opportunity_id}</code> : "skip"}</dd><dt>Reward</dt><dd>{replay.reward.toFixed(3)} (기본 {replay.reward_breakdown.strip_base.toFixed(3)} · 각도 {replay.reward_breakdown.angle_bonus.toFixed(3)} · 미완료 {replay.reward_breakdown.missed_penalty.toFixed(3)})</dd></dl></div><div className="detail-card"><h2>지도 동기화</h2><ScenarioMap scenario={scenario.value} selectedPassId={selected?.pass_id} selectedStripId={selected?.strip_id} onSelectStrip={() => undefined} /><p className="map-note">촬영 action은 해당 pass와 strip을 강조합니다. skip step은 현재 지도 선택을 바꾸지 않습니다.</p></div></section><section className="detail-card"><h2>action 후보와 mask</h2>{replay.candidates.length === 0 ? <EmptyState>이 step에는 촬영 후보가 없습니다.</EmptyState> : <div className="table-wrap"><table><thead><tr><th>Opportunity</th><th>Pass</th><th>시각</th><th>자세</th><th>상태</th></tr></thead><tbody>{replay.candidates.map((candidate) => <tr key={candidate.opportunity_id} className={candidate.opportunity_id === replay.selected_opportunity_id ? "replay-selected" : ""}><td><code>{candidate.opportunity_id}</code></td><td>{candidate.pass_id}</td><td>{time(candidate.capture_time_sec)}</td><td>{candidate.required_roll_deg.toFixed(1)}° / {candidate.required_tilt_deg.toFixed(1)}°</td><td>{candidate.valid ? "가능" : candidate.mask_reasons.join(", ")}</td></tr>)}</tbody></table></div>}</section></section>;
}
