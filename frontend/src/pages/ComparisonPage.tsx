import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { createPolicyComparison, listCompletedEvaluationRuns, type EvaluationRun, type PolicyComparisonResponse } from "../api/runs";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";

function useLoad<Value>(request: (signal: AbortSignal) => Promise<Value>, dependencies: unknown[]) {
  const [value, setValue] = useState<Value | null>(null); const [error, setError] = useState<Error | null>(null); const [token, setToken] = useState(0);
  const reload = useCallback(() => setToken((current) => current + 1), []);
  useEffect(() => { const controller = new AbortController(); setValue(null); setError(null); void request(controller.signal).then(setValue).catch((reason: unknown) => { if (!(reason instanceof DOMException && reason.name === "AbortError")) setError(reason instanceof Error ? reason : new Error("비교 대상을 읽지 못했습니다.")); }); return () => controller.abort(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, token]);
  return { value, error, reload };
}

/** 같은 scenario·seed 실행을 명시적으로 골라 재현 가능한 비교 artifact를 생성한다. */
export function ComparisonPage() {
  const loaded = useLoad(listCompletedEvaluationRuns, []); const [groupKey, setGroupKey] = useState(""); const [selected, setSelected] = useState<string[]>([]); const [result, setResult] = useState<PolicyComparisonResponse | null>(null); const [error, setError] = useState<Error | null>(null); const [creating, setCreating] = useState(false);
  const groups = useMemo(() => { const map = new Map<string, EvaluationRun[]>(); loaded.value?.items.forEach((run) => { const key = `${run.scenario_id}::${run.seed}`; map.set(key, [...(map.get(key) ?? []), run]); }); return map; }, [loaded.value]);
  useEffect(() => { if (!groupKey && groups.size) setGroupKey([...groups.keys()][0]); }, [groupKey, groups]);
  const runs = groups.get(groupKey) ?? []; const [scenarioId, seedText] = groupKey.split("::"); const seed = Number(seedText);
  useEffect(() => { setSelected(runs.map((run) => run.run_id)); setResult(null); }, [groupKey]);
  const toggle = (runId: string) => setSelected((current) => current.includes(runId) ? current.filter((item) => item !== runId) : [...current, runId]);
  const create = async () => { if (!scenarioId || selected.length === 0) return; setCreating(true); setError(null); try { setResult(await createPolicyComparison(scenarioId, seed, selected)); } catch (reason) { setError(reason instanceof Error ? reason : new Error("비교 artifact를 만들지 못했습니다.")); } finally { setCreating(false); } };
  if (loaded.error) return <section className="page-section"><ErrorState error={loaded.error} onRetry={loaded.reload} /></section>;
  if (!loaded.value) return <section className="page-section"><LoadingState /></section>;
  if (!loaded.value.items.length) return <section className="page-section"><EmptyState>완료된 평가 run이 없습니다.</EmptyState></section>;
  return <section className="page-section"><div className="page-heading"><div><p className="eyebrow">POLICY COMPARISON</p><h1>정책 비교</h1><p>같은 시나리오와 seed의 완료 평가만 선택할 수 있습니다.</p></div><button onClick={loaded.reload}>새로고침</button></div>{error ? <ErrorState error={error} onRetry={() => setError(null)} /> : null}<section className="detail-card"><label className="training-field">비교 그룹<select value={groupKey} onChange={(event) => setGroupKey(event.target.value)}>{[...groups.entries()].map(([key, items]) => <option key={key} value={key}>{items[0].scenario_id} · Seed {items[0].seed} ({items.length} runs)</option>)}</select></label><div className="comparison-select">{runs.map((run) => <label key={run.run_id}><input type="checkbox" checked={selected.includes(run.run_id)} onChange={() => toggle(run.run_id)} /> {run.policy_name} <code>{run.run_id}</code></label>)}</div><button disabled={creating || selected.length === 0} onClick={create}>{creating ? "비교 생성 중…" : `${selected.length}개 run 비교`}</button></section>{result ? <section className="detail-card"><h2>비교 결과 · 최고 정책: {result.result.best_policy_name}</h2><div className="table-wrap"><table><thead><tr><th>정책</th><th>Total return</th><th>Priority</th><th>완료 strip/order</th><th>Captures</th><th /></tr></thead><tbody>{result.result.entries.map((entry) => { const run = entry.evaluation_run_id ? runs.find((item) => item.run_id === entry.evaluation_run_id) : undefined; return <tr key={entry.evaluation_run_id ?? entry.policy_name} className={entry.policy_name === result.result.best_policy_name ? "comparison-best" : ""}><td>{entry.policy_name}</td><td>{entry.total_return.toFixed(3)}</td><td>{entry.priority_score.toFixed(3)}</td><td>{entry.completed_strips} / {entry.completed_orders}</td><td>{entry.captures}</td><td>{run ? <><Link className="text-link" to={`/results/${run.run_id}`}>결과</Link> · <Link className="text-link" to={`/results/${run.run_id}/replay`}>재생</Link></> : "-"}</td></tr>; })}</tbody></table></div></section> : null}</section>;
}
