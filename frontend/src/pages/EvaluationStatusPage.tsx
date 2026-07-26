import { useEffect } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { getEvaluationRun } from "../api/runs";
import { ErrorState, LoadingState } from "../components/AsyncState";
import { useRequest } from "./TrainingPage";

/** worker 기반 평가의 queued/running 상태를 polling하고 완료 결과로 넘긴다. */
export function EvaluationStatusPage() {
  const { runId = "" } = useParams();
  const navigate = useNavigate();
  const loaded = useRequest((signal) => getEvaluationRun(runId, signal), [runId], 3_000);
  useEffect(() => {
    if (loaded.value?.run.status === "completed") navigate(`/results/${runId}`, { replace: true });
  }, [loaded.value, navigate, runId]);
  if (loaded.error) return <section className="page-section"><ErrorState error={loaded.error} onRetry={loaded.reload} /></section>;
  if (!loaded.value) return <section className="page-section"><LoadingState label="평가 실행 상태를 읽는 중입니다." /></section>;
  const run = loaded.value.run;
  if (run.status === "completed") return <section className="page-section"><LoadingState label="완료된 결과를 여는 중입니다." /></section>;
  return <section className="page-section"><Link className="text-link" to="/training">← 학습 및 기준해 제어</Link><div className="page-heading"><div><p className="eyebrow">EVALUATION RUN</p><h1>{run.policy_name}</h1><p><code>{run.run_id}</code> · {run.scenario_id} · Seed {run.seed}</p></div><span className={`run-status run-status--${run.status}`}>{run.status}</span></div>{run.status === "failed" ? <div className="state-message state-message--error">{run.error_message ?? "평가 실행이 실패했습니다."}</div> : <section className="detail-card"><p>CP-SAT 기준해를 계산 중입니다. 완료되면 결과 화면으로 자동 이동합니다.</p></section>}</section>;
}
