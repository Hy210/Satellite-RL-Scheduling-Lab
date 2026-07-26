import { getJson, postJson } from "./client";
import type { PageResponse } from "./scenarios";

export type RunStatus = "queued" | "running" | "stop_requested" | "completed" | "stopped" | "failed";

export type TrainingRun = {
  run_id: string; scenario_id: string; algorithm: string; seed: number;
  total_timesteps: number; status: RunStatus; error_message: string | null;
};
export type TrainingConfig = {
  total_timesteps: number; learning_seed: number; evaluation_seed: number; n_steps: number;
  batch_size: number; n_epochs: number; learning_rate: number; gamma: number;
  checkpoint_interval: number; evaluation_interval: number; deterministic_eval: boolean;
};
export type TrainingMetric = {
  timesteps: number;
  evaluation: { total_return: number; priority_score: number; angle_bonus: number; missed_penalty: number; completed_strips: number; completed_orders: number; captures: number; steps: number };
};
export type TrainingDetail = { run: TrainingRun; config: TrainingConfig; checkpoints: string[]; final_model_available: boolean; final_evaluation_available: boolean };

export type EvaluationRun = {
  run_id: string; scenario_id: string; policy_name: string; seed: number;
  status: RunStatus; source_training_run_id: string | null; result_path: string | null; error_message: string | null;
};

export type EvaluationSummary = {
  policy_name: string; scenario_id: string; seed: number; steps: number; captures: number;
  total_return: number; priority_score: number; angle_bonus: number; missed_penalty: number;
  completed_strips: number; completed_orders: number; average_off_nadir_deg: number; replay_path: string;
};

export type EvaluationResult = { run: EvaluationRun; summary: EvaluationSummary };
export type TimelineCapture = {
  step_index: number; opportunity_id: string; order_id: string; strip_id: string; pass_id: string;
  capture_time_sec: number; roll_deg: number; tilt_deg: number; reward: number;
};
export type ReplayStep = {
  step_index: number; action: number; selected_opportunity_id: string | null; expired_order_ids: string[]; reward: number; cumulative_return: number;
  state_before: { time_sec: number; roll_deg: number; tilt_deg: number; completed_strips: number; completed_orders: number };
  state_after: { time_sec: number; roll_deg: number; tilt_deg: number; completed_strips: number; completed_orders: number };
  reward_breakdown: { strip_base: number; angle_bonus: number; missed_penalty: number; total: number };
  candidates: Array<{ opportunity_id: string; order_id: string; strip_id: string; pass_id: string; capture_time_sec: number; required_roll_deg: number; required_tilt_deg: number; valid: boolean; mask_reasons: string[] }>;
};
export type EpisodeSummary = { episode_id: string; policy_name: string; scenario_id: string; seed: number; steps: number; captures: number; total_return: number; completed_strips: number; completed_orders: number };
export type PolicyComparison = { scenario_id: string; entries: Array<{ evaluation_run_id: string | null; policy_name: string; seed: number; total_return: number; priority_score: number; angle_bonus: number; missed_penalty: number; completed_strips: number; completed_orders: number; captures: number; steps: number; replay_path: string | null }>; best_policy_name: string };
export type PolicyComparisonResponse = { comparison: { comparison_id: string; scenario_id: string; seed: number; evaluation_run_ids: string[]; artifact_path: string }; result: PolicyComparison };

/** run 목록은 artifact를 열지 않는 metadata 조회이므로 대시보드에 안전하게 사용한다. */
export function listTrainingRuns(signal?: AbortSignal): Promise<PageResponse<TrainingRun>> {
  return getJson<PageResponse<TrainingRun>>("/api/training-runs?limit=5", signal);
}

export function createTrainingRun(scenarioId: string, config: TrainingConfig): Promise<{ run: TrainingRun }> {
  return postJson<{ run: TrainingRun }>("/api/training-runs", { scenario_id: scenarioId, config });
}

/** CP-SAT은 PPO와 같은 로컬 실행 슬롯을 쓰므로 queued run만 즉시 반환한다. */
export function createCpSatEvaluationRun(scenarioId: string, seed: number, timeLimitSec: number): Promise<{ run: EvaluationRun }> {
  return postJson<{ run: EvaluationRun }>("/api/cp-sat-evaluation-runs", { scenario_id: scenarioId, seed, time_limit_sec: timeLimitSec });
}

export function requestTrainingStop(runId: string): Promise<{ run: TrainingRun }> {
  return postJson<{ run: TrainingRun }>(`/api/training-runs/${encodeURIComponent(runId)}/stop`);
}

export function getTrainingDetail(runId: string, signal?: AbortSignal): Promise<TrainingDetail> {
  return getJson<TrainingDetail>(`/api/training-runs/${encodeURIComponent(runId)}/detail`, signal);
}

export function getTrainingMetrics(runId: string, signal?: AbortSignal): Promise<PageResponse<TrainingMetric>> {
  return getJson<PageResponse<TrainingMetric>>(`/api/training-runs/${encodeURIComponent(runId)}/metrics?limit=500`, signal);
}

export function listEvaluationRuns(signal?: AbortSignal): Promise<PageResponse<EvaluationRun>> {
  return getJson<PageResponse<EvaluationRun>>("/api/evaluation-runs?limit=5", signal);
}

export function listCompletedEvaluationRuns(signal?: AbortSignal): Promise<PageResponse<EvaluationRun>> {
  return getJson<PageResponse<EvaluationRun>>("/api/evaluation-runs?status=completed&limit=100", signal);
}

export function getEvaluationRun(runId: string, signal?: AbortSignal): Promise<{ run: EvaluationRun }> {
  return getJson<{ run: EvaluationRun }>(`/api/evaluation-runs/${encodeURIComponent(runId)}`, signal);
}

export function getEvaluationResult(runId: string, signal?: AbortSignal): Promise<EvaluationResult> {
  return getJson<EvaluationResult>(`/api/results/${encodeURIComponent(runId)}`, signal);
}

/** 결과 지도와 타임라인은 replay 전체 대신 촬영 schedule만 먼저 읽는다. */
export function getEvaluationTimeline(runId: string, signal?: AbortSignal): Promise<PageResponse<TimelineCapture>> {
  return getJson<PageResponse<TimelineCapture>>(`/api/results/${encodeURIComponent(runId)}/timeline?limit=500`, signal);
}

/** pagination 위치와 무관하게 선택 capture에 대응하는 원본 step을 읽는다. */
export function getEvaluationStep(runId: string, stepIndex: number, signal?: AbortSignal): Promise<ReplayStep> {
  return getJson<ReplayStep>(`/api/results/${encodeURIComponent(runId)}/episodes/evaluation/steps/${stepIndex}`, signal);
}

export function getEvaluationEpisodes(runId: string, signal?: AbortSignal): Promise<{ items: EpisodeSummary[] }> {
  return getJson<{ items: EpisodeSummary[] }>(`/api/results/${encodeURIComponent(runId)}/episodes`, signal);
}

export function createPolicyComparison(scenarioId: string, seed: number, evaluationRunIds: string[]): Promise<PolicyComparisonResponse> {
  return postJson<PolicyComparisonResponse>("/api/policy-comparisons", { scenario_id: scenarioId, seed, evaluation_run_ids: evaluationRunIds });
}
