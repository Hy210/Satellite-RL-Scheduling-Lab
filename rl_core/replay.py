"""평가 episode를 사후 재생할 수 있는 공통 로그로 변환한다."""

from __future__ import annotations

from pathlib import Path

from rl_core.models import (
    EpisodeReplay,
    PolicyComparison,
    PolicyComparisonEntry,
    ReplayCandidate,
    ReplayCapture,
    ReplayRewardBreakdown,
    ReplayState,
    ReplayStep,
)
from rl_core.simulator import (
    RewardBreakdown,
    SatelliteSchedulingSimulator,
    SimulationObservation,
)


def replay_state(observation: SimulationObservation) -> ReplayState:
    """시뮬레이터 관측 중 재생 화면에 필요한 상태만 안정적인 계약으로 옮긴다."""

    return ReplayState(
        time_sec=observation.current_time_sec,
        roll_deg=observation.current_roll_deg,
        tilt_deg=observation.current_tilt_deg,
        completed_strips=observation.completed_strip_count,
        completed_orders=observation.completed_order_count,
    )


def replay_candidates(
    simulator: SatelliteSchedulingSimulator,
    observation: SimulationObservation,
) -> list[ReplayCandidate]:
    """선택 당시 후보와 마스킹 사유를 opportunity 식별자와 함께 보존한다."""

    candidates: list[ReplayCandidate] = []
    for candidate in observation.candidates:
        opportunity = simulator.opportunity_for_slot(candidate.slot)
        candidates.append(
            ReplayCandidate(
                slot=candidate.slot,
                opportunity_id=opportunity.opportunity_id,
                order_id=opportunity.order_id,
                strip_id=opportunity.strip_id,
                pass_id=opportunity.pass_id,
                capture_time_sec=opportunity.capture_time_sec,
                required_roll_deg=opportunity.required_roll_deg,
                required_tilt_deg=opportunity.required_tilt_deg,
                valid=candidate.valid,
                mask_reasons=list(candidate.mask_reasons),
            )
        )
    return candidates


def replay_reward_breakdown(breakdown: RewardBreakdown) -> ReplayRewardBreakdown:
    """reward 합계 검증을 위해 구성 요소와 total을 함께 저장한다."""

    return ReplayRewardBreakdown(
        strip_base=breakdown.strip_base,
        angle_bonus=breakdown.angle_bonus,
        missed_penalty=breakdown.missed_penalty,
        total=breakdown.total,
    )


def replay_step(
    *,
    step_index: int,
    simulator: SatelliteSchedulingSimulator,
    observation_before: SimulationObservation,
    candidates_before: list[ReplayCandidate],
    action: int,
    selected_opportunity_id: str | None,
    expired_order_ids: tuple[str, ...],
    breakdown: RewardBreakdown,
    observation_after: SimulationObservation,
) -> ReplayStep:
    """선택 전 후보 목록부터 선택 후 상태까지 한 step 로그를 만든다."""

    return ReplayStep(
        step_index=step_index,
        state_before=replay_state(observation_before),
        candidates=candidates_before,
        action=action,
        selected_opportunity_id=selected_opportunity_id,
        expired_order_ids=list(expired_order_ids),
        reward=breakdown.total,
        reward_breakdown=replay_reward_breakdown(breakdown),
        cumulative_return=simulator.cumulative_return,
        state_after=replay_state(observation_after),
    )


def replay_capture(
    *,
    step_index: int,
    simulator: SatelliteSchedulingSimulator,
    opportunity_id: str,
    reward: float,
) -> ReplayCapture:
    """최종 촬영 스케줄 표시에 필요한 촬영 action 요약을 만든다."""

    opportunity = simulator.opportunity_by_id(opportunity_id)
    return ReplayCapture(
        step_index=step_index,
        opportunity_id=opportunity.opportunity_id,
        order_id=opportunity.order_id,
        strip_id=opportunity.strip_id,
        pass_id=opportunity.pass_id,
        capture_time_sec=opportunity.capture_time_sec,
        roll_deg=opportunity.required_roll_deg,
        tilt_deg=opportunity.required_tilt_deg,
        reward=reward,
    )


def episode_replay(
    *,
    policy_name: str,
    scenario_id: str,
    seed: int,
    steps: list[ReplayStep],
    schedule: list[ReplayCapture],
) -> EpisodeReplay:
    """step 로그와 최종 스케줄을 episode 단위 결과로 묶는다."""

    final_state = steps[-1].state_after
    total_return = steps[-1].cumulative_return
    return EpisodeReplay(
        policy_name=policy_name,
        scenario_id=scenario_id,
        seed=seed,
        steps=steps,
        schedule=schedule,
        total_return=total_return,
        completed_strips=final_state.completed_strips,
        completed_orders=final_state.completed_orders,
    )


def save_episode_replay(path: Path, replay: EpisodeReplay) -> None:
    """Backend 저장 계층 전에도 평가 episode를 JSON 파일로 남길 수 있게 한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")


def load_episode_replay(path: Path) -> EpisodeReplay:
    """저장된 replay JSON이 공통 계약을 만족하는지 검증하며 다시 읽는다."""

    return EpisodeReplay.model_validate_json(path.read_text(encoding="utf-8"))


def policy_comparison_entry(
    *,
    replay: EpisodeReplay,
    priority_score: float,
    angle_bonus: float,
    missed_penalty: float,
    replay_path: Path | None = None,
    evaluation_run_id: str | None = None,
) -> PolicyComparisonEntry:
    """정책 평가 결과와 원본 실행을 비교 artifact의 한 행으로 요약한다."""

    return PolicyComparisonEntry(
        evaluation_run_id=evaluation_run_id,
        policy_name=replay.policy_name,
        scenario_id=replay.scenario_id,
        seed=replay.seed,
        total_return=replay.total_return,
        priority_score=priority_score,
        angle_bonus=angle_bonus,
        missed_penalty=missed_penalty,
        completed_strips=replay.completed_strips,
        completed_orders=replay.completed_orders,
        captures=len(replay.schedule),
        steps=len(replay.steps),
        replay_path=str(replay_path) if replay_path is not None else None,
    )


def policy_comparison(entries: list[PolicyComparisonEntry]) -> PolicyComparison:
    """동일 시나리오의 여러 정책 결과를 return 기준 비교 artifact로 묶는다."""

    if not entries:
        raise ValueError("policy comparison requires at least one entry")
    best_entry = max(
        entries,
        key=lambda entry: (
            entry.total_return,
            entry.completed_orders,
            entry.completed_strips,
            entry.captures,
            entry.policy_name,
        ),
    )
    return PolicyComparison(
        scenario_id=entries[0].scenario_id,
        entries=sorted(entries, key=lambda entry: entry.policy_name),
        best_policy_name=best_entry.policy_name,
    )


def save_policy_comparison(path: Path, comparison: PolicyComparison) -> None:
    """정책별 비교 요약을 웹과 저장 계층이 재사용할 JSON 파일로 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")


def load_policy_comparison(path: Path) -> PolicyComparison:
    """저장된 정책 비교 JSON을 검증하며 다시 읽는다."""

    return PolicyComparison.model_validate_json(path.read_text(encoding="utf-8"))
