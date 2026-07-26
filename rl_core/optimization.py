"""CP-SAT으로 축소 시나리오의 최적화 기준해를 만드는 기능이다."""

from __future__ import annotations

from collections.abc import Iterable
from math import isclose
from pathlib import Path

from ortools.sat.python import cp_model

from rl_core.models import (
    EpisodeReplay,
    Opportunity,
    OptimizationBaselineResult,
    ReplayCapture,
    ReplayStep,
    Scenario,
)
from rl_core.replay import (
    episode_replay,
    replay_candidates,
    replay_capture,
    replay_step,
)
from rl_core.simulator import SatelliteSchedulingSimulator

OBJECTIVE_SCALE = 1_000_000
CP_SAT_POLICY_NAME = "cp_sat_baseline"


def solve_cp_sat_baseline(
    scenario: Scenario,
    *,
    time_limit_sec: float = 10.0,
    seed: int = 0,
) -> OptimizationBaselineResult:
    """opportunity 선택 문제를 CP-SAT으로 풀고 기존 simulator로 결과를 재검증한다."""

    if time_limit_sec <= 0.0:
        raise ValueError("time_limit_sec must be positive")

    candidates = _statically_feasible_opportunities(scenario)
    model = cp_model.CpModel()
    variables = {
        opportunity.opportunity_id: model.new_bool_var(opportunity.opportunity_id)
        for opportunity in candidates
    }

    _add_strip_uniqueness_constraints(model, candidates, variables)
    _add_pairwise_transition_constraints(model, scenario, candidates, variables)
    _set_objective(model, scenario, candidates, variables)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.random_seed = seed
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    selected_ids = _selected_ids(status_code, solver, candidates, variables)
    replay = replay_cp_sat_selection(scenario, selected_ids, seed=seed)

    objective_value = (
        solver.ObjectiveValue() / OBJECTIVE_SCALE if _has_solution(status_code) else None
    )
    best_bound = (
        solver.BestObjectiveBound() / OBJECTIVE_SCALE if _has_solution(status_code) else None
    )
    return OptimizationBaselineResult(
        solver_name="ortools_cp_sat",
        scenario_id=scenario.scenario_id,
        seed=seed,
        status=status,
        objective_value=objective_value,
        best_objective_bound=best_bound,
        optimality_gap=_optimality_gap(objective_value, best_bound),
        time_limit_sec=time_limit_sec,
        selected_opportunity_ids=selected_ids,
        replay=replay,
    )


def replay_cp_sat_selection(
    scenario: Scenario,
    selected_opportunity_ids: Iterable[str],
    *,
    seed: int,
    max_steps: int = 1_000_000,
) -> EpisodeReplay:
    """solver 선택 ID를 시간순 action으로 바꾸어 simulator의 실제 제약으로 검증한다."""

    selected = set(selected_opportunity_ids)
    simulator = SatelliteSchedulingSimulator(scenario)
    replay_steps: list[ReplayStep] = []
    schedule: list[ReplayCapture] = []
    consumed: set[str] = set()

    for step_index in range(max_steps):
        if simulator.terminated:
            break
        observation = simulator.observe()
        candidates_before = replay_candidates(simulator, observation)
        selected_slots = [
            candidate.slot
            for candidate in observation.candidates
            if candidate.opportunity_id in selected
        ]
        if len(selected_slots) > 1:
            raise ValueError("CP-SAT selected multiple current-time opportunities")
        action = selected_slots[0] if selected_slots else 0
        if not observation.action_mask[action]:
            raise ValueError(f"CP-SAT selected a masked action slot {action}")

        selected_opportunity = simulator.opportunity_for_slot(action) if action else None
        result = simulator.step(action)
        replay_steps.append(
            replay_step(
                step_index=step_index,
                simulator=simulator,
                observation_before=observation,
                candidates_before=candidates_before,
                action=action,
                selected_opportunity_id=result.selected_opportunity_id,
                expired_order_ids=result.expired_order_ids,
                breakdown=result.breakdown,
                observation_after=result.observation,
            )
        )
        if selected_opportunity is not None:
            consumed.add(selected_opportunity.opportunity_id)
            schedule.append(
                replay_capture(
                    step_index=step_index,
                    simulator=simulator,
                    opportunity_id=selected_opportunity.opportunity_id,
                    reward=result.reward,
                )
            )
    else:
        raise RuntimeError(f"CP-SAT replay exceeded max_steps={max_steps}")

    missing = selected - consumed
    if missing:
        raise ValueError(f"CP-SAT selected opportunities that were not replayed: {sorted(missing)}")
    return episode_replay(
        policy_name=CP_SAT_POLICY_NAME,
        scenario_id=scenario.scenario_id,
        seed=seed,
        steps=replay_steps,
        schedule=schedule,
    )


def save_optimization_baseline(path: Path, result: OptimizationBaselineResult) -> None:
    """CP-SAT baseline artifact를 JSON으로 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_optimization_baseline(path: Path) -> OptimizationBaselineResult:
    """저장된 CP-SAT baseline artifact를 검증하며 다시 읽는다."""

    return OptimizationBaselineResult.model_validate_json(path.read_text(encoding="utf-8"))


def _statically_feasible_opportunities(scenario: Scenario) -> list[Opportunity]:
    simulator = SatelliteSchedulingSimulator(scenario)
    feasible = [
        opportunity
        for opportunity in scenario.opportunities
        if not simulator.mask_reasons(opportunity)
    ]
    return sorted(feasible, key=lambda item: (item.capture_time_sec, item.opportunity_id))


def _add_strip_uniqueness_constraints(
    model: cp_model.CpModel,
    opportunities: list[Opportunity],
    variables: dict[str, cp_model.IntVar],
) -> None:
    by_strip: dict[str, list[cp_model.IntVar]] = {}
    for opportunity in opportunities:
        by_strip.setdefault(opportunity.strip_id, []).append(variables[opportunity.opportunity_id])
    for strip_variables in by_strip.values():
        model.add(sum(strip_variables) <= 1)


def _add_pairwise_transition_constraints(
    model: cp_model.CpModel,
    scenario: Scenario,
    opportunities: list[Opportunity],
    variables: dict[str, cp_model.IntVar],
) -> None:
    for left_index, left in enumerate(opportunities):
        for right in opportunities[left_index + 1 :]:
            if _incompatible_pair(scenario, left, right):
                model.add(variables[left.opportunity_id] + variables[right.opportunity_id] <= 1)


def _incompatible_pair(scenario: Scenario, left: Opportunity, right: Opportunity) -> bool:
    if left.strip_id == right.strip_id:
        return True
    if isclose(left.capture_time_sec, right.capture_time_sec, abs_tol=1e-9):
        return True

    earlier, later = (
        (left, right) if left.capture_time_sec < right.capture_time_sec else (right, left)
    )
    earliest_later_start = (
        earlier.capture_time_sec
        + scenario.environment.imaging_duration_sec
        + max(
            scenario.environment.minimum_interval_sec,
            _slew_time(
                scenario,
                earlier.required_roll_deg,
                earlier.required_tilt_deg,
                later.required_roll_deg,
                later.required_tilt_deg,
            ),
        )
    )
    return later.capture_time_sec + 1e-9 < earliest_later_start


def _set_objective(
    model: cp_model.CpModel,
    scenario: Scenario,
    opportunities: list[Opportunity],
    variables: dict[str, cp_model.IntVar],
) -> None:
    coefficients = {
        opportunity.opportunity_id: int(
            round(_opportunity_reward(scenario, opportunity) * OBJECTIVE_SCALE)
        )
        for opportunity in opportunities
    }
    model.maximize(
        sum(
            coefficients[opportunity.opportunity_id] * variables[opportunity.opportunity_id]
            for opportunity in opportunities
        )
    )


def _opportunity_reward(scenario: Scenario, opportunity: Opportunity) -> float:
    order_by_id = {item.order_id: item for item in scenario.orders}
    strip_count_by_order: dict[str, int] = {item.order_id: 0 for item in scenario.orders}
    for strip in scenario.strips:
        strip_count_by_order[strip.order_id] += 1
    order = order_by_id[opportunity.order_id]
    strip_base = order.priority.score / strip_count_by_order[order.order_id]
    angle_quality = 1.0 - min(
        opportunity.off_nadir_deg / scenario.satellite.combined_off_nadir_limit_deg,
        1.0,
    )
    return strip_base + strip_base * scenario.reward.angle_bonus_weight * angle_quality


def _slew_time(
    scenario: Scenario,
    from_roll: float,
    from_tilt: float,
    to_roll: float,
    to_tilt: float,
) -> float:
    delta_roll = abs(to_roll - from_roll)
    delta_tilt = abs(to_tilt - from_tilt)
    if isclose(delta_roll, 0.0, abs_tol=1e-12) and isclose(delta_tilt, 0.0, abs_tol=1e-12):
        return 0.0
    satellite = scenario.satellite
    return (
        delta_roll / satellite.roll_rate_deg_per_sec
        + delta_tilt / satellite.tilt_rate_deg_per_sec
        + satellite.settling_time_sec
    )


def _selected_ids(
    status_code: int,
    solver: cp_model.CpSolver,
    opportunities: list[Opportunity],
    variables: dict[str, cp_model.IntVar],
) -> list[str]:
    if not _has_solution(status_code):
        return []
    selected = [
        opportunity
        for opportunity in opportunities
        if solver.BooleanValue(variables[opportunity.opportunity_id])
    ]
    selected.sort(key=lambda item: (item.capture_time_sec, item.opportunity_id))
    return [item.opportunity_id for item in selected]


def _has_solution(status_code: int) -> bool:
    return status_code in {cp_model.OPTIMAL, cp_model.FEASIBLE}


def _optimality_gap(objective_value: float | None, best_bound: float | None) -> float | None:
    if objective_value is None or best_bound is None:
        return None
    denominator = max(abs(objective_value), 1.0)
    return max(0.0, (best_bound - objective_value) / denominator)
