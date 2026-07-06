"""RL 성능의 비교 기준이 되는 결정론적 휴리스틱 정책과 평가 기능이다."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from rl_core.models import Opportunity, Scenario
from rl_core.simulator import SatelliteSchedulingSimulator, SimulationObservation


@dataclass(frozen=True, slots=True)
class DecisionLog:
    """정책이 각 step에서 본 후보와 선택, reward를 재생할 수 있게 기록한다."""

    step_index: int
    time_sec: float
    action: int
    opportunity_id: str | None
    valid_candidate_slots: tuple[int, ...]
    reward: float
    cumulative_return: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """동일 시나리오에서 정책들을 정량 비교하기 위한 episode 요약이다."""

    policy_name: str
    scenario_id: str
    seed: int
    steps: int
    captures: int
    total_return: float
    priority_score: float
    angle_bonus: float
    missed_penalty: float
    completed_strips: int
    completed_orders: int
    average_off_nadir_deg: float
    decisions: tuple[DecisionLog, ...]


class BaselinePolicy(ABC):
    """학습하지 않는 모든 비교 정책이 따르는 공통 선택 인터페이스다."""

    name: ClassVar[str]

    @abstractmethod
    def select_action(
        self,
        simulator: SatelliteSchedulingSimulator,
        observation: SimulationObservation,
        rng: random.Random,
    ) -> int:
        """유효 후보 slot을 반환하고 후보가 없으면 항상 가능한 skip 0을 반환한다."""


class RandomValidPolicy(BaselinePolicy):
    """마스킹되지 않은 후보를 seed 기반으로 무작위 선택하는 최저 기준선이다."""

    name = "random_valid"

    def select_action(
        self,
        simulator: SatelliteSchedulingSimulator,
        observation: SimulationObservation,
        rng: random.Random,
    ) -> int:
        del simulator
        valid_slots = _valid_slots(observation)
        return rng.choice(valid_slots) if valid_slots else 0


class EarliestDeadlineFirstPolicy(BaselinePolicy):
    """주문 마감이 가장 가까운 후보를 우선하는 휴리스틱이다."""

    name = "earliest_deadline_first"

    def select_action(
        self,
        simulator: SatelliteSchedulingSimulator,
        observation: SimulationObservation,
        rng: random.Random,
    ) -> int:
        del rng
        return _select_min(
            simulator,
            observation,
            key=lambda opportunity: (
                simulator.order_for(opportunity).request_end_sec,
                -simulator.order_for(opportunity).priority.score,
                opportunity.off_nadir_deg,
                opportunity.opportunity_id,
            ),
        )


class PriorityGreedyPolicy(BaselinePolicy):
    """미래 영향보다 현재 주문의 우선순위 점수를 먼저 보는 휴리스틱이다."""

    name = "priority_greedy"

    def select_action(
        self,
        simulator: SatelliteSchedulingSimulator,
        observation: SimulationObservation,
        rng: random.Random,
    ) -> int:
        del rng
        return _select_min(
            simulator,
            observation,
            key=lambda opportunity: (
                -simulator.order_for(opportunity).priority.score,
                simulator.order_for(opportunity).request_end_sec,
                opportunity.off_nadir_deg,
                opportunity.opportunity_id,
            ),
        )


class PriorityEfficiencyGreedyPolicy(BaselinePolicy):
    """우선순위를 촬영·기동 비용으로 나눈 효율이 높은 후보를 선택한다."""

    name = "priority_efficiency_greedy"

    def select_action(
        self,
        simulator: SatelliteSchedulingSimulator,
        observation: SimulationObservation,
        rng: random.Random,
    ) -> int:
        del rng

        def key(opportunity: Opportunity) -> tuple[float, float, float, str]:
            order = simulator.order_for(opportunity)
            cost = simulator.scenario.environment.imaging_duration_sec + simulator.slew_time_to(
                opportunity.required_roll_deg,
                opportunity.required_tilt_deg,
            )
            efficiency = order.priority.score / cost
            return (
                -efficiency,
                order.request_end_sec,
                opportunity.off_nadir_deg,
                opportunity.opportunity_id,
            )

        return _select_min(simulator, observation, key=key)


def evaluate_policy(
    policy: BaselinePolicy,
    scenario: Scenario,
    *,
    seed: int,
    max_steps: int = 1_000_000,
) -> EvaluationResult:
    """정책 하나를 동일 seed로 한 episode 실행해 비교 지표와 로그를 만든다."""

    simulator = SatelliteSchedulingSimulator(scenario)
    rng = random.Random(seed)
    decisions: list[DecisionLog] = []
    capture_count = 0
    priority_score = 0.0
    angle_bonus = 0.0
    missed_penalty = 0.0
    total_off_nadir = 0.0

    for step_index in range(max_steps):
        if simulator.terminated:
            break
        observation = simulator.observe()
        action = policy.select_action(simulator, observation, rng)
        if not 0 <= action < len(observation.action_mask) or not observation.action_mask[action]:
            raise ValueError(f"policy {policy.name} selected masked action {action}")

        selected_opportunity = simulator.opportunity_for_slot(action) if action else None
        result = simulator.step(action)
        if selected_opportunity is not None:
            capture_count += 1
            total_off_nadir += selected_opportunity.off_nadir_deg
        priority_score += result.breakdown.strip_base
        angle_bonus += result.breakdown.angle_bonus
        missed_penalty += result.breakdown.missed_penalty
        decisions.append(
            DecisionLog(
                step_index=step_index,
                time_sec=observation.current_time_sec,
                action=action,
                opportunity_id=result.selected_opportunity_id,
                valid_candidate_slots=tuple(_valid_slots(observation)),
                reward=result.reward,
                cumulative_return=simulator.cumulative_return,
            )
        )
    else:
        raise RuntimeError(f"policy evaluation exceeded max_steps={max_steps}")

    final_observation = simulator.observe()
    return EvaluationResult(
        policy_name=policy.name,
        scenario_id=scenario.scenario_id,
        seed=seed,
        steps=len(decisions),
        captures=capture_count,
        total_return=simulator.cumulative_return,
        priority_score=priority_score,
        angle_bonus=angle_bonus,
        missed_penalty=missed_penalty,
        completed_strips=final_observation.completed_strip_count,
        completed_orders=final_observation.completed_order_count,
        average_off_nadir_deg=(total_off_nadir / capture_count if capture_count else 0.0),
        decisions=tuple(decisions),
    )


def _valid_slots(observation: SimulationObservation) -> list[int]:
    return [
        candidate.slot
        for candidate in observation.candidates
        if observation.action_mask[candidate.slot]
    ]


def _select_min(
    simulator: SatelliteSchedulingSimulator,
    observation: SimulationObservation,
    *,
    key: Callable[[Opportunity], tuple[float, float, float, str]],
) -> int:
    valid_slots = _valid_slots(observation)
    if not valid_slots:
        return 0
    return min(
        valid_slots,
        key=lambda slot: key(simulator.opportunity_for_slot(slot)),
    )
