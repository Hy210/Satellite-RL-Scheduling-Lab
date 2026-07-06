"""결정론적 simulator를 Gymnasium과 Maskable PPO 규약에 연결하는 wrapper다."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from rl_core.models import OpportunityKind, Scenario
from rl_core.simulator import SatelliteSchedulingSimulator, SimulationObservation

Observation = dict[str, NDArray[np.float32]]

GLOBAL_FEATURES = (
    "current_time",
    "roll",
    "tilt",
    "has_previous_capture",
    "time_since_previous_capture",
    "completed_strip_ratio",
    "completed_order_ratio",
)

STRIP_FEATURES = (
    "priority",
    "deadline_remaining",
    "order_completion_ratio",
    "strip_completed",
    "next_opportunity_time",
    "remaining_opportunity_count",
    "best_future_off_nadir",
    "has_future_opportunity",
)

CANDIDATE_FEATURES = (
    "priority",
    "deadline_remaining",
    "required_roll",
    "required_tilt",
    "off_nadir",
    "slew_time",
    "order_completion_ratio",
    "opportunity_kind",
    "strip_completed",
    "is_action_valid",
)


class SatelliteSchedulingEnv(gym.Env[Observation, int]):
    """고정 크기 관측과 action mask를 제공하는 위성 스케줄링 Gym 환경이다."""

    metadata = {"render_modes": []}

    def __init__(self, scenario: Scenario):
        super().__init__()
        self.scenario = scenario
        self.simulator = SatelliteSchedulingSimulator(scenario)
        self._ordered_strips = tuple(
            sorted(scenario.strips, key=lambda item: (item.order_id, item.sequence, item.strip_id))
        )
        self._orders = {item.order_id: item for item in scenario.orders}
        max_strips = scenario.environment.max_strips
        max_candidates = scenario.environment.max_candidates

        # Dict 관측은 성격이 다른 전역·strip·후보 정보를 분리하고,
        # 향후 Maskable PPO의 MultiInputPolicy가 각 배열을 함께 처리하게 한다.
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(len(GLOBAL_FEATURES),),
                    dtype=np.float32,
                ),
                "strips": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(max_strips, len(STRIP_FEATURES)),
                    dtype=np.float32,
                ),
                "strip_presence": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(max_strips,),
                    dtype=np.float32,
                ),
                "candidates": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(max_candidates, len(CANDIDATE_FEATURES)),
                    dtype=np.float32,
                ),
                "candidate_presence": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(max_candidates,),
                    dtype=np.float32,
                ),
            }
        )
        # 0은 skip, 1~128은 현재 후보 slot이다.
        self.action_space = spaces.Discrete(max_candidates + 1)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Gymnasium seed 규약을 적용하고 simulator episode를 초기화한다."""

        super().reset(seed=seed)
        del options
        state = self.simulator.reset()
        return self._observation(state), self._base_info()

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """action을 적용하고 Gymnasium의 observation·reward·종료 형식으로 변환한다."""

        requested_action = int(action)
        masks = self.action_masks()
        action_was_masked = (
            not self.action_space.contains(requested_action) or not masks[requested_action]
        )
        # Action mask를 사용하지 않는 일반 환경 검사나 외부 호출이 잘못된 action을
        # 보내도 하드 제약을 실행하지 않도록 안전한 skip으로 변환한다.
        executed_action = 0 if action_was_masked else requested_action
        result = self.simulator.step(executed_action)
        info = self._base_info()
        info.update(
            {
                "requested_action": requested_action,
                "executed_action": executed_action,
                "action_was_masked": action_was_masked,
                "selected_opportunity_id": result.selected_opportunity_id,
                "expired_order_ids": result.expired_order_ids,
                "reward_breakdown": {
                    "strip_base": result.breakdown.strip_base,
                    "angle_bonus": result.breakdown.angle_bonus,
                    "missed_penalty": result.breakdown.missed_penalty,
                },
            }
        )
        return (
            self._observation(result.observation),
            float(result.reward),
            result.terminated,
            False,
            info,
        )

    def action_masks(self) -> NDArray[np.bool_]:
        """Maskable PPO가 실행 불가능한 action 확률을 0으로 만들 때 사용하는 mask다."""

        return np.asarray(self.simulator.observe().action_mask, dtype=np.bool_)

    def _observation(self, state: SimulationObservation) -> Observation:
        return {
            "global": self._global_features(state),
            "strips": self._strip_features(),
            "strip_presence": self._strip_presence(),
            "candidates": self._candidate_features(state),
            "candidate_presence": self._candidate_presence(state),
        }

    def _global_features(self, state: SimulationObservation) -> NDArray[np.float32]:
        duration = self.scenario.environment.duration_sec
        satellite = self.scenario.satellite
        last_end = self.simulator.last_imaging_end_sec
        completed_strip_ratio = state.completed_strip_count / len(self.scenario.strips)
        completed_order_ratio = state.completed_order_count / len(self.scenario.orders)
        values = (
            _clip01(state.current_time_sec / duration),
            _clip_signed(state.current_roll_deg / satellite.roll_limit_deg),
            _clip_signed(state.current_tilt_deg / satellite.tilt_limit_deg),
            float(last_end is not None),
            _clip01((state.current_time_sec - last_end) / duration)
            if last_end is not None
            else 0.0,
            _clip01(completed_strip_ratio),
            _clip01(completed_order_ratio),
        )
        return np.asarray(values, dtype=np.float32)

    def _strip_features(self) -> NDArray[np.float32]:
        rows = np.zeros(
            (self.scenario.environment.max_strips, len(STRIP_FEATURES)),
            dtype=np.float32,
        )
        duration = self.scenario.environment.duration_sec
        max_future_count = max(1, len(self.scenario.passes) * 3)
        angle_limit = self.scenario.satellite.combined_off_nadir_limit_deg
        for index, strip in enumerate(self._ordered_strips):
            order = self._orders[strip.order_id]
            future = self.simulator.future_opportunities_for_strip(strip.strip_id)
            rows[index] = np.asarray(
                (
                    order.priority.score / 5.0,
                    _clip01((order.request_end_sec - self.simulator.current_time_sec) / duration),
                    self.simulator.completion_ratio(order.order_id),
                    float(strip.strip_id in self.simulator.completed_strip_ids),
                    _clip01(
                        (future[0].capture_time_sec - self.simulator.current_time_sec) / duration
                    )
                    if future
                    else 1.0,
                    _clip01(len(future) / max_future_count),
                    _clip01(min(item.off_nadir_deg for item in future) / angle_limit)
                    if future
                    else 1.0,
                    float(bool(future)),
                ),
                dtype=np.float32,
            )
        return rows

    def _strip_presence(self) -> NDArray[np.float32]:
        presence = np.zeros(self.scenario.environment.max_strips, dtype=np.float32)
        presence[: len(self._ordered_strips)] = 1.0
        return presence

    def _candidate_features(self, state: SimulationObservation) -> NDArray[np.float32]:
        rows = np.zeros(
            (self.scenario.environment.max_candidates, len(CANDIDATE_FEATURES)),
            dtype=np.float32,
        )
        duration = self.scenario.environment.duration_sec
        satellite = self.scenario.satellite
        max_slew = (
            2.0 * satellite.roll_limit_deg / satellite.roll_rate_deg_per_sec
            + 2.0 * satellite.tilt_limit_deg / satellite.tilt_rate_deg_per_sec
            + satellite.settling_time_sec
        )
        for candidate in state.candidates:
            opportunity = self.simulator.opportunity_for_slot(candidate.slot)
            order = self.simulator.order_for(opportunity)
            rows[candidate.slot - 1] = np.asarray(
                (
                    order.priority.score / 5.0,
                    _clip01((order.request_end_sec - state.current_time_sec) / duration),
                    _clip_signed(opportunity.required_roll_deg / satellite.roll_limit_deg),
                    _clip_signed(opportunity.required_tilt_deg / satellite.tilt_limit_deg),
                    _clip01(opportunity.off_nadir_deg / satellite.combined_off_nadir_limit_deg),
                    _clip01(
                        self.simulator.slew_time_to(
                            opportunity.required_roll_deg,
                            opportunity.required_tilt_deg,
                        )
                        / max_slew
                    ),
                    self.simulator.completion_ratio(order.order_id),
                    _kind_value(opportunity.kind),
                    float(opportunity.strip_id in self.simulator.completed_strip_ids),
                    float(candidate.valid),
                ),
                dtype=np.float32,
            )
        return rows

    def _candidate_presence(self, state: SimulationObservation) -> NDArray[np.float32]:
        presence = np.zeros(self.scenario.environment.max_candidates, dtype=np.float32)
        presence[: len(state.candidates)] = 1.0
        return presence

    def _base_info(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "cumulative_return": self.simulator.cumulative_return,
            "completed_strip_count": len(self.simulator.completed_strip_ids),
            "completed_order_count": sum(
                self.simulator.is_order_complete(order.order_id) for order in self.scenario.orders
            ),
        }


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _clip_signed(value: float) -> float:
    return min(max(value, -1.0), 1.0)


def _kind_value(kind: OpportunityKind) -> float:
    return {
        OpportunityKind.EARLY: 0.0,
        OpportunityKind.MIN_OFF_NADIR: 0.5,
        OpportunityKind.LATE: 1.0,
    }[kind]
