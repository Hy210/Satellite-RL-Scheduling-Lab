"""RL 라이브러리와 독립적으로 동작하는 결정론적 이벤트 기반 시뮬레이터다."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isclose

from rl_core.models import Opportunity, Order, Scenario


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """총 reward를 기본 점수, 각도 보너스와 미완료 패널티로 분해한다."""

    strip_base: float = 0.0
    angle_bonus: float = 0.0
    missed_penalty: float = 0.0

    @property
    def total(self) -> float:
        return self.strip_base + self.angle_bonus + self.missed_penalty


@dataclass(frozen=True, slots=True)
class CandidateView:
    """현재 action slot의 선택 가능 여부와 마스킹 사유를 노출한다."""

    slot: int
    opportunity_id: str
    valid: bool
    mask_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SimulationObservation:
    """현재 시뮬레이션 상태 중 정책과 디버깅에 필요한 요약 정보다."""

    current_time_sec: float
    current_roll_deg: float
    current_tilt_deg: float
    completed_strip_count: int
    completed_order_count: int
    candidates: tuple[CandidateView, ...]
    action_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """하나의 state-action 전이가 만든 다음 관측과 reward 결과다."""

    observation: SimulationObservation
    reward: float
    terminated: bool
    selected_opportunity_id: str | None
    breakdown: RewardBreakdown
    expired_order_ids: tuple[str, ...] = field(default_factory=tuple)


class SatelliteSchedulingSimulator:
    """Gymnasium과 웹에 의존하지 않는 순수 Python 환경 상태 머신이다."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self._orders = {item.order_id: item for item in scenario.orders}
        self._strips = {item.strip_id: item for item in scenario.strips}
        self._opportunities = {item.opportunity_id: item for item in scenario.opportunities}
        self._strips_by_order: dict[str, set[str]] = {order_id: set() for order_id in self._orders}
        for strip in scenario.strips:
            self._strips_by_order[strip.order_id].add(strip.strip_id)
        self.reset()

    def reset(self) -> SimulationObservation:
        """하루의 시작 시각과 초기 자세로 episode 상태를 초기화한다."""

        satellite = self.scenario.satellite
        self.current_time_sec = 0.0
        self.current_roll_deg = satellite.initial_roll_deg
        self.current_tilt_deg = satellite.initial_tilt_deg
        self.last_imaging_end_sec: float | None = None
        self.completed_strip_ids: set[str] = set()
        self.processed_opportunity_ids: set[str] = set()
        self.penalized_order_ids: set[str] = set()
        self.cumulative_return = 0.0
        self.terminated = False
        return self.observe()

    def observe(self) -> SimulationObservation:
        """현재 후보를 고정 slot에 매핑하고 action mask를 함께 반환한다."""

        candidates = self._current_candidates()
        views: list[CandidateView] = []
        action_mask = [True] + [False] * self.scenario.environment.max_candidates
        for slot, opportunity in enumerate(candidates, start=1):
            reasons = self.mask_reasons(opportunity)
            valid = not reasons
            action_mask[slot] = valid
            views.append(
                CandidateView(
                    slot=slot,
                    opportunity_id=opportunity.opportunity_id,
                    valid=valid,
                    mask_reasons=tuple(reasons),
                )
            )
        return SimulationObservation(
            current_time_sec=self.current_time_sec,
            current_roll_deg=self.current_roll_deg,
            current_tilt_deg=self.current_tilt_deg,
            completed_strip_count=len(self.completed_strip_ids),
            completed_order_count=sum(
                self._is_order_complete(order_id) for order_id in self._orders
            ),
            candidates=tuple(views),
            action_mask=tuple(action_mask),
        )

    def step(self, action: int) -> SimulationStep:
        """촬영 또는 skip 행동 하나를 적용하고 다음 이벤트까지 진행한다."""

        if self.terminated:
            raise RuntimeError("cannot step a terminated simulator")
        max_action = self.scenario.environment.max_candidates
        if not 0 <= action <= max_action:
            raise ValueError(f"action must be between 0 and {max_action}")

        candidates = self._current_candidates()
        selected: Opportunity | None = None
        strip_base = 0.0
        angle_bonus = 0.0
        if action > 0:
            if action > len(candidates):
                raise ValueError("action points to an empty candidate slot")
            selected = candidates[action - 1]
            reasons = self.mask_reasons(selected)
            if reasons:
                raise ValueError(f"masked action cannot be selected: {', '.join(reasons)}")
            strip_base, angle_bonus = self._capture(selected)

        # 같은 시각에는 한 번만 촬영할 수 있으므로 선택하지 않은 동시 후보도 만료한다.
        for opportunity in self._opportunities.values():
            if isclose(opportunity.capture_time_sec, self.current_time_sec, abs_tol=1e-9):
                self.processed_opportunity_ids.add(opportunity.opportunity_id)

        expired_orders, missed_penalty = self._advance_to_next_event()
        breakdown = RewardBreakdown(
            strip_base=strip_base,
            angle_bonus=angle_bonus,
            missed_penalty=missed_penalty,
        )
        self.cumulative_return += breakdown.total
        return SimulationStep(
            observation=self.observe(),
            reward=breakdown.total,
            terminated=self.terminated,
            selected_opportunity_id=(selected.opportunity_id if selected else None),
            breakdown=breakdown,
            expired_order_ids=tuple(expired_orders),
        )

    def mask_reasons(self, opportunity: Opportunity) -> list[str]:
        """현재 state에서 해당 촬영을 실행할 수 없는 하드 제약 사유를 계산한다."""

        reasons: list[str] = []
        order = self._orders[opportunity.order_id]
        satellite = self.scenario.satellite
        environment = self.scenario.environment
        capture_end = opportunity.capture_time_sec + environment.imaging_duration_sec

        if opportunity.opportunity_id in self.processed_opportunity_ids:
            reasons.append("opportunity_processed")
        if opportunity.strip_id in self.completed_strip_ids:
            reasons.append("strip_completed")
        if self._is_order_complete(order.order_id):
            reasons.append("order_completed")
        if order.order_id in self.penalized_order_ids:
            reasons.append("order_expired")
        if (
            opportunity.capture_time_sec < order.request_start_sec
            or capture_end > order.request_end_sec
        ):
            reasons.append("outside_order_period")
        if capture_end > opportunity.window_end_sec:
            reasons.append("outside_opportunity_window")
        if capture_end > environment.duration_sec:
            reasons.append("outside_episode")
        if abs(opportunity.required_roll_deg) > satellite.roll_limit_deg:
            reasons.append("roll_limit")
        if abs(opportunity.required_tilt_deg) > satellite.tilt_limit_deg:
            reasons.append("tilt_limit")
        if opportunity.off_nadir_deg > satellite.combined_off_nadir_limit_deg:
            reasons.append("combined_off_nadir_limit")
        if not order.allowed_roll_deg.contains(opportunity.required_roll_deg):
            reasons.append("order_roll_limit")
        if not order.allowed_tilt_deg.contains(opportunity.required_tilt_deg):
            reasons.append("order_tilt_limit")

        slew_time = self.slew_time_to(opportunity.required_roll_deg, opportunity.required_tilt_deg)
        if self.last_imaging_end_sec is None:
            earliest_start = slew_time
        else:
            earliest_start = self.last_imaging_end_sec + max(
                environment.minimum_interval_sec, slew_time
            )
        if opportunity.capture_time_sec + 1e-9 < earliest_start:
            reasons.append("insufficient_transition_time")
        return reasons

    def slew_time_to(self, target_roll_deg: float, target_tilt_deg: float) -> float:
        """두 축을 동시에 돌리지 않는다는 가정으로 순차 자세 전환시간을 계산한다."""

        delta_roll = abs(target_roll_deg - self.current_roll_deg)
        delta_tilt = abs(target_tilt_deg - self.current_tilt_deg)
        if isclose(delta_roll, 0.0, abs_tol=1e-12) and isclose(delta_tilt, 0.0, abs_tol=1e-12):
            return 0.0
        satellite = self.scenario.satellite
        return (
            delta_roll / satellite.roll_rate_deg_per_sec
            + delta_tilt / satellite.tilt_rate_deg_per_sec
            + satellite.settling_time_sec
        )

    def opportunity_for_slot(self, slot: int) -> Opportunity:
        """현재 action slot을 실제 촬영 기회로 변환한다."""

        candidates = self._current_candidates()
        if not 1 <= slot <= len(candidates):
            raise ValueError("slot does not reference a current opportunity")
        return candidates[slot - 1]

    def opportunity_by_id(self, opportunity_id: str) -> Opportunity:
        """로그와 재생 단계에서 ID만 남은 촬영 기회를 다시 조회한다."""

        return self._opportunities[opportunity_id]

    def order_for(self, opportunity: Opportunity) -> Order:
        """정책이 우선순위와 마감 정보를 비교할 수 있도록 소유 주문을 반환한다."""

        return self._orders[opportunity.order_id]

    def completion_ratio(self, order_id: str) -> float:
        """관측과 평가가 같은 정의를 사용하도록 주문의 strip 완료 비율을 반환한다."""

        return self._completion_ratio(self._orders[order_id])

    def is_order_complete(self, order_id: str) -> bool:
        """주문의 모든 strip 촬영 여부를 외부 wrapper에 안전하게 노출한다."""

        return self._is_order_complete(order_id)

    def future_opportunities_for_strip(self, strip_id: str) -> tuple[Opportunity, ...]:
        """현재 이후에 남은 촬영 기회를 시간순으로 반환해 미래 요약 관측에 사용한다."""

        opportunities = [
            item
            for item in self._opportunities.values()
            if item.strip_id == strip_id
            and item.opportunity_id not in self.processed_opportunity_ids
            and item.capture_time_sec >= self.current_time_sec
            and item.order_id not in self.penalized_order_ids
            and item.strip_id not in self.completed_strip_ids
        ]
        return tuple(
            sorted(opportunities, key=lambda item: (item.capture_time_sec, item.opportunity_id))
        )

    def _capture(self, opportunity: Opportunity) -> tuple[float, float]:
        # 주문 총점 P를 strip 수 N으로 나눠 주문 크기에 따른 보상 왜곡을 막는다.
        order = self._orders[opportunity.order_id]
        strip_count = len(self._strips_by_order[order.order_id])
        strip_base = order.priority.score / strip_count
        angle_quality = 1.0 - min(
            opportunity.off_nadir_deg / self.scenario.satellite.combined_off_nadir_limit_deg,
            1.0,
        )
        angle_bonus = strip_base * self.scenario.reward.angle_bonus_weight * angle_quality
        self.completed_strip_ids.add(opportunity.strip_id)
        self.current_roll_deg = opportunity.required_roll_deg
        self.current_tilt_deg = opportunity.required_tilt_deg
        self.last_imaging_end_sec = (
            opportunity.capture_time_sec + self.scenario.environment.imaging_duration_sec
        )
        return strip_base, angle_bonus

    def _advance_to_next_event(self) -> tuple[list[str], float]:
        # 매초 진행하지 않고 다음 촬영 후보 또는 주문 마감으로 시간을 건너뛴다.
        future_opportunity_times = [
            item.capture_time_sec
            for item in self._opportunities.values()
            if item.opportunity_id not in self.processed_opportunity_ids
            and item.capture_time_sec > self.current_time_sec + 1e-9
            and item.strip_id not in self.completed_strip_ids
            and item.order_id not in self.penalized_order_ids
        ]
        future_deadlines = [
            order.request_end_sec
            for order in self._orders.values()
            if order.order_id not in self.penalized_order_ids
            and not self._is_order_complete(order.order_id)
            and order.request_end_sec > self.current_time_sec + 1e-9
        ]
        next_events = future_opportunity_times + future_deadlines
        if self._all_strips_complete():
            self.terminated = True
            return [], 0.0

        if next_events:
            self.current_time_sec = min(next_events)
        else:
            self.current_time_sec = self.scenario.environment.duration_sec

        expired_orders, penalty = self._apply_due_deadlines()
        if self.current_time_sec >= self.scenario.environment.duration_sec or not next_events:
            self.terminated = True
        return expired_orders, penalty

    def _apply_due_deadlines(self) -> tuple[list[str], float]:
        # 주문별 패널티는 마감 시점에 한 번만 적용한다.
        expired: list[str] = []
        penalty = 0.0
        for order in self._orders.values():
            if order.order_id in self.penalized_order_ids or self._is_order_complete(
                order.order_id
            ):
                continue
            if order.request_end_sec <= self.current_time_sec + 1e-9:
                ratio = self._completion_ratio(order)
                penalty -= (
                    self.scenario.reward.missed_penalty_weight
                    * order.priority.score
                    * (1.0 - ratio)
                )
                self.penalized_order_ids.add(order.order_id)
                expired.append(order.order_id)
        return expired, penalty

    def _completion_ratio(self, order: Order) -> float:
        strip_ids = self._strips_by_order[order.order_id]
        return len(strip_ids & self.completed_strip_ids) / len(strip_ids)

    def _is_order_complete(self, order_id: str) -> bool:
        strip_ids = self._strips_by_order[order_id]
        return bool(strip_ids) and strip_ids <= self.completed_strip_ids

    def _all_strips_complete(self) -> bool:
        return len(self.completed_strip_ids) == len(self._strips)

    def _current_candidates(self) -> list[Opportunity]:
        candidates = [
            item
            for item in self._opportunities.values()
            if item.opportunity_id not in self.processed_opportunity_ids
            and isclose(item.capture_time_sec, self.current_time_sec, abs_tol=1e-9)
        ]
        candidates.sort(key=self._candidate_sort_key)
        return candidates[: self.scenario.environment.max_candidates]

    def _candidate_sort_key(self, opportunity: Opportunity) -> tuple[float, float, float, str]:
        order = self._orders[opportunity.order_id]
        return (
            -order.priority.score,
            order.request_end_sec,
            hypot(opportunity.required_roll_deg, opportunity.required_tilt_deg),
            opportunity.opportunity_id,
        )
