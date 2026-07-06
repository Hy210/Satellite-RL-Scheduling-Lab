"""생성기, 시뮬레이터와 API가 함께 사용하는 검증된 데이터 계약이다."""

from __future__ import annotations

from enum import StrEnum
from math import hypot
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """오타나 잘못된 입력을 조기에 찾도록 알 수 없는 필드를 거부하는 공통 모델이다."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Priority(StrEnum):
    """주문의 업무 우선순위이며 score는 보상 함수의 기본 점수다."""

    RED = "red"
    BLUE = "blue"
    BACKGROUND = "background"

    @property
    def score(self) -> float:
        return {
            Priority.RED: 5.0,
            Priority.BLUE: 3.0,
            Priority.BACKGROUND: 1.0,
        }[self]


class OpportunityKind(StrEnum):
    """연속 접근 구간을 최대 세 개의 대표 촬영 시점으로 이산화한 종류다."""

    EARLY = "early"
    MIN_OFF_NADIR = "min_off_nadir"
    LATE = "late"


class RunStatus(StrEnum):
    """향후 학습 worker와 웹이 공유할 실행 생명주기 상태다."""

    QUEUED = "queued"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class Rectangle(StrictModel):
    """날짜변경선을 넘지 않는 초기 프로토타입용 직사각형 위경도 영역이다."""

    min_lat: float = Field(ge=-90.0, le=90.0)
    min_lon: float = Field(ge=-180.0, le=180.0)
    max_lat: float = Field(ge=-90.0, le=90.0)
    max_lon: float = Field(ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be smaller than max_lat")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be smaller than max_lon")
        return self


class AngleRange(StrictModel):
    """주문 또는 위성이 허용하는 최소·최대 자세각 범위다."""

    minimum: float = Field(ge=-180.0, le=180.0)
    maximum: float = Field(ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("angle minimum must not exceed maximum")
        return self

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


class SatelliteConfig(StrictModel):
    """초기 자세, 물리적 자세 한계와 축별 기동 속도를 정의한다."""

    satellite_id: str = Field(min_length=1)
    initial_roll_deg: float = 0.0
    initial_tilt_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_limit_deg: float = Field(default=30.0, gt=0.0, le=90.0)
    tilt_limit_deg: float = Field(default=30.0, gt=0.0, le=90.0)
    combined_off_nadir_limit_deg: float = Field(default=30.0, gt=0.0, le=90.0)
    roll_rate_deg_per_sec: float = Field(default=5.0, gt=0.0)
    tilt_rate_deg_per_sec: float = Field(default=5.0, gt=0.0)
    settling_time_sec: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def validate_initial_attitude(self) -> Self:
        if abs(self.initial_roll_deg) > self.roll_limit_deg:
            raise ValueError("initial roll exceeds satellite limit")
        if abs(self.initial_tilt_deg) > self.tilt_limit_deg:
            raise ValueError("initial tilt exceeds satellite limit")
        if hypot(self.initial_roll_deg, self.initial_tilt_deg) > self.combined_off_nadir_limit_deg:
            raise ValueError("initial attitude exceeds combined off-nadir limit")
        if self.pitch_deg != 0.0:
            raise ValueError("pitch must be fixed to 0 in the initial prototype")
        return self


class EnvironmentConfig(StrictModel):
    """episode 길이와 촬영시간 등 환경 전체에 공통인 파라미터다."""

    duration_sec: float = Field(default=86_400.0, gt=0.0)
    imaging_duration_sec: float = Field(default=5.0, gt=0.0)
    minimum_interval_sec: float = Field(default=5.0, ge=0.0)
    max_strips: int = Field(default=2_000, gt=0)
    max_candidates: int = Field(default=128, gt=0)


class RewardConfig(StrictModel):
    """보상 shaping의 강도를 조절하며 기본 우선순위 점수와 분리한다."""

    angle_bonus_weight: float = Field(default=0.1, ge=0.0)
    missed_penalty_weight: float = Field(default=0.5, ge=0.0)


class OrbitPass(StrictModel):
    """단일 위성이 하루 동안 지나는 하나의 연속 궤도 구간이다."""

    pass_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    start_time_sec: float = Field(ge=0.0)
    end_time_sec: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if self.start_time_sec >= self.end_time_sec:
            raise ValueError("pass start must be before pass end")
        return self


class Order(StrictModel):
    """공통 우선순위와 요구 기간을 가지는 지리적 촬영 주문이다."""

    order_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    priority: Priority
    request_start_sec: float = Field(ge=0.0)
    request_end_sec: float = Field(gt=0.0)
    geometry: Rectangle
    allowed_roll_deg: AngleRange = Field(
        default_factory=lambda: AngleRange(minimum=-30.0, maximum=30.0)
    )
    allowed_tilt_deg: AngleRange = Field(
        default_factory=lambda: AngleRange(minimum=-30.0, maximum=30.0)
    )

    @model_validator(mode="after")
    def validate_period(self) -> Self:
        if self.request_start_sec >= self.request_end_sec:
            raise ValueError("order request start must be before request end")
        return self


class Strip(StrictModel):
    """주문 영역을 나눈 위성의 1회 촬영 단위다."""

    strip_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    geometry: Rectangle


class Opportunity(StrictModel):
    """특정 strip을 특정 pass·시각·자세로 촬영할 수 있는 후보 행동이다."""

    opportunity_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    strip_id: str = Field(min_length=1)
    pass_id: str = Field(min_length=1)
    kind: OpportunityKind
    window_start_sec: float = Field(ge=0.0)
    window_end_sec: float = Field(gt=0.0)
    capture_time_sec: float = Field(ge=0.0)
    required_roll_deg: float
    required_tilt_deg: float
    required_pitch_deg: float = 0.0

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.window_start_sec >= self.window_end_sec:
            raise ValueError("opportunity window start must be before its end")
        if not self.window_start_sec <= self.capture_time_sec < self.window_end_sec:
            raise ValueError("capture time must lie inside the opportunity window")
        if self.required_pitch_deg != 0.0:
            raise ValueError("pitch must be fixed to 0 in the initial prototype")
        return self

    @property
    def off_nadir_deg(self) -> float:
        """roll과 tilt를 하나의 촬영 각도 선호도로 단순화한 근사값이다."""

        return hypot(self.required_roll_deg, self.required_tilt_deg)


class Scenario(StrictModel):
    """한 episode를 재현하는 데 필요한 모든 고정 입력을 묶은 최상위 모델이다."""

    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    seed: int
    satellite: SatelliteConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    passes: list[OrbitPass]
    orders: list[Order]
    strips: list[Strip]
    opportunities: list[Opportunity]

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """ID 참조와 시간 구간처럼 실행 전에 알 수 있는 구조적 오류를 검사한다."""

        self._require_unique("pass", [item.pass_id for item in self.passes])
        self._require_unique("order", [item.order_id for item in self.orders])
        self._require_unique("strip", [item.strip_id for item in self.strips])
        self._require_unique("opportunity", [item.opportunity_id for item in self.opportunities])

        if not self.orders:
            raise ValueError("scenario must contain at least one order")
        if not self.strips:
            raise ValueError("scenario must contain at least one strip")
        if len(self.strips) > self.environment.max_strips:
            raise ValueError("scenario exceeds max_strips")

        pass_by_id = {item.pass_id: item for item in self.passes}
        order_by_id = {item.order_id: item for item in self.orders}
        strip_by_id = {item.strip_id: item for item in self.strips}

        strip_counts = {order_id: 0 for order_id in order_by_id}
        for strip in self.strips:
            if strip.order_id not in order_by_id:
                raise ValueError(f"strip {strip.strip_id} references an unknown order")
            strip_counts[strip.order_id] += 1
        empty_orders = [order_id for order_id, count in strip_counts.items() if count == 0]
        if empty_orders:
            raise ValueError(f"orders must contain at least one strip: {empty_orders}")

        for orbit_pass in self.passes:
            if orbit_pass.end_time_sec > self.environment.duration_sec:
                raise ValueError(f"pass {orbit_pass.pass_id} exceeds scenario duration")

        for order in self.orders:
            if order.request_end_sec > self.environment.duration_sec:
                raise ValueError(f"order {order.order_id} exceeds scenario duration")

        # 자세 전환 가능 여부처럼 state에 따라 달라지는 조건은 여기서 거부하지 않고
        # simulator의 action mask에서 판정한다.
        for opportunity in self.opportunities:
            referenced_pass = pass_by_id.get(opportunity.pass_id)
            referenced_order = order_by_id.get(opportunity.order_id)
            referenced_strip = strip_by_id.get(opportunity.strip_id)
            if referenced_pass is None or referenced_order is None or referenced_strip is None:
                raise ValueError(
                    f"opportunity {opportunity.opportunity_id} has an unknown reference"
                )
            if referenced_strip.order_id != opportunity.order_id:
                raise ValueError("opportunity order and strip owner do not match")
            capture_end = opportunity.capture_time_sec + self.environment.imaging_duration_sec
            if opportunity.window_start_sec < referenced_pass.start_time_sec:
                raise ValueError("opportunity window starts before its pass")
            if opportunity.window_end_sec > referenced_pass.end_time_sec:
                raise ValueError("opportunity window ends after its pass")
            if opportunity.capture_time_sec < referenced_order.request_start_sec:
                raise ValueError("opportunity capture starts before its order period")
            if capture_end > referenced_order.request_end_sec:
                raise ValueError("opportunity capture ends after its order period")
            if capture_end > opportunity.window_end_sec:
                raise ValueError("opportunity cannot contain the full imaging duration")
            if capture_end > self.environment.duration_sec:
                raise ValueError("opportunity capture exceeds scenario duration")

        return self

    @staticmethod
    def _require_unique(kind: str, values: list[str]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {kind} ID")

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, value: str) -> Scenario:
        return cls.model_validate_json(value)

    def save(self, path: Path) -> None:
        """재현 가능한 시나리오를 UTF-8 JSON 파일로 저장한다."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Scenario:
        """JSON 시나리오를 읽고 모든 구조 검증을 다시 수행한다."""

        return cls.from_json(path.read_text(encoding="utf-8"))


class TrainingRun(StrictModel):
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    seed: int
    total_timesteps: int = Field(gt=0)
    status: RunStatus = RunStatus.QUEUED
    artifact_directory: str | None = None
    error_message: str | None = None


class EvaluationRun(StrictModel):
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    seed: int
    status: RunStatus = RunStatus.QUEUED
    source_training_run_id: str | None = None
    result_path: str | None = None
    error_message: str | None = None
