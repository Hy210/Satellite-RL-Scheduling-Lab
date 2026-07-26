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


class GeoPoint(StrictModel):
    """지도 geometry를 구성하는 위경도 꼭짓점이다."""

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)


class Polygon(StrictModel):
    """날짜변경선을 넘지 않는 단순 polygon geometry다."""

    vertices: list[GeoPoint] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_area(self) -> Self:
        if abs(self.signed_area) < 1e-12:
            raise ValueError("polygon area must be non-zero")
        return self

    @property
    def signed_area(self) -> float:
        total = 0.0
        for index, vertex in enumerate(self.vertices):
            next_vertex = self.vertices[(index + 1) % len(self.vertices)]
            total += vertex.lon * next_vertex.lat - next_vertex.lon * vertex.lat
        return total / 2.0

    @property
    def min_lat(self) -> float:
        return min(vertex.lat for vertex in self.vertices)

    @property
    def max_lat(self) -> float:
        return max(vertex.lat for vertex in self.vertices)

    @property
    def min_lon(self) -> float:
        return min(vertex.lon for vertex in self.vertices)

    @property
    def max_lon(self) -> float:
        return max(vertex.lon for vertex in self.vertices)


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


class GroundTrackPoint(StrictModel):
    """pass 중 위성 지상점이 특정 시각에 위치한 가상 또는 실제 좌표다."""

    point_id: str = Field(min_length=1)
    pass_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    time_sec: float = Field(ge=0.0)
    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)


class FootprintSample(StrictModel):
    """특정 시각의 센서 지상 커버리지를 단순 직사각형으로 근사한 샘플이다."""

    footprint_id: str = Field(min_length=1)
    pass_id: str = Field(min_length=1)
    ground_track_point_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    time_sec: float = Field(ge=0.0)
    center_latitude_deg: float = Field(ge=-90.0, le=90.0)
    center_longitude_deg: float = Field(ge=-180.0, le=180.0)
    geometry: Polygon


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
    geometry: Polygon


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
    source_access_window_id: str | None = None

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


class AccessWindow(StrictModel):
    """footprint와 strip이 연속적으로 교차해 촬영 후보를 만들 수 있는 구간이다."""

    access_window_id: str = Field(min_length=1)
    pass_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    strip_id: str = Field(min_length=1)
    window_start_sec: float = Field(ge=0.0)
    window_end_sec: float = Field(gt=0.0)
    min_off_nadir_time_sec: float = Field(ge=0.0)
    footprint_ids: list[str] = Field(default_factory=list)
    opportunity_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.window_start_sec >= self.window_end_sec:
            raise ValueError("access window start must be before its end")
        if not self.window_start_sec <= self.min_off_nadir_time_sec < self.window_end_sec:
            raise ValueError("minimum off-nadir time must lie inside the access window")
        return self


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
    ground_track_points: list[GroundTrackPoint] = Field(default_factory=list)
    footprint_samples: list[FootprintSample] = Field(default_factory=list)
    access_windows: list[AccessWindow] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """ID 참조와 시간 구간처럼 실행 전에 알 수 있는 구조적 오류를 검사한다."""

        self._require_unique("pass", [item.pass_id for item in self.passes])
        self._require_unique("order", [item.order_id for item in self.orders])
        self._require_unique("strip", [item.strip_id for item in self.strips])
        self._require_unique("opportunity", [item.opportunity_id for item in self.opportunities])
        self._require_unique(
            "ground track point", [item.point_id for item in self.ground_track_points]
        )
        self._require_unique(
            "footprint sample", [item.footprint_id for item in self.footprint_samples]
        )
        self._require_unique(
            "access window", [item.access_window_id for item in self.access_windows]
        )

        if not self.orders:
            raise ValueError("scenario must contain at least one order")
        if not self.strips:
            raise ValueError("scenario must contain at least one strip")
        if len(self.strips) > self.environment.max_strips:
            raise ValueError("scenario exceeds max_strips")

        pass_by_id = {item.pass_id: item for item in self.passes}
        order_by_id = {item.order_id: item for item in self.orders}
        strip_by_id = {item.strip_id: item for item in self.strips}
        ground_track_by_id = {item.point_id: item for item in self.ground_track_points}
        footprint_by_id = {item.footprint_id: item for item in self.footprint_samples}
        access_window_by_id = {item.access_window_id: item for item in self.access_windows}

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

        for point in self.ground_track_points:
            referenced_pass = pass_by_id.get(point.pass_id)
            if referenced_pass is None:
                raise ValueError(f"ground track point {point.point_id} references an unknown pass")
            if not referenced_pass.start_time_sec <= point.time_sec <= referenced_pass.end_time_sec:
                raise ValueError("ground track point time must lie inside its pass")

        for footprint in self.footprint_samples:
            referenced_pass = pass_by_id.get(footprint.pass_id)
            referenced_point = ground_track_by_id.get(footprint.ground_track_point_id)
            if referenced_pass is None or referenced_point is None:
                raise ValueError(
                    f"footprint sample {footprint.footprint_id} has an unknown reference"
                )
            if referenced_point.pass_id != footprint.pass_id:
                raise ValueError("footprint sample and ground track point pass do not match")
            if footprint.time_sec != referenced_point.time_sec:
                raise ValueError("footprint sample time must match its ground track point")
            footprint_inside_pass = (
                referenced_pass.start_time_sec <= footprint.time_sec <= referenced_pass.end_time_sec
            )
            if not footprint_inside_pass:
                raise ValueError("footprint sample time must lie inside its pass")

        for access_window in self.access_windows:
            referenced_pass = pass_by_id.get(access_window.pass_id)
            referenced_order = order_by_id.get(access_window.order_id)
            referenced_strip = strip_by_id.get(access_window.strip_id)
            if referenced_pass is None or referenced_order is None or referenced_strip is None:
                raise ValueError(
                    f"access window {access_window.access_window_id} has an unknown reference"
                )
            if referenced_strip.order_id != access_window.order_id:
                raise ValueError("access window order and strip owner do not match")
            if access_window.window_start_sec < referenced_pass.start_time_sec:
                raise ValueError("access window starts before its pass")
            if access_window.window_end_sec > referenced_pass.end_time_sec:
                raise ValueError("access window ends after its pass")
            if access_window.window_start_sec < referenced_order.request_start_sec:
                raise ValueError("access window starts before its order period")
            if access_window.window_end_sec > referenced_order.request_end_sec:
                raise ValueError("access window ends after its order period")
            for footprint_id in access_window.footprint_ids:
                referenced_footprint = footprint_by_id.get(footprint_id)
                if referenced_footprint is None:
                    raise ValueError("access window references an unknown footprint sample")
                if referenced_footprint.pass_id != access_window.pass_id:
                    raise ValueError("access window footprint pass does not match")

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
            if opportunity.source_access_window_id is not None:
                referenced_window = access_window_by_id.get(opportunity.source_access_window_id)
                if referenced_window is None:
                    raise ValueError("opportunity references an unknown access window")
                if (
                    referenced_window.pass_id != opportunity.pass_id
                    or referenced_window.order_id != opportunity.order_id
                    or referenced_window.strip_id != opportunity.strip_id
                ):
                    raise ValueError("opportunity and access window references do not match")
                if opportunity.opportunity_id not in referenced_window.opportunity_ids:
                    raise ValueError("access window does not list the derived opportunity")
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
            if opportunity.source_access_window_id is not None:
                referenced_window = access_window_by_id[opportunity.source_access_window_id]
                if opportunity.window_start_sec < referenced_window.window_start_sec:
                    raise ValueError("opportunity starts before its source access window")
                if opportunity.window_end_sec > referenced_window.window_end_sec:
                    raise ValueError("opportunity ends after its source access window")

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
    """학습 실행 하나를 추적하기 위한 최소 메타데이터다."""

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    seed: int
    total_timesteps: int = Field(gt=0)
    status: RunStatus = RunStatus.QUEUED
    artifact_directory: str | None = None
    error_message: str | None = None


class EvaluationRun(StrictModel):
    """학습된 정책 또는 기준 정책의 평가 실행 메타데이터다."""

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    seed: int
    status: RunStatus = RunStatus.QUEUED
    source_training_run_id: str | None = None
    result_path: str | None = None
    error_message: str | None = None


class EvaluationSummary(StrictModel):
    """정책 종류와 무관하게 결과 화면이 읽는 평가 요약 artifact다."""

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
    replay_path: str


class ReplayState(StrictModel):
    """재생 로그의 한 시점에서 화면과 분석에 필요한 상태 요약이다."""

    time_sec: float
    roll_deg: float
    tilt_deg: float
    completed_strips: int
    completed_orders: int


class ReplayCandidate(StrictModel):
    """선택 당시 후보 action과 action mask 판정 사유를 함께 저장한다."""

    slot: int = Field(ge=1)
    opportunity_id: str
    order_id: str
    strip_id: str
    pass_id: str
    capture_time_sec: float
    required_roll_deg: float
    required_tilt_deg: float
    valid: bool
    mask_reasons: list[str] = Field(default_factory=list)


class ReplayRewardBreakdown(StrictModel):
    """재생 step의 reward를 구성 요소별로 복원하기 위한 값이다."""

    strip_base: float = 0.0
    angle_bonus: float = 0.0
    missed_penalty: float = 0.0
    total: float = 0.0


class ReplayStep(StrictModel):
    """episode 재생에서 한 번의 선택과 그 전후 상태를 나타낸다."""

    step_index: int = Field(ge=0)
    state_before: ReplayState
    candidates: list[ReplayCandidate] = Field(default_factory=list)
    action: int = Field(ge=0)
    selected_opportunity_id: str | None = None
    expired_order_ids: list[str] = Field(default_factory=list)
    reward: float
    reward_breakdown: ReplayRewardBreakdown
    cumulative_return: float
    state_after: ReplayState


class ReplayCapture(StrictModel):
    """최종 촬영 스케줄을 빠르게 표시하기 위한 촬영 action 요약이다."""

    step_index: int = Field(ge=0)
    opportunity_id: str
    order_id: str
    strip_id: str
    pass_id: str
    capture_time_sec: float
    roll_deg: float
    tilt_deg: float
    reward: float


class EpisodeReplay(StrictModel):
    """저장된 로그만으로 평가 episode를 재생하기 위한 공통 결과 형식이다."""

    policy_name: str
    scenario_id: str
    seed: int
    steps: list[ReplayStep] = Field(default_factory=list)
    schedule: list[ReplayCapture] = Field(default_factory=list)
    total_return: float
    completed_strips: int
    completed_orders: int


class PolicyComparisonEntry(StrictModel):
    """동일 시나리오에서 정책 하나의 성능과 원본 실행을 비교하기 위한 요약이다."""

    evaluation_run_id: str | None = None
    policy_name: str
    scenario_id: str
    seed: int
    total_return: float
    priority_score: float
    angle_bonus: float
    missed_penalty: float
    completed_strips: int
    completed_orders: int
    captures: int
    steps: int
    replay_path: str | None = None


class PolicyComparison(StrictModel):
    """여러 정책의 평가 결과를 한 파일에서 비교하기 위한 artifact 계약이다."""

    scenario_id: str
    entries: list[PolicyComparisonEntry] = Field(min_length=1)
    best_policy_name: str

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        if any(entry.scenario_id != self.scenario_id for entry in self.entries):
            raise ValueError("all policy comparison entries must share scenario_id")
        if self.best_policy_name not in {entry.policy_name for entry in self.entries}:
            raise ValueError("best_policy_name must reference an entry")
        return self


class PolicyComparisonRun(StrictModel):
    """동일 scenario·seed 결과 집합을 고정해 재현 가능한 비교 artifact로 연결한다."""

    comparison_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    seed: int
    evaluation_run_ids: list[str] = Field(min_length=1)
    artifact_path: str = Field(min_length=1)


class OptimizationBaselineResult(StrictModel):
    """최적화 solver가 만든 기준해와 simulator replay 검증 결과를 묶은 artifact다."""

    solver_name: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    seed: int
    status: str = Field(min_length=1)
    objective_value: float | None = None
    best_objective_bound: float | None = None
    optimality_gap: float | None = Field(default=None, ge=0.0)
    time_limit_sec: float = Field(gt=0.0)
    selected_opportunity_ids: list[str] = Field(default_factory=list)
    replay: EpisodeReplay


class MaskablePPOTrainingConfig(StrictModel):
    """Maskable PPO 학습을 재현하기 위한 하이퍼파라미터와 저장 설정이다."""

    total_timesteps: int = Field(default=1_024, gt=0)
    learning_seed: int
    evaluation_seed: int
    n_steps: int = Field(default=64, gt=0)
    batch_size: int = Field(default=32, gt=0)
    n_epochs: int = Field(default=2, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    checkpoint_interval: int = Field(default=512, gt=0)
    evaluation_interval: int = Field(default=512, gt=0)
    deterministic_eval: bool = True
    artifact_root: Path = Path("data/runs")

    @model_validator(mode="after")
    def validate_batch_shape(self) -> Self:
        if self.batch_size > self.n_steps:
            raise ValueError(
                "batch_size must not exceed n_steps for the initial single-env trainer"
            )
        return self
