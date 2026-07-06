"""동일 seed에서 재현 가능한 학습용 가상 시나리오를 생성한다."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import hypot

from rl_core.models import (
    AngleRange,
    EnvironmentConfig,
    Opportunity,
    OpportunityKind,
    OrbitPass,
    Order,
    Priority,
    Rectangle,
    RewardConfig,
    SatelliteConfig,
    Scenario,
    Strip,
)


@dataclass(frozen=True, slots=True)
class ScenarioScale:
    """환경을 작은 문제부터 확장하기 위한 시나리오 규모 설정이다."""

    order_count: int
    pass_count: int
    max_strips_per_order: int
    max_accesses_per_strip: int


SCALES: dict[str, ScenarioScale] = {
    "tiny": ScenarioScale(5, 3, 3, 2),
    "small": ScenarioScale(20, 10, 6, 2),
    "full": ScenarioScale(100, 30, 10, 3),
}


def generate_scenario(seed: int, size: str = "tiny") -> Scenario:
    """전 세계에 분포한 재현 가능한 가상 주문과 촬영 기회를 생성한다."""

    try:
        scale = SCALES[size]
    except KeyError as exc:
        raise ValueError(f"unknown scenario size: {size}") from exc

    rng = random.Random(seed)
    environment = EnvironmentConfig()
    passes = _generate_passes(scale.pass_count, environment.duration_sec)
    orders: list[Order] = []
    strips: list[Strip] = []
    opportunities: list[Opportunity] = []

    for order_index in range(scale.order_count):
        strip_count = rng.randint(1, scale.max_strips_per_order)
        order_id = f"order-{order_index:03d}"
        strip_width = 0.12
        strip_height = 0.18
        min_lat = rng.uniform(-60.0, 60.0 - strip_height)
        min_lon = rng.uniform(-170.0, 170.0 - strip_width * strip_count)
        request_start = rng.uniform(0.0, environment.duration_sec * 0.15)
        request_end = rng.uniform(environment.duration_sec * 0.85, environment.duration_sec)
        priority = rng.choices(
            [Priority.RED, Priority.BLUE, Priority.BACKGROUND],
            weights=[0.2, 0.35, 0.45],
            k=1,
        )[0]
        order_geometry = Rectangle(
            min_lat=min_lat,
            min_lon=min_lon,
            max_lat=min_lat + strip_height,
            max_lon=min_lon + strip_width * strip_count,
        )
        orders.append(
            Order(
                order_id=order_id,
                name=f"Synthetic order {order_index + 1}",
                priority=priority,
                request_start_sec=request_start,
                request_end_sec=request_end,
                geometry=order_geometry,
                allowed_roll_deg=AngleRange(minimum=-30.0, maximum=30.0),
                allowed_tilt_deg=AngleRange(minimum=-30.0, maximum=30.0),
            )
        )

        # 주문 요구 기간 안에 완전히 들어가는 pass를 우선 사용한다.
        valid_passes = [
            item
            for item in passes
            if item.start_time_sec >= request_start and item.end_time_sec <= request_end
        ]
        if not valid_passes:
            valid_passes = [
                item
                for item in passes
                if item.end_time_sec > request_start and item.start_time_sec < request_end
            ]

        for strip_index in range(strip_count):
            strip_id = f"{order_id}-strip-{strip_index:02d}"
            strip = Strip(
                strip_id=strip_id,
                order_id=order_id,
                sequence=strip_index,
                geometry=Rectangle(
                    min_lat=min_lat,
                    min_lon=min_lon + strip_width * strip_index,
                    max_lat=min_lat + strip_height,
                    max_lon=min_lon + strip_width * (strip_index + 1),
                ),
            )
            strips.append(strip)

            access_count = min(rng.randint(1, scale.max_accesses_per_strip), len(valid_passes))
            selected_passes = rng.sample(valid_passes, k=access_count)
            for access_index, orbit_pass in enumerate(selected_passes):
                opportunities.extend(
                    _generate_access_opportunities(
                        rng=rng,
                        order_id=order_id,
                        strip_id=strip_id,
                        orbit_pass=orbit_pass,
                        access_index=access_index,
                        imaging_duration=environment.imaging_duration_sec,
                        request_start=request_start,
                        request_end=request_end,
                    )
                )

    opportunities.sort(key=lambda item: (item.capture_time_sec, item.opportunity_id))
    return Scenario(
        scenario_id=f"synthetic-{size}-{seed}",
        name=f"Synthetic {size} scenario (seed={seed})",
        seed=seed,
        satellite=SatelliteConfig(satellite_id="satellite-001"),
        environment=environment,
        reward=RewardConfig(),
        passes=passes,
        orders=orders,
        strips=strips,
        opportunities=opportunities,
    )


def _generate_passes(pass_count: int, duration: float) -> list[OrbitPass]:
    spacing = duration / pass_count
    pass_duration = min(1_200.0, spacing * 0.75)
    return [
        OrbitPass(
            pass_id=f"pass-{index:02d}",
            sequence=index,
            start_time_sec=index * spacing + (spacing - pass_duration) / 2.0,
            end_time_sec=index * spacing + (spacing + pass_duration) / 2.0,
        )
        for index in range(pass_count)
    ]


def _generate_access_opportunities(
    *,
    rng: random.Random,
    order_id: str,
    strip_id: str,
    orbit_pass: OrbitPass,
    access_index: int,
    imaging_duration: float,
    request_start: float,
    request_end: float,
) -> list[Opportunity]:
    window_duration = float(rng.choice((40, 60, 80)))
    earliest = max(orbit_pass.start_time_sec, request_start)
    latest = min(orbit_pass.end_time_sec, request_end)
    available = latest - earliest
    if available < window_duration:
        window_duration = available
    if window_duration < imaging_duration:
        return []

    max_offset = max(0.0, available - window_duration)
    # 연속 난수 시각은 동시 후보를 거의 만들지 않는다. 10초 grid는 여러 후보가
    # 경쟁하는 실제 의사결정 state를 만들기 위한 가상 시나리오용 단순화다.
    time_grid_sec = 10.0
    offset_slot_count = int(max_offset // time_grid_sec)
    window_start = earliest + rng.randint(0, offset_slot_count) * time_grid_sec
    window_end = window_start + window_duration
    early_time = window_start
    late_time = window_end - imaging_duration
    min_angle_time = (early_time + late_time) / 2.0

    early_roll, early_tilt = _random_attitude(rng, max_off_nadir=27.0)
    late_roll, late_tilt = _random_attitude(rng, max_off_nadir=27.0)
    # 최소각 후보가 이름 그대로 초반·후반 후보보다 좋은 각도를 갖게 보장한다.
    if hypot(early_roll, early_tilt) <= hypot(late_roll, late_tilt):
        min_roll, min_tilt = early_roll * 0.2, early_tilt * 0.2
    else:
        min_roll, min_tilt = late_roll * 0.2, late_tilt * 0.2
    candidates = [
        (OpportunityKind.EARLY, early_time, early_roll, early_tilt),
        (OpportunityKind.MIN_OFF_NADIR, min_angle_time, min_roll, min_tilt),
        (OpportunityKind.LATE, late_time, late_roll, late_tilt),
    ]

    result: list[Opportunity] = []
    seen_times: set[float] = set()
    for kind, capture_time, roll, tilt in candidates:
        rounded_time = round(capture_time, 6)
        if rounded_time in seen_times:
            continue
        seen_times.add(rounded_time)
        result.append(
            Opportunity(
                opportunity_id=(
                    f"{strip_id}-{orbit_pass.pass_id}-access-{access_index}-{kind.value}"
                ),
                order_id=order_id,
                strip_id=strip_id,
                pass_id=orbit_pass.pass_id,
                kind=kind,
                window_start_sec=window_start,
                window_end_sec=window_end,
                capture_time_sec=capture_time,
                required_roll_deg=roll,
                required_tilt_deg=tilt,
            )
        )
    return result


def _random_attitude(rng: random.Random, max_off_nadir: float) -> tuple[float, float]:
    while True:
        roll = rng.uniform(-max_off_nadir, max_off_nadir)
        tilt = rng.uniform(-max_off_nadir, max_off_nadir)
        if hypot(roll, tilt) <= max_off_nadir:
            return roll, tilt
