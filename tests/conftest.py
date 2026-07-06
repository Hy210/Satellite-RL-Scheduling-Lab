from __future__ import annotations

from collections.abc import Iterable

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


def build_scenario(
    *,
    opportunities: Iterable[Opportunity] | None = None,
    strip_count: int = 2,
    priority: Priority = Priority.RED,
    request_end: float = 100.0,
) -> Scenario:
    order = Order(
        order_id="order-1",
        name="Test order",
        priority=priority,
        request_start_sec=0.0,
        request_end_sec=request_end,
        geometry=Rectangle(min_lat=0.0, min_lon=0.0, max_lat=1.0, max_lon=1.0),
        allowed_roll_deg=AngleRange(minimum=-30.0, maximum=30.0),
        allowed_tilt_deg=AngleRange(minimum=-30.0, maximum=30.0),
    )
    strips = [
        Strip(
            strip_id=f"strip-{index + 1}",
            order_id=order.order_id,
            sequence=index,
            geometry=Rectangle(
                min_lat=0.0,
                min_lon=index * 0.1,
                max_lat=1.0,
                max_lon=(index + 1) * 0.1,
            ),
        )
        for index in range(strip_count)
    ]
    return Scenario(
        scenario_id="scenario-test",
        name="Test scenario",
        seed=1,
        satellite=SatelliteConfig(satellite_id="sat-1"),
        environment=EnvironmentConfig(duration_sec=200.0),
        reward=RewardConfig(),
        passes=[OrbitPass(pass_id="pass-1", sequence=0, start_time_sec=0.0, end_time_sec=120.0)],
        orders=[order],
        strips=strips,
        opportunities=list(opportunities or []),
    )


def make_opportunity(
    opportunity_id: str,
    strip_id: str,
    capture_time: float,
    *,
    roll: float = 0.0,
    tilt: float = 0.0,
) -> Opportunity:
    return Opportunity(
        opportunity_id=opportunity_id,
        order_id="order-1",
        strip_id=strip_id,
        pass_id="pass-1",
        kind=OpportunityKind.MIN_OFF_NADIR,
        window_start_sec=max(0.0, capture_time - 1.0),
        window_end_sec=capture_time + 5.0,
        capture_time_sec=capture_time,
        required_roll_deg=roll,
        required_tilt_deg=tilt,
    )
