from __future__ import annotations

import pytest
from pydantic import ValidationError

from rl_core.generator import generate_scenario
from rl_core.models import AngleRange, Rectangle, SatelliteConfig, Scenario


def test_scenario_json_round_trip(tmp_path) -> None:
    original = generate_scenario(seed=42, size="tiny")
    scenario_path = tmp_path / "scenario.json"
    original.save(scenario_path)

    restored = Scenario.load(scenario_path)

    assert restored == original


def test_rectangle_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError, match="min_lat must be smaller"):
        Rectangle(min_lat=1.0, min_lon=0.0, max_lat=0.0, max_lon=1.0)


def test_angle_range_rejects_reversed_values() -> None:
    with pytest.raises(ValidationError, match="minimum must not exceed"):
        AngleRange(minimum=10.0, maximum=-10.0)


def test_scenario_rejects_duplicate_ids() -> None:
    scenario = generate_scenario(seed=7, size="tiny")
    duplicated = scenario.model_dump()
    duplicated["strips"].append(duplicated["strips"][0])

    with pytest.raises(ValidationError, match="duplicate strip ID"):
        Scenario.model_validate(duplicated)


def test_scenario_rejects_unknown_opportunity_reference() -> None:
    scenario = generate_scenario(seed=7, size="tiny")
    invalid = scenario.model_dump()
    invalid["opportunities"][0]["strip_id"] = "missing-strip"

    with pytest.raises(ValidationError, match="unknown reference"):
        Scenario.model_validate(invalid)


def test_satellite_rejects_invalid_initial_attitude() -> None:
    with pytest.raises(ValidationError, match="initial roll exceeds"):
        SatelliteConfig(satellite_id="sat-1", initial_roll_deg=31.0)


def test_scenario_rejects_order_without_strips() -> None:
    scenario = generate_scenario(seed=7, size="tiny")
    invalid = scenario.model_dump()
    first_order_id = invalid["orders"][0]["order_id"]
    invalid["strips"] = [item for item in invalid["strips"] if item["order_id"] != first_order_id]
    invalid["opportunities"] = [
        item for item in invalid["opportunities"] if item["order_id"] != first_order_id
    ]

    with pytest.raises(ValidationError, match="at least one strip"):
        Scenario.model_validate(invalid)


def test_scenario_rejects_capture_that_does_not_fit_window() -> None:
    scenario = generate_scenario(seed=7, size="tiny")
    invalid = scenario.model_dump()
    opportunity = invalid["opportunities"][0]
    opportunity["window_end_sec"] = opportunity["capture_time_sec"] + 1.0

    with pytest.raises(ValidationError, match="full imaging duration"):
        Scenario.model_validate(invalid)


def test_scenario_rejects_unknown_access_window_reference() -> None:
    scenario = generate_scenario(seed=7, size="tiny")
    invalid = scenario.model_dump()
    invalid["opportunities"][0]["source_access_window_id"] = "missing-access-window"

    with pytest.raises(ValidationError, match="unknown access window"):
        Scenario.model_validate(invalid)
