from __future__ import annotations

import pytest

from rl_core.simulator import SatelliteSchedulingSimulator
from tests.conftest import build_scenario, make_opportunity


def advance_to_first_event(simulator: SatelliteSchedulingSimulator) -> None:
    result = simulator.step(0)
    assert not result.terminated


def test_same_attitude_has_zero_slew_time() -> None:
    simulator = SatelliteSchedulingSimulator(build_scenario())

    assert simulator.slew_time_to(0.0, 0.0) == 0.0


def test_roll_and_tilt_slew_times_are_sequential() -> None:
    simulator = SatelliteSchedulingSimulator(build_scenario())

    assert simulator.slew_time_to(10.0, 5.0) == pytest.approx(4.0)


def test_capture_reward_and_duplicate_strip_mask() -> None:
    scenario = build_scenario(
        opportunities=[
            make_opportunity("op-1", "strip-1", 10.0),
            make_opportunity("op-2", "strip-1", 30.0),
            make_opportunity("op-3", "strip-2", 30.0),
        ]
    )
    simulator = SatelliteSchedulingSimulator(scenario)
    advance_to_first_event(simulator)

    capture = simulator.step(1)

    assert capture.reward == pytest.approx(2.75)
    assert capture.breakdown.strip_base == pytest.approx(2.5)
    assert capture.breakdown.angle_bonus == pytest.approx(0.25)
    assert capture.observation.current_time_sec == pytest.approx(30.0)
    assert capture.observation.candidates[0].valid is False
    assert "strip_completed" in capture.observation.candidates[0].mask_reasons


def test_minimum_interval_masks_too_early_capture() -> None:
    scenario = build_scenario(
        opportunities=[
            make_opportunity("op-1", "strip-1", 10.0),
            make_opportunity("op-2", "strip-2", 18.0),
        ]
    )
    simulator = SatelliteSchedulingSimulator(scenario)
    advance_to_first_event(simulator)

    capture = simulator.step(1)

    assert capture.observation.current_time_sec == pytest.approx(18.0)
    assert capture.observation.candidates[0].valid is False
    assert "insufficient_transition_time" in capture.observation.candidates[0].mask_reasons


def test_attitude_limit_is_masked() -> None:
    scenario = build_scenario(opportunities=[make_opportunity("op-1", "strip-1", 10.0, roll=31.0)])
    simulator = SatelliteSchedulingSimulator(scenario)
    advance_to_first_event(simulator)

    candidate = simulator.observe().candidates[0]

    assert candidate.valid is False
    assert "roll_limit" in candidate.mask_reasons
    assert "combined_off_nadir_limit" in candidate.mask_reasons


def test_masked_action_cannot_be_selected() -> None:
    scenario = build_scenario(opportunities=[make_opportunity("op-1", "strip-1", 1.0, roll=30.0)])
    simulator = SatelliteSchedulingSimulator(scenario)
    advance_to_first_event(simulator)

    with pytest.raises(ValueError, match="masked action"):
        simulator.step(1)


def test_partial_completion_reduces_missed_penalty() -> None:
    scenario = build_scenario(
        opportunities=[make_opportunity("op-1", "strip-1", 10.0)],
        request_end=100.0,
    )
    simulator = SatelliteSchedulingSimulator(scenario)
    advance_to_first_event(simulator)
    capture = simulator.step(1)

    assert capture.observation.current_time_sec == pytest.approx(100.0)
    assert capture.breakdown.missed_penalty == pytest.approx(-1.25)
    assert capture.reward == pytest.approx(1.5)
    assert capture.expired_order_ids == ("order-1",)


def test_skip_has_zero_reward_before_deadline() -> None:
    scenario = build_scenario(
        opportunities=[
            make_opportunity("op-1", "strip-1", 10.0),
            make_opportunity("op-2", "strip-2", 20.0),
        ]
    )
    simulator = SatelliteSchedulingSimulator(scenario)
    advance_to_first_event(simulator)

    skipped = simulator.step(0)

    assert skipped.reward == 0.0
    assert skipped.observation.current_time_sec == pytest.approx(20.0)


def test_simultaneous_candidates_allow_only_one_capture() -> None:
    scenario = build_scenario(
        opportunities=[
            make_opportunity("op-1", "strip-1", 10.0),
            make_opportunity("op-2", "strip-2", 10.0),
        ]
    )
    simulator = SatelliteSchedulingSimulator(scenario)
    advance_to_first_event(simulator)

    result = simulator.step(1)

    assert result.observation.completed_strip_count == 1
    assert simulator.processed_opportunity_ids == {"op-1", "op-2"}


def test_all_strips_completed_terminates_episode() -> None:
    scenario = build_scenario(
        opportunities=[make_opportunity("op-1", "strip-1", 10.0)],
        strip_count=1,
    )
    simulator = SatelliteSchedulingSimulator(scenario)
    advance_to_first_event(simulator)

    result = simulator.step(1)

    assert result.terminated is True
    assert result.observation.completed_order_count == 1


def test_deterministic_action_sequence_produces_same_result() -> None:
    scenario = build_scenario(
        opportunities=[
            make_opportunity("op-1", "strip-1", 10.0),
            make_opportunity("op-2", "strip-2", 30.0),
        ]
    )
    first = SatelliteSchedulingSimulator(scenario)
    second = SatelliteSchedulingSimulator(scenario)

    first_steps = [first.step(0), first.step(1), first.step(1)]
    second_steps = [second.step(0), second.step(1), second.step(1)]

    assert first_steps == second_steps
    assert first.cumulative_return == second.cumulative_return


def test_no_opportunity_scenario_reaches_horizon_and_penalizes_order() -> None:
    simulator = SatelliteSchedulingSimulator(build_scenario())

    first = simulator.step(0)
    final = simulator.step(0)

    assert first.observation.current_time_sec == pytest.approx(100.0)
    assert first.breakdown.missed_penalty == pytest.approx(-2.5)
    assert final.terminated is True
    assert final.observation.current_time_sec == pytest.approx(200.0)
    assert simulator.cumulative_return == pytest.approx(-2.5)
