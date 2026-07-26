from __future__ import annotations

from rl_core.generator import generate_scenario
from rl_core.policies import PriorityGreedyPolicy, evaluate_policy
from rl_core.simulator import SatelliteSchedulingSimulator


def test_tiny_generated_scenario_finishes_with_valid_actions() -> None:
    simulator = SatelliteSchedulingSimulator(generate_scenario(seed=20260706, size="tiny"))
    steps = 0

    while not simulator.terminated and steps < 10_000:
        observation = simulator.observe()
        action = next(
            (index for index, valid in enumerate(observation.action_mask) if index and valid),
            0,
        )
        simulator.step(action)
        steps += 1

    assert simulator.terminated
    assert steps < 10_000
    assert simulator.completed_strip_ids


def test_replay_schedule_respects_capture_window_and_minimum_interval() -> None:
    # 시뮬레이터의 action mask가 매 step 강제하는 시간/자세 제약(rl_core/simulator.py의
    # mask_reasons, slew_time_to)을, 저장된 replay 결과만으로 시뮬레이터를 다시 호출하지
    # 않고 독립적으로 재확인한다 — "저장된 결과가 실제로 유효한가"를 별도로 증명하기 위함.
    scenario = generate_scenario(seed=20260706, size="tiny")
    result = evaluate_policy(PriorityGreedyPolicy(), scenario, seed=17)
    schedule = result.replay.schedule
    assert schedule

    opportunities_by_id = {
        opportunity.opportunity_id: opportunity for opportunity in scenario.opportunities
    }
    satellite = scenario.satellite
    environment = scenario.environment

    previous_end_sec: float | None = None
    previous_roll_deg = satellite.initial_roll_deg
    previous_tilt_deg = satellite.initial_tilt_deg
    for capture in schedule:
        opportunity = opportunities_by_id[capture.opportunity_id]
        capture_end_sec = capture.capture_time_sec + environment.imaging_duration_sec
        assert capture_end_sec <= opportunity.window_end_sec

        delta_roll = abs(capture.roll_deg - previous_roll_deg)
        delta_tilt = abs(capture.tilt_deg - previous_tilt_deg)
        slew_time_sec = (
            0.0
            if delta_roll < 1e-12 and delta_tilt < 1e-12
            else delta_roll / satellite.roll_rate_deg_per_sec
            + delta_tilt / satellite.tilt_rate_deg_per_sec
            + satellite.settling_time_sec
        )
        earliest_start_sec = (
            slew_time_sec
            if previous_end_sec is None
            else previous_end_sec + max(environment.minimum_interval_sec, slew_time_sec)
        )
        assert capture.capture_time_sec + 1e-6 >= earliest_start_sec

        previous_end_sec = capture_end_sec
        previous_roll_deg = capture.roll_deg
        previous_tilt_deg = capture.tilt_deg
