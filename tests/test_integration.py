from __future__ import annotations

from rl_core.generator import generate_scenario
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
