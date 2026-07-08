from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import pytest

from rl_core.generator import generate_scenario
from rl_core.models import (
    AngleRange,
    EnvironmentConfig,
    Opportunity,
    OpportunityKind,
    OrbitPass,
    Order,
    Polygon,
    Priority,
    Rectangle,
    RewardConfig,
    SatelliteConfig,
    Scenario,
    Strip,
)
from rl_core.policies import (
    BaselinePolicy,
    EarliestDeadlineFirstPolicy,
    PriorityEfficiencyGreedyPolicy,
    PriorityGreedyPolicy,
    RandomValidPolicy,
    evaluate_policy,
)
from rl_core.replay import (
    load_episode_replay,
    load_policy_comparison,
    policy_comparison,
    policy_comparison_entry,
    save_episode_replay,
    save_policy_comparison,
)
from rl_core.simulator import SatelliteSchedulingSimulator

POLICIES: tuple[BaselinePolicy, ...] = (
    RandomValidPolicy(),
    EarliestDeadlineFirstPolicy(),
    PriorityGreedyPolicy(),
    PriorityEfficiencyGreedyPolicy(),
)


@pytest.mark.parametrize("size", ["tiny", "small", "full"])
@pytest.mark.parametrize("policy", POLICIES, ids=lambda policy: policy.name)
def test_all_policies_finish_all_scenario_sizes(
    size: str,
    policy: BaselinePolicy,
) -> None:
    scenario = generate_scenario(seed=20260707, size=size)

    result = evaluate_policy(policy, scenario, seed=17)

    assert result.steps > 0
    assert result.captures > 0
    assert result.total_return == pytest.approx(
        result.priority_score + result.angle_bonus + result.missed_penalty
    )
    assert all(
        decision.action == 0 or decision.action in decision.valid_candidate_slots
        for decision in result.decisions
    )


@pytest.mark.parametrize("policy", POLICIES, ids=lambda policy: policy.name)
def test_policy_evaluation_is_reproducible(policy: BaselinePolicy) -> None:
    scenario = generate_scenario(seed=91, size="tiny")

    first = evaluate_policy(policy, scenario, seed=123)
    second = evaluate_policy(policy, scenario, seed=123)

    assert first == second


def test_earliest_deadline_policy_prefers_closest_deadline() -> None:
    scenario = build_competing_scenario(
        [
            ("red", Priority.RED, 100.0, 0.0),
            ("background", Priority.BACKGROUND, 50.0, 0.0),
        ]
    )

    selected_order = first_selected_order(EarliestDeadlineFirstPolicy(), scenario)

    assert selected_order == "background"


def test_priority_policy_prefers_highest_priority() -> None:
    scenario = build_competing_scenario(
        [
            ("red", Priority.RED, 100.0, 0.0),
            ("background", Priority.BACKGROUND, 50.0, 0.0),
        ]
    )

    selected_order = first_selected_order(PriorityGreedyPolicy(), scenario)

    assert selected_order == "red"


def test_efficiency_policy_balances_priority_and_slew_cost() -> None:
    scenario = build_competing_scenario(
        [
            ("red", Priority.RED, 100.0, 30.0),
            ("blue", Priority.BLUE, 100.0, 0.0),
        ]
    )

    selected_order = first_selected_order(PriorityEfficiencyGreedyPolicy(), scenario)

    assert selected_order == "blue"


def test_random_policy_seed_changes_decision_sequence() -> None:
    scenario = build_competing_scenario(
        [
            ("red", Priority.RED, 100.0, 0.0),
            ("blue", Priority.BLUE, 100.0, 0.0),
            ("background", Priority.BACKGROUND, 100.0, 0.0),
        ]
    )

    selected = {
        first_selected_order(RandomValidPolicy(), scenario, seed=seed) for seed in range(10)
    }

    assert len(selected) > 1


def test_full_scenario_creates_meaningful_policy_comparison() -> None:
    scenario = generate_scenario(seed=20260707, size="full")

    results = [evaluate_policy(policy, scenario, seed=17) for policy in POLICIES]

    assert len({result.total_return for result in results}) > 1
    assert any(
        len(decision.valid_candidate_slots) > 1
        for result in results
        for decision in result.decisions
    )


def first_selected_order(
    policy: BaselinePolicy,
    scenario: Scenario,
    *,
    seed: int = 1,
) -> str:
    result = evaluate_policy(policy, scenario, seed=seed)
    opportunity_id = next(
        decision.opportunity_id for decision in result.decisions if decision.opportunity_id
    )
    opportunity = next(
        item for item in scenario.opportunities if item.opportunity_id == opportunity_id
    )
    return opportunity.order_id


def build_competing_scenario(
    specs: Sequence[tuple[str, Priority, float, float]],
) -> Scenario:
    rectangle = Rectangle(min_lat=0.0, min_lon=0.0, max_lat=1.0, max_lon=1.0)
    polygon = Polygon.model_validate(
        {
            "vertices": [
                {"lat": 0.0, "lon": 0.0},
                {"lat": 0.0, "lon": 1.0},
                {"lat": 1.0, "lon": 1.0},
                {"lat": 1.0, "lon": 0.0},
            ]
        }
    )
    orders: list[Order] = []
    strips: list[Strip] = []
    opportunities: list[Opportunity] = []
    for index, (order_id, priority, deadline, roll) in enumerate(specs):
        strip_id = f"{order_id}-strip"
        orders.append(
            Order(
                order_id=order_id,
                name=order_id,
                priority=priority,
                request_start_sec=0.0,
                request_end_sec=deadline,
                geometry=rectangle,
                allowed_roll_deg=AngleRange(minimum=-30.0, maximum=30.0),
                allowed_tilt_deg=AngleRange(minimum=-30.0, maximum=30.0),
            )
        )
        strips.append(
            Strip(
                strip_id=strip_id,
                order_id=order_id,
                sequence=index,
                geometry=polygon,
            )
        )
        opportunities.append(
            Opportunity(
                opportunity_id=f"{order_id}-opportunity",
                order_id=order_id,
                strip_id=strip_id,
                pass_id="pass-1",
                kind=OpportunityKind.MIN_OFF_NADIR,
                window_start_sec=9.0,
                window_end_sec=15.0,
                capture_time_sec=10.0,
                required_roll_deg=roll,
                required_tilt_deg=0.0,
            )
        )
    return Scenario(
        scenario_id="competing-scenario",
        name="Competing candidates",
        seed=1,
        satellite=SatelliteConfig(satellite_id="sat-1"),
        environment=EnvironmentConfig(duration_sec=120.0),
        reward=RewardConfig(),
        passes=[
            OrbitPass(
                pass_id="pass-1",
                sequence=0,
                start_time_sec=0.0,
                end_time_sec=120.0,
            )
        ],
        orders=orders,
        strips=strips,
        opportunities=opportunities,
    )


def test_policy_can_only_select_current_unmasked_slot() -> None:
    scenario = build_competing_scenario([("red", Priority.RED, 100.0, 0.0)])
    simulator = SatelliteSchedulingSimulator(scenario)
    simulator.step(0)
    observation = simulator.observe()

    for policy in POLICIES:
        action = policy.select_action(simulator, observation, random.Random(1))
        assert observation.action_mask[action]


def test_policy_replay_records_mask_reasons_and_reward_sum(tmp_path: Path) -> None:
    scenario = build_competing_scenario(
        [
            ("valid", Priority.RED, 100.0, 0.0),
            ("masked", Priority.BLUE, 100.0, 45.0),
        ]
    )

    result = evaluate_policy(RandomValidPolicy(), scenario, seed=1)
    masked_candidate = next(
        candidate
        for step in result.replay.steps
        for candidate in step.candidates
        if candidate.order_id == "masked"
    )

    assert masked_candidate.valid is False
    assert "roll_limit" in masked_candidate.mask_reasons
    assert "order_roll_limit" in masked_candidate.mask_reasons
    assert result.replay.total_return == pytest.approx(
        sum(step.reward_breakdown.total for step in result.replay.steps)
    )
    assert result.replay.schedule
    replay_path = tmp_path / "random-valid-replay.json"
    save_episode_replay(replay_path, result.replay)
    assert load_episode_replay(replay_path) == result.replay


def test_policy_comparison_artifact_summarizes_multiple_replays(tmp_path: Path) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    results = [evaluate_policy(policy, scenario, seed=17) for policy in POLICIES]
    entries = []
    for result in results:
        replay_path = tmp_path / f"{result.policy_name}-replay.json"
        save_episode_replay(replay_path, result.replay)
        entries.append(
            policy_comparison_entry(
                replay=result.replay,
                priority_score=result.priority_score,
                angle_bonus=result.angle_bonus,
                missed_penalty=result.missed_penalty,
                replay_path=replay_path,
            )
        )

    comparison = policy_comparison(entries)
    comparison_path = tmp_path / "policy-comparison.json"
    save_policy_comparison(comparison_path, comparison)
    loaded = load_policy_comparison(comparison_path)

    assert loaded == comparison
    assert {entry.policy_name for entry in loaded.entries} == {policy.name for policy in POLICIES}
    assert loaded.best_policy_name == max(
        loaded.entries,
        key=lambda entry: (
            entry.total_return,
            entry.completed_orders,
            entry.completed_strips,
            entry.captures,
            entry.policy_name,
        ),
    ).policy_name
    assert all(entry.replay_path is not None for entry in loaded.entries)
