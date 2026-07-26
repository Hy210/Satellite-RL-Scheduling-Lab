from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_core.generator import generate_scenario
from rl_core.models import MaskablePPOTrainingConfig
from rl_core.optimization import (
    CP_SAT_POLICY_NAME,
    load_optimization_baseline,
    save_optimization_baseline,
    solve_cp_sat_baseline,
)
from rl_core.policies import (
    EarliestDeadlineFirstPolicy,
    PriorityEfficiencyGreedyPolicy,
    PriorityGreedyPolicy,
    RandomValidPolicy,
    evaluate_policy,
)
from rl_core.replay import (
    load_policy_comparison,
    policy_comparison,
    policy_comparison_entry,
    save_policy_comparison,
)
from rl_core.training import train_maskable_ppo

POLICIES = (
    RandomValidPolicy(),
    EarliestDeadlineFirstPolicy(),
    PriorityGreedyPolicy(),
    PriorityEfficiencyGreedyPolicy(),
)


def test_cp_sat_baseline_solves_tiny_scenario_and_replays(tmp_path) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")

    result = solve_cp_sat_baseline(scenario, time_limit_sec=10.0, seed=17)

    assert result.status == "OPTIMAL"
    assert result.objective_value is not None
    assert result.best_objective_bound is not None
    assert result.optimality_gap == pytest.approx(0.0)
    assert result.selected_opportunity_ids
    assert result.replay.policy_name == CP_SAT_POLICY_NAME
    assert result.replay.schedule
    assert result.replay.total_return == pytest.approx(
        sum(step.reward_breakdown.total for step in result.replay.steps)
    )
    assert {capture.opportunity_id for capture in result.replay.schedule} == set(
        result.selected_opportunity_ids
    )

    path = tmp_path / "cp-sat-baseline.json"
    save_optimization_baseline(path, result)
    assert load_optimization_baseline(path) == result


def test_cp_sat_baseline_joins_all_policy_types_in_one_comparison(tmp_path) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    baseline_results = [evaluate_policy(policy, scenario, seed=17) for policy in POLICIES]
    training_artifacts = train_maskable_ppo(
        scenario,
        MaskablePPOTrainingConfig(
            total_timesteps=8,
            learning_seed=11,
            evaluation_seed=17,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            checkpoint_interval=8,
            evaluation_interval=8,
            artifact_root=tmp_path,
        ),
        run_id="comparison-ppo",
    )
    cp_sat_result = solve_cp_sat_baseline(scenario, time_limit_sec=10.0, seed=17)

    comparison = policy_comparison(
        [
            *[
                policy_comparison_entry(
                    replay=result.replay,
                    priority_score=result.priority_score,
                    angle_bonus=result.angle_bonus,
                    missed_penalty=result.missed_penalty,
                )
                for result in baseline_results
            ],
            policy_comparison_entry(
                replay=training_artifacts.final_evaluation.replay,
                priority_score=training_artifacts.final_evaluation.priority_score,
                angle_bonus=training_artifacts.final_evaluation.angle_bonus,
                missed_penalty=training_artifacts.final_evaluation.missed_penalty,
            ),
            policy_comparison_entry(
                replay=cp_sat_result.replay,
                priority_score=sum(
                    step.reward_breakdown.strip_base for step in cp_sat_result.replay.steps
                ),
                angle_bonus=sum(
                    step.reward_breakdown.angle_bonus for step in cp_sat_result.replay.steps
                ),
                missed_penalty=sum(
                    step.reward_breakdown.missed_penalty for step in cp_sat_result.replay.steps
                ),
            ),
        ]
    )

    comparison_path = tmp_path / "policy-comparison.json"
    save_policy_comparison(comparison_path, comparison)
    loaded = load_policy_comparison(comparison_path)

    assert loaded == comparison
    assert {entry.policy_name for entry in loaded.entries} == {
        *(policy.name for policy in POLICIES),
        "maskable_ppo",
        CP_SAT_POLICY_NAME,
    }


@pytest.mark.parametrize(
    ("status_code", "status_name"),
    [
        ("INFEASIBLE", "INFEASIBLE"),
        ("UNKNOWN", "UNKNOWN"),
    ],
)
def test_cp_sat_baseline_persists_no_solution_status_artifact(
    tmp_path,
    monkeypatch,
    status_code: str,
    status_name: str,
) -> None:
    """해가 없거나 시간 제한 전에 탐색이 끝난 경우에도 결과를 분석 가능하게 남긴다."""

    import rl_core.optimization as optimization

    resolved_status = getattr(optimization.cp_model, status_code)

    class NoSolutionSolver:
        def __init__(self) -> None:
            self.parameters = SimpleNamespace(max_time_in_seconds=0.0, random_seed=0)

        def Solve(self, model) -> int:
            del model
            return resolved_status

        def StatusName(self, status: int) -> str:
            assert status == resolved_status
            return status_name

    monkeypatch.setattr(optimization.cp_model, "CpSolver", NoSolutionSolver)
    scenario = generate_scenario(seed=20260707, size="tiny")

    result = solve_cp_sat_baseline(scenario, time_limit_sec=0.001, seed=17)

    assert result.status == status_name
    assert result.objective_value is None
    assert result.best_objective_bound is None
    assert result.optimality_gap is None
    assert result.selected_opportunity_ids == []
    assert result.replay.policy_name == CP_SAT_POLICY_NAME
    assert result.replay.schedule == []

    path = tmp_path / f"{status_name.lower()}-cp-sat.json"
    save_optimization_baseline(path, result)
    assert load_optimization_baseline(path) == result
