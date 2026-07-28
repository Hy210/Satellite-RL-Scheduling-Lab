from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_core.generator import generate_scenario
from rl_core.models import MaskablePPOTrainingConfig, RunStatus
from rl_core.replay import load_episode_replay
from rl_core.storage import StorageRepository
from rl_core.training import (
    TrainingStoppedError,
    evaluate_trained_policy,
    load_maskable_ppo_model,
    train_maskable_ppo,
    train_maskable_ppo_with_scenario_pool,
)


def test_maskable_ppo_training_saves_reloadable_artifacts(tmp_path: Path) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=11,
        evaluation_seed=17,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        checkpoint_interval=4,
        evaluation_interval=4,
        artifact_root=tmp_path,
    )

    artifacts = train_maskable_ppo(scenario, config, run_id="test-run")
    reloaded = load_maskable_ppo_model(artifacts.final_model_path, scenario)
    reloaded_evaluation = evaluate_trained_policy(
        reloaded,
        scenario,
        seed=config.evaluation_seed,
    )

    assert artifacts.run.status == RunStatus.COMPLETED
    assert artifacts.final_model_path.exists()
    assert artifacts.checkpoints
    assert all(path.exists() for path in artifacts.checkpoints)
    assert artifacts.metrics_path.exists()
    assert artifacts.replay_path.exists()
    assert load_episode_replay(artifacts.replay_path).total_return == pytest.approx(
        artifacts.final_evaluation.total_return
    )
    final_evaluation = json.loads(
        (artifacts.run_directory / "metrics" / "final-evaluation.json").read_text(encoding="utf-8")
    )
    assert final_evaluation["replay"]["steps"]
    assert final_evaluation["replay"]["total_return"] == pytest.approx(
        artifacts.final_evaluation.total_return
    )
    assert reloaded_evaluation.total_return == pytest.approx(
        artifacts.final_evaluation.total_return
    )
    assert reloaded_evaluation.priority_score + reloaded_evaluation.angle_bonus + (
        reloaded_evaluation.missed_penalty
    ) == pytest.approx(reloaded_evaluation.total_return)
    assert reloaded_evaluation.replay.total_return == pytest.approx(
        sum(step.reward_breakdown.total for step in reloaded_evaluation.replay.steps)
    )


def test_maskable_ppo_evaluation_is_reproducible(tmp_path: Path) -> None:
    # "동일 정책 평가 결과 재현": deterministic_eval=True에서 같은 모델을 두 번
    # 평가하면 완전히 같은 action·return이 나와야 한다 (학습 품질과는 별개 성질).
    scenario = generate_scenario(seed=20260707, size="tiny")
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=11,
        evaluation_seed=17,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        checkpoint_interval=8,
        evaluation_interval=8,
        artifact_root=tmp_path,
    )
    artifacts = train_maskable_ppo(scenario, config, run_id="reproducible-run")
    model = load_maskable_ppo_model(artifacts.final_model_path, scenario)

    first = evaluate_trained_policy(model, scenario, seed=config.evaluation_seed)
    second = evaluate_trained_policy(model, scenario, seed=config.evaluation_seed)

    assert first == second


def test_training_metrics_include_domain_evaluation(tmp_path: Path) -> None:
    scenario = generate_scenario(seed=31, size="tiny")
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=3,
        evaluation_seed=5,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        checkpoint_interval=8,
        evaluation_interval=4,
        artifact_root=tmp_path,
    )

    artifacts = train_maskable_ppo(scenario, config, run_id="metric-run")
    metric_rows = [
        json.loads(line) for line in artifacts.metrics_path.read_text(encoding="utf-8").splitlines()
    ]

    assert metric_rows
    assert {"timesteps", "evaluation"} <= metric_rows[0].keys()
    assert {
        "total_return",
        "priority_score",
        "angle_bonus",
        "missed_penalty",
        "completed_strips",
        "completed_orders",
    } <= metric_rows[0]["evaluation"].keys()


def test_training_config_rejects_batch_larger_than_rollout() -> None:
    with pytest.raises(ValidationError):
        MaskablePPOTrainingConfig(
            total_timesteps=8,
            learning_seed=1,
            evaluation_seed=2,
            n_steps=4,
            batch_size=8,
        )


def test_training_failure_is_recorded_in_optional_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository = StorageRepository(tmp_path / "data")
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=11,
        evaluation_seed=17,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        checkpoint_interval=8,
        evaluation_interval=8,
        artifact_root=tmp_path / "runs",
    )

    def fail_learn(self, *args, **kwargs):
        del self, args, kwargs
        raise RuntimeError("simulated learning failure")

    monkeypatch.setattr("rl_core.training.MaskablePPO.learn", fail_learn)

    with pytest.raises(RuntimeError, match="simulated learning failure"):
        train_maskable_ppo(scenario, config, run_id="failed-run", storage=repository)

    failed_run = repository.load_training_run("failed-run")
    assert failed_run.status == RunStatus.FAILED
    assert failed_run.error_message == "simulated learning failure"


def test_scenario_pool_training_saves_reloadable_artifacts(tmp_path: Path) -> None:
    training_scenarios = [generate_scenario(seed=seed, size="tiny") for seed in (1, 2, 3)]
    held_out_scenario = generate_scenario(seed=4, size="tiny")
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=11,
        evaluation_seed=17,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        checkpoint_interval=4,
        evaluation_interval=4,
        artifact_root=tmp_path,
    )

    artifacts = train_maskable_ppo_with_scenario_pool(
        training_scenarios,
        held_out_scenario,
        config,
        run_id="pool-test-run",
    )

    assert artifacts.training_scenario_ids == tuple(
        scenario.scenario_id for scenario in training_scenarios
    )
    assert artifacts.held_out_scenario_id == held_out_scenario.scenario_id
    assert artifacts.final_model_path.exists()
    assert artifacts.checkpoints
    assert artifacts.metrics_path.exists()
    assert artifacts.replay_path.exists()

    # held-out 평가는 pool에 없는 시나리오를 대상으로 해야 한다 — 학습이 pool을
    # 외운 것과 무관하게 실제로 일반화 신호를 추적하려는 목적과 어긋나면 안 된다.
    assert artifacts.final_evaluation.scenario_id == held_out_scenario.scenario_id

    # VecNormalize로 학습됐어도, 평가는 관측만 보고 reward 정규화 상태를 몰라도 되므로
    # raw 환경으로 다시 불러와 평가해도 동일한 결과가 나와야 한다.
    reloaded = load_maskable_ppo_model(artifacts.final_model_path, held_out_scenario)
    reloaded_evaluation = evaluate_trained_policy(
        reloaded, held_out_scenario, seed=config.evaluation_seed
    )
    assert reloaded_evaluation.total_return == pytest.approx(
        artifacts.final_evaluation.total_return
    )

    metric_rows = [
        json.loads(line) for line in artifacts.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert metric_rows


def test_training_stop_request_saves_checkpoint_without_final_evaluation(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=17,
        evaluation_seed=23,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        checkpoint_interval=8,
        evaluation_interval=8,
        artifact_root=repository.data_root / "runs",
    )

    with pytest.raises(TrainingStoppedError, match="training stopped"):
        train_maskable_ppo(
            scenario,
            config,
            run_id="stopped-run",
            storage=repository,
            should_stop=lambda: True,
        )

    run_directory = config.artifact_root / "stopped-run"
    stopped_run = repository.load_training_run("stopped-run")
    assert stopped_run.status == RunStatus.STOPPED
    assert list((run_directory / "checkpoints").glob("checkpoint-*.zip"))
    assert not (run_directory / "metrics" / "final-evaluation.json").exists()
    assert not (run_directory / "metrics" / "replay.json").exists()
    assert not (run_directory / "model" / "final-model.zip").exists()
