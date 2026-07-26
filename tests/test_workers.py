from __future__ import annotations

from pathlib import Path

from backend.workers import run_cp_sat_worker, run_training_worker
from rl_core.generator import generate_scenario
from rl_core.models import EvaluationRun, MaskablePPOTrainingConfig, RunStatus, TrainingRun
from rl_core.storage import StorageRepository


def test_worker_records_failure_before_trainer_starts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    run = TrainingRun(
        run_id="worker-failure",
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=17,
        total_timesteps=8,
        artifact_directory="runs/worker-failure",
    )
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=17,
        evaluation_seed=23,
        n_steps=4,
        batch_size=4,
        artifact_root=repository.data_root / "runs",
    )
    repository.save_training_config(run, config)

    def fail_training(*args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("simulated worker failure")

    monkeypatch.setattr("backend.workers.train_maskable_ppo", fail_training)
    run_training_worker(str(repository.data_root), run.run_id)

    failed_run = repository.load_training_run(run.run_id)
    assert failed_run.status == RunStatus.FAILED
    assert failed_run.error_message == "simulated worker failure"


def test_worker_does_not_start_a_run_stopped_before_process_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    run = TrainingRun(
        run_id="stopped-before-worker",
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=17,
        total_timesteps=8,
    )
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=17,
        evaluation_seed=23,
        n_steps=4,
        batch_size=4,
        artifact_root=repository.data_root / "runs",
    )
    repository.save_training_config(run, config)
    repository.save_training_run(run.model_copy(update={"status": RunStatus.STOPPED}))

    def unexpected_training(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("stopped run must not start trainer")

    monkeypatch.setattr("backend.workers.train_maskable_ppo", unexpected_training)
    run_training_worker(str(repository.data_root), run.run_id)

    assert repository.load_training_run(run.run_id).status == RunStatus.STOPPED


def test_cp_sat_worker_saves_completed_evaluation_artifacts(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    run = EvaluationRun(
        run_id="cp-sat-worker",
        scenario_id=scenario.scenario_id,
        policy_name="cp_sat_baseline",
        seed=17,
        status=RunStatus.QUEUED,
    )
    repository.save_evaluation_run(run)

    run_cp_sat_worker(str(repository.data_root), run.run_id, 10.0)

    completed = repository.load_evaluation_run(run.run_id)
    assert completed.status is RunStatus.COMPLETED
    assert completed.result_path == "evaluations/cp-sat-worker/summary.json"
    assert (
        repository.load_json_artifact(Path(completed.result_path))["policy_name"]
        == "cp_sat_baseline"
    )
