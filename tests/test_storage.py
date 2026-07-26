from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_core.generator import generate_scenario
from rl_core.models import EvaluationRun, MaskablePPOTrainingConfig, RunStatus, TrainingRun
from rl_core.storage import ArtifactNotFoundError, StorageRepository


@pytest.fixture
def repository(tmp_path: Path) -> StorageRepository:
    return StorageRepository(tmp_path / "data")


def test_scenario_round_trip_and_metadata_are_saved(repository: StorageRepository) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")

    relative_path = repository.save_scenario(scenario)

    assert relative_path == Path("scenarios") / scenario.scenario_id / "scenario.json"
    assert repository.load_scenario(scenario.scenario_id) == scenario
    assert repository.list_missing_artifacts() == []


def test_run_metadata_config_and_result_artifacts_round_trip(repository: StorageRepository) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    training_run = TrainingRun(
        run_id="training-1",
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=11,
        total_timesteps=8,
        status=RunStatus.RUNNING,
        artifact_directory="runs/training-1",
    )
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=11,
        evaluation_seed=17,
        n_steps=8,
        batch_size=4,
    )

    config_path = repository.save_training_config(training_run, config)
    completed_run = training_run.model_copy(update={"status": RunStatus.COMPLETED})
    repository.save_training_run(completed_run)
    evaluation_run = EvaluationRun(
        run_id="evaluation-1",
        scenario_id=scenario.scenario_id,
        policy_name="maskable_ppo",
        seed=17,
        status=RunStatus.COMPLETED,
        source_training_run_id=training_run.run_id,
        result_path="runs/training-1/metrics/replay.json",
    )
    repository.save_evaluation_run(evaluation_run)
    result_path = repository.save_json_artifact(
        artifact_type="episode_replay",
        owner_type="evaluation_run",
        owner_id=evaluation_run.run_id,
        scenario_id=scenario.scenario_id,
        relative_path=Path("runs") / training_run.run_id / "metrics" / "replay.json",
        payload={"total_return": 5.0},
    )
    model_path = Path("runs") / training_run.run_id / "model" / "final-model.zip"
    absolute_model_path = repository.data_root / model_path
    absolute_model_path.parent.mkdir(parents=True)
    absolute_model_path.write_bytes(b"model-content")
    assert (
        repository.register_existing_artifact(
            artifact_type="model",
            owner_type="training_run",
            owner_id=training_run.run_id,
            scenario_id=scenario.scenario_id,
            relative_path=model_path,
        )
        == model_path
    )

    assert repository.load_training_run(training_run.run_id) == completed_run
    assert repository.load_evaluation_run(evaluation_run.run_id) == evaluation_run
    assert repository.load_json_artifact(config_path)["learning_seed"] == 11
    assert repository.load_json_artifact(result_path) == {"total_return": 5.0}
    assert repository.list_missing_artifacts() == []


def test_artifact_path_escape_and_owner_mismatch_are_rejected(
    repository: StorageRepository,
) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)

    with pytest.raises(ValueError, match="inside data_root"):
        repository.save_json_artifact(
            artifact_type="invalid",
            owner_type="scenario",
            owner_id=scenario.scenario_id,
            scenario_id=scenario.scenario_id,
            relative_path=Path("..") / "outside.json",
            payload={},
        )
    with pytest.raises(ValueError, match="must match scenario_id"):
        repository.save_json_artifact(
            artifact_type="invalid",
            owner_type="scenario",
            owner_id="other-scenario",
            scenario_id=scenario.scenario_id,
            relative_path=Path("scenarios") / scenario.scenario_id / "invalid.json",
            payload={},
        )


def test_missing_artifact_is_reported_and_index_can_be_reconciled(
    repository: StorageRepository,
) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    relative_path = repository.save_scenario(scenario)
    (repository.data_root / relative_path).unlink()

    with pytest.raises(ArtifactNotFoundError, match="artifact file is missing"):
        repository.load_scenario(scenario.scenario_id)
    assert [item.relative_path for item in repository.list_missing_artifacts()] == [relative_path]


def test_failed_artifact_replacement_keeps_previous_file(
    repository: StorageRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """임시 파일 교체가 실패해도 이미 완성된 artifact는 손상시키지 않는다."""

    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    relative_path = Path("scenarios") / scenario.scenario_id / "note.json"
    repository.save_json_artifact(
        artifact_type="note",
        owner_type="scenario",
        owner_id=scenario.scenario_id,
        scenario_id=scenario.scenario_id,
        relative_path=relative_path,
        payload={"revision": 1},
    )
    original_replace = Path.replace

    def fail_replace(source: Path, target: Path) -> Path:
        if target == repository.data_root / relative_path:
            raise OSError("simulated replacement failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replacement failure"):
        repository.save_json_artifact(
            artifact_type="note",
            owner_type="scenario",
            owner_id=scenario.scenario_id,
            scenario_id=scenario.scenario_id,
            relative_path=relative_path,
            payload={"revision": 2},
        )

    assert repository.load_json_artifact(relative_path) == {"revision": 1}


def test_recovery_transitions_only_interrupted_runs_to_terminal_states(
    repository: StorageRepository,
) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    running_training = TrainingRun(
        run_id="running-training",
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=1,
        total_timesteps=8,
        status=RunStatus.RUNNING,
    )
    stopping_training = running_training.model_copy(
        update={"run_id": "stopping-training", "status": RunStatus.STOP_REQUESTED}
    )
    completed_training = running_training.model_copy(
        update={"run_id": "completed-training", "status": RunStatus.COMPLETED}
    )
    running_evaluation = EvaluationRun(
        run_id="running-evaluation",
        scenario_id=scenario.scenario_id,
        policy_name="random_valid",
        seed=2,
        status=RunStatus.RUNNING,
    )
    for run in (running_training, stopping_training, completed_training):
        repository.save_training_run(run)
    repository.save_evaluation_run(running_evaluation)

    recovered = repository.recover_interrupted_runs()

    assert [(item.run_kind, item.run_id, item.recovered_status) for item in recovered] == [
        ("training", "running-training", "failed"),
        ("training", "stopping-training", "stopped"),
        ("evaluation", "running-evaluation", "failed"),
    ]
    assert repository.load_training_run("running-training").status == RunStatus.FAILED
    assert repository.load_training_run("stopping-training").status == RunStatus.STOPPED
    assert repository.load_training_run("completed-training").status == RunStatus.COMPLETED
    assert repository.load_evaluation_run("running-evaluation").status == RunStatus.FAILED
    assert "worker process was unavailable" in (
        repository.load_training_run("running-training").error_message or ""
    )


def test_terminal_run_cannot_return_to_running(repository: StorageRepository) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    completed_run = TrainingRun(
        run_id="completed-training",
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=1,
        total_timesteps=8,
        status=RunStatus.COMPLETED,
    )
    repository.save_training_run(completed_run)

    with pytest.raises(ValueError, match="completed -> running"):
        repository.save_training_run(completed_run.model_copy(update={"status": RunStatus.RUNNING}))


def test_scenario_validation_reports_checksum_and_structure_issues(
    repository: StorageRepository,
) -> None:
    scenario = generate_scenario(seed=20260707, size="tiny")
    relative_path = repository.save_scenario(scenario)
    artifact_path = repository.data_root / relative_path

    assert repository.validate_scenario(scenario.scenario_id).valid is True

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["name"] = "changed outside repository"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    checksum_result = repository.validate_scenario(scenario.scenario_id)
    assert checksum_result.valid is False
    assert [issue.code for issue in checksum_result.issues] == ["artifact_checksum_mismatch"]

    artifact_path.write_text(json.dumps({"scenario_id": scenario.scenario_id}), encoding="utf-8")
    structure_result = repository.validate_scenario(scenario.scenario_id)
    assert structure_result.valid is False
    assert "artifact_checksum_mismatch" in [issue.code for issue in structure_result.issues]
    assert any(issue.location == ("name",) for issue in structure_result.issues)
