from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from backend.app import API_VERSION, create_app
from backend.workers import TrainingWorkerBusyError, TrainingWorkerStartError
from rl_core.generator import generate_scenario
from rl_core.models import EvaluationRun, MaskablePPOTrainingConfig, RunStatus, TrainingRun
from rl_core.storage import StorageRepository


def test_health_version_and_empty_scenario_list(tmp_path: Path) -> None:
    client = TestClient(create_app(data_root=tmp_path / "data"))

    health = client.get("/api/health")
    version = client.get("/api/version")
    scenarios = client.get("/api/scenarios")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "storage_schema_version": 1}
    assert version.json() == {"api_version": API_VERSION, "storage_schema_version": 1}
    assert scenarios.json() == {"items": []}


def test_scenario_list_and_detail_use_storage_repository(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    older = generate_scenario(seed=1, size="tiny")
    newer = generate_scenario(seed=2, size="tiny")
    repository.save_scenario(older)
    repository.save_scenario(newer)
    client = TestClient(create_app(repository=repository))

    response = client.get("/api/scenarios")
    detail = client.get(f"/api/scenarios/{newer.scenario_id}")

    assert response.status_code == 200
    assert {item["scenario_id"] for item in response.json()["items"]} == {
        older.scenario_id,
        newer.scenario_id,
    }
    assert all(
        {"scenario_id", "name", "seed", "created_at", "updated_at"} <= item.keys()
        for item in response.json()["items"]
    )
    assert detail.status_code == 200
    assert detail.json() == newer.model_dump(mode="json")


def test_scenario_errors_use_structured_responses(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    path = repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))

    missing = client.get("/api/scenarios/does-not-exist")
    (repository.data_root / path).unlink()
    missing_artifact = client.get(f"/api/scenarios/{scenario.scenario_id}")

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "scenario_not_found"
    assert missing_artifact.status_code == 409
    assert missing_artifact.json()["error"]["code"] == "scenario_artifact_missing"


def test_order_strip_and_opportunity_endpoints_filter_and_paginate(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))
    selected_order = scenario.orders[0]
    selected_strip = next(
        strip for strip in scenario.strips if strip.order_id == selected_order.order_id
    )
    selected_opportunity = next(
        opportunity
        for opportunity in scenario.opportunities
        if opportunity.strip_id == selected_strip.strip_id
    )

    orders = client.get(f"/api/scenarios/{scenario.scenario_id}/orders?limit=2")
    strips = client.get(
        f"/api/scenarios/{scenario.scenario_id}/strips",
        params={"order_id": selected_order.order_id, "limit": 100},
    )
    opportunities = client.get(
        f"/api/scenarios/{scenario.scenario_id}/opportunities",
        params={
            "strip_id": selected_strip.strip_id,
            "pass_id": selected_opportunity.pass_id,
            "kind": selected_opportunity.kind.value,
            "limit": 1,
        },
    )

    assert orders.status_code == 200
    assert orders.json()["total"] == len(scenario.orders)
    assert len(orders.json()["items"]) == 2
    assert {"strip_count", "opportunity_count"} <= orders.json()["items"][0].keys()
    assert strips.status_code == 200
    assert strips.json()["total"] == len(
        [strip for strip in scenario.strips if strip.order_id == selected_order.order_id]
    )
    assert all(item["order_id"] == selected_order.order_id for item in strips.json()["items"])
    assert opportunities.status_code == 200
    assert opportunities.json()["total"] >= 1
    assert len(opportunities.json()["items"]) == 1
    assert opportunities.json()["items"][0]["off_nadir_deg"] >= 0.0


def test_collection_query_validation_and_empty_filtered_page(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))

    invalid = client.get(f"/api/scenarios/{scenario.scenario_id}/opportunities?limit=0")
    empty = client.get(
        f"/api/scenarios/{scenario.scenario_id}/opportunities",
        params={"pass_id": "not-a-pass", "offset": 500, "limit": 10},
    )

    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "request_validation_error"
    assert empty.status_code == 200
    assert empty.json() == {"items": [], "offset": 500, "limit": 10, "total": 0}


def test_scenario_validation_reports_valid_checksum_and_structure_results(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=20260707, size="tiny")
    relative_path = repository.save_scenario(scenario)
    artifact_path = repository.data_root / relative_path
    client = TestClient(create_app(repository=repository))

    valid = client.get(f"/api/scenarios/{scenario.scenario_id}/validation")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["name"] = "changed outside repository"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    checksum = client.get(f"/api/scenarios/{scenario.scenario_id}/validation")
    artifact_path.write_text(json.dumps({"scenario_id": scenario.scenario_id}), encoding="utf-8")
    structure = client.get(f"/api/scenarios/{scenario.scenario_id}/validation")

    assert valid.json() == {"scenario_id": scenario.scenario_id, "valid": True, "issues": []}
    assert checksum.status_code == 200
    assert checksum.json()["valid"] is False
    assert checksum.json()["issues"][0]["code"] == "artifact_checksum_mismatch"
    assert structure.status_code == 200
    assert any(issue["location"] == ["name"] for issue in structure.json()["issues"])


def test_baseline_evaluation_persists_run_summary_and_replay(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))

    response = client.post(
        "/api/evaluation-runs",
        json={
            "scenario_id": scenario.scenario_id,
            "policy_name": "priority_greedy",
            "seed": 17,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    run = repository.load_evaluation_run(payload["run"]["run_id"])
    summary_path = Path(payload["run"]["result_path"])
    replay_path = Path(payload["summary"]["replay_path"])
    assert run.status.value == "completed"
    assert run.policy_name == "priority_greedy"
    assert (
        payload["summary"]["total_return"]
        == payload["summary"]["priority_score"]
        + payload["summary"]["angle_bonus"]
        + payload["summary"]["missed_penalty"]
    )
    assert repository.load_json_artifact(summary_path)["policy_name"] == "priority_greedy"
    assert repository.load_json_artifact(replay_path)["policy_name"] == "priority_greedy"


def test_baseline_evaluation_rejects_unknown_scenario_and_policy(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))

    unknown_scenario = client.post(
        "/api/evaluation-runs",
        json={"scenario_id": "does-not-exist", "policy_name": "random_valid", "seed": 1},
    )
    unknown_policy = client.post(
        "/api/evaluation-runs",
        json={"scenario_id": scenario.scenario_id, "policy_name": "unknown", "seed": 1},
    )

    assert unknown_scenario.status_code == 404
    assert unknown_scenario.json()["error"]["code"] == "scenario_not_found"
    assert unknown_policy.status_code == 422
    assert unknown_policy.json()["error"]["code"] == "unsupported_baseline_policy"


def test_baseline_evaluation_records_failed_run_when_policy_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))

    def fail_evaluation(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("simulated evaluation failure")

    monkeypatch.setattr("backend.app.uuid4", lambda: SimpleNamespace(hex="failure"))
    monkeypatch.setattr("backend.app.evaluate_policy", fail_evaluation)

    response = client.post(
        "/api/evaluation-runs",
        json={"scenario_id": scenario.scenario_id, "policy_name": "random_valid", "seed": 1},
    )

    failed_run = repository.load_evaluation_run("evaluation-failure")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "baseline_evaluation_failed"
    assert failed_run.status.value == "failed"
    assert failed_run.error_message == "simulated evaluation failure"


def test_evaluation_result_run_status_and_timeline_are_loaded_from_artifacts(
    tmp_path: Path,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))

    created = client.post(
        "/api/evaluation-runs",
        json={
            "scenario_id": scenario.scenario_id,
            "policy_name": "priority_greedy",
            "seed": 17,
        },
    ).json()
    run_id = created["run"]["run_id"]
    status = client.get(f"/api/evaluation-runs/{run_id}")
    result = client.get(f"/api/results/{run_id}")
    timeline = client.get(f"/api/results/{run_id}/timeline", params={"limit": 1})
    episodes = client.get(f"/api/results/{run_id}/episodes")
    steps = client.get(f"/api/results/{run_id}/episodes/evaluation/steps", params={"limit": 1})
    step = client.get(f"/api/results/{run_id}/episodes/evaluation/steps/0")
    replay = repository.load_json_artifact(Path(created["summary"]["replay_path"]))

    assert status.json() == {"run": created["run"]}
    assert result.json() == {"run": created["run"], "summary": created["summary"]}
    assert timeline.status_code == 200
    assert timeline.json()["total"] == len(replay["schedule"])
    assert timeline.json()["items"] == replay["schedule"][:1]
    assert timeline.json()["offset"] == 0
    assert timeline.json()["limit"] == 1
    assert episodes.status_code == 200
    assert episodes.json()["run"] == created["run"]
    assert episodes.json()["items"] == [
        {
            "episode_id": "evaluation",
            "policy_name": replay["policy_name"],
            "scenario_id": replay["scenario_id"],
            "seed": replay["seed"],
            "steps": len(replay["steps"]),
            "captures": len(replay["schedule"]),
            "total_return": replay["total_return"],
            "completed_strips": replay["completed_strips"],
            "completed_orders": replay["completed_orders"],
        }
    ]
    assert steps.status_code == 200
    assert steps.json()["episode"] == episodes.json()["items"][0]
    assert steps.json()["total"] == len(replay["steps"])
    assert steps.json()["items"] == replay["steps"][:1]
    assert step.status_code == 200
    assert step.json() == replay["steps"][0]
    missing_step = client.get(f"/api/results/{run_id}/episodes/evaluation/steps/99999")
    assert missing_step.status_code == 404
    assert missing_step.json()["error"]["code"] == "episode_step_not_found"


def test_training_and_evaluation_run_lists_filter_and_paginate(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    training_runs = [
        TrainingRun(
            run_id=f"training-{index}",
            scenario_id=scenario.scenario_id,
            algorithm="maskable_ppo",
            seed=index,
            total_timesteps=8,
            status=status,
        )
        for index, status in enumerate((RunStatus.COMPLETED, RunStatus.RUNNING), start=1)
    ]
    evaluation_runs = [
        EvaluationRun(
            run_id=f"evaluation-{index}",
            scenario_id=scenario.scenario_id,
            policy_name="random_valid",
            seed=index,
            status=status,
        )
        for index, status in enumerate((RunStatus.COMPLETED, RunStatus.FAILED), start=1)
    ]
    for run in [*training_runs, *evaluation_runs]:
        if isinstance(run, TrainingRun):
            repository.save_training_run(run)
        else:
            repository.save_evaluation_run(run)
    client = TestClient(create_app(repository=repository))

    training = client.get("/api/training-runs", params={"limit": 1})
    running = client.get("/api/training-runs", params={"status": "running"})
    evaluations = client.get("/api/evaluation-runs", params={"limit": 1})
    completed = client.get("/api/evaluation-runs", params={"status": "completed"})

    assert training.status_code == 200
    assert training.json()["total"] == 2
    assert len(training.json()["items"]) == 1
    assert running.json()["items"] == [training_runs[1].model_dump(mode="json")]
    assert evaluations.status_code == 200
    assert evaluations.json()["total"] == 2
    assert len(evaluations.json()["items"]) == 1
    assert completed.json()["items"] == [evaluation_runs[0].model_dump(mode="json")]


def test_evaluation_episode_steps_reject_unknown_episode_and_reuse_result_validation(
    tmp_path: Path,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=20260707, size="tiny")
    repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))
    created = client.post(
        "/api/evaluation-runs",
        json={
            "scenario_id": scenario.scenario_id,
            "policy_name": "random_valid",
            "seed": 17,
        },
    ).json()
    run_id = created["run"]["run_id"]

    unknown_episode = client.get(f"/api/results/{run_id}/episodes/not-stored/steps")
    replay_path = repository.data_root / created["summary"]["replay_path"]
    replay_path.write_text("{}", encoding="utf-8")
    invalid_replay = client.get(f"/api/results/{run_id}/episodes")

    assert unknown_episode.status_code == 404
    assert unknown_episode.json()["error"]["code"] == "episode_not_found"
    assert invalid_replay.status_code == 409
    assert invalid_replay.json()["error"]["code"] == "evaluation_artifact_invalid"


def test_evaluation_result_distinguishes_run_state_missing_and_invalid_artifacts(
    tmp_path: Path,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    pending = EvaluationRun(
        run_id="pending-evaluation",
        scenario_id=scenario.scenario_id,
        policy_name="random_valid",
        seed=1,
        status=RunStatus.RUNNING,
    )
    missing = pending.model_copy(update={"run_id": "missing-result", "status": RunStatus.COMPLETED})
    failed = pending.model_copy(update={"run_id": "failed-evaluation", "status": RunStatus.FAILED})
    repository.save_evaluation_run(pending)
    repository.save_evaluation_run(missing)
    repository.save_evaluation_run(failed)
    client = TestClient(create_app(repository=repository))

    not_found = client.get("/api/results/not-found")
    not_ready = client.get("/api/results/pending-evaluation")
    no_result = client.get("/api/results/missing-result")
    not_completed = client.get("/api/results/failed-evaluation")

    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "evaluation_run_not_found"
    assert not_ready.status_code == 409
    assert not_ready.json()["error"]["code"] == "evaluation_result_not_ready"
    assert no_result.status_code == 409
    assert no_result.json()["error"]["code"] == "evaluation_result_missing"
    assert not_completed.status_code == 409
    assert not_completed.json()["error"]["code"] == "evaluation_run_not_completed"

    created = client.post(
        "/api/evaluation-runs",
        json={
            "scenario_id": scenario.scenario_id,
            "policy_name": "random_valid",
            "seed": 1,
        },
    ).json()
    summary_path = repository.data_root / created["run"]["result_path"]
    summary_path.unlink()
    artifact_missing = client.get(f"/api/results/{created['run']['run_id']}")

    created_invalid = client.post(
        "/api/evaluation-runs",
        json={
            "scenario_id": scenario.scenario_id,
            "policy_name": "random_valid",
            "seed": 2,
        },
    ).json()
    invalid_path = repository.data_root / created_invalid["run"]["result_path"]
    invalid_path.write_text("{}", encoding="utf-8")
    invalid = client.get(f"/api/results/{created_invalid['run']['run_id']}")

    assert artifact_missing.status_code == 409
    assert artifact_missing.json()["error"]["code"] == "evaluation_artifact_missing"
    assert invalid.status_code == 409
    assert invalid.json()["error"]["code"] == "evaluation_artifact_invalid"


def test_training_run_queues_server_owned_config_and_starts_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    app = create_app(repository=repository)
    client = TestClient(app)
    started_run_ids: list[str] = []
    monkeypatch.setattr(app.state.training_supervisor, "start", started_run_ids.append)

    response = client.post(
        "/api/training-runs",
        json={
            "scenario_id": scenario.scenario_id,
            "config": {
                "total_timesteps": 8,
                "learning_seed": 17,
                "evaluation_seed": 23,
                "n_steps": 4,
                "batch_size": 4,
                "checkpoint_interval": 4,
                "evaluation_interval": 4,
                "artifact_root": "outside-server-control",
            },
        },
    )

    assert response.status_code == 202
    run = response.json()["run"]
    stored_run = repository.load_training_run(run["run_id"])
    config = repository.load_json_artifact(Path("runs") / run["run_id"] / "config.json")
    assert started_run_ids == [run["run_id"]]
    assert run["status"] == "queued"
    assert stored_run.artifact_directory == f"runs/{run['run_id']}"
    assert config["artifact_root"] == str(repository.data_root / "runs")


@pytest.mark.parametrize(
    ("worker_error", "expected_code", "expected_status"),
    [
        (TrainingWorkerBusyError("busy"), "training_worker_busy", "failed"),
        (TrainingWorkerStartError("start failed"), "training_worker_start_failed", "failed"),
    ],
)
def test_training_run_worker_start_errors_are_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_error: RuntimeError,
    expected_code: str,
    expected_status: str,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    app = create_app(repository=repository)
    client = TestClient(app)
    monkeypatch.setattr("backend.app.uuid4", lambda: SimpleNamespace(hex="training-error"))

    def raise_worker_error(_: str) -> None:
        raise worker_error

    monkeypatch.setattr(app.state.training_supervisor, "start", raise_worker_error)
    response = client.post(
        "/api/training-runs",
        json={
            "scenario_id": scenario.scenario_id,
            "config": {
                "total_timesteps": 8,
                "learning_seed": 17,
                "evaluation_seed": 23,
                "n_steps": 4,
                "batch_size": 4,
            },
        },
    )

    stored_run = repository.load_training_run("training-training-error")
    assert response.status_code == (409 if expected_code == "training_worker_busy" else 500)
    assert response.json()["error"]["code"] == expected_code
    assert stored_run.status.value == expected_status
    assert stored_run.error_message == str(worker_error)


def test_training_stop_request_transitions_and_rejects_terminal_runs(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    queued_run = TrainingRun(
        run_id="queued-training",
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=1,
        total_timesteps=8,
    )
    running_run = queued_run.model_copy(
        update={"run_id": "running-training", "status": RunStatus.RUNNING}
    )
    completed_run = queued_run.model_copy(
        update={"run_id": "completed-training", "status": RunStatus.COMPLETED}
    )
    for run in (queued_run, running_run, completed_run):
        repository.save_training_run(run)
    client = TestClient(create_app(repository=repository))

    missing = client.post("/api/training-runs/missing/stop")
    queued_response = client.post("/api/training-runs/queued-training/stop")
    requested = client.post("/api/training-runs/running-training/stop")
    repeated = client.post("/api/training-runs/running-training/stop")
    terminal = client.post("/api/training-runs/completed-training/stop")

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "training_run_not_found"
    assert queued_response.status_code == 202
    assert queued_response.json()["run"]["status"] == "stopped"
    assert requested.status_code == 202
    assert requested.json()["run"]["status"] == "stop_requested"
    assert repeated.json() == requested.json()
    assert terminal.status_code == 409
    assert terminal.json()["error"]["code"] == "training_run_not_stoppable"


def test_training_run_status_and_metrics_support_pagination_and_active_partial_line(
    tmp_path: Path,
) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    completed_run = TrainingRun(
        run_id="metrics-training",
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=17,
        total_timesteps=8,
        status=RunStatus.COMPLETED,
    )
    active_run = completed_run.model_copy(
        update={"run_id": "active-training", "status": RunStatus.RUNNING}
    )
    repository.save_training_run(completed_run)
    repository.save_training_run(active_run)
    metrics_path = repository.data_root / "runs" / completed_run.run_id / "metrics"
    metrics_path.mkdir(parents=True)
    metric_rows = [
        {
            "timesteps": timestep,
            "evaluation": {
                "policy_name": "maskable_ppo",
                "scenario_id": scenario.scenario_id,
                "seed": 23,
                "steps": 10,
                "captures": 2,
                "total_return": 1.5,
                "priority_score": 1.6,
                "angle_bonus": 0.1,
                "missed_penalty": -0.2,
                "completed_strips": 2,
                "completed_orders": 1,
            },
        }
        for timestep in (4, 8)
    ]
    (metrics_path / "training-metrics.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in metric_rows), encoding="utf-8"
    )
    active_metrics_path = repository.data_root / "runs" / active_run.run_id / "metrics"
    active_metrics_path.mkdir(parents=True)
    (active_metrics_path / "training-metrics.jsonl").write_text('{"timesteps":', encoding="utf-8")
    client = TestClient(create_app(repository=repository))

    status = client.get(f"/api/training-runs/{completed_run.run_id}")
    metrics = client.get(f"/api/training-runs/{completed_run.run_id}/metrics", params={"limit": 1})
    active_partial = client.get(f"/api/training-runs/{active_run.run_id}/metrics")
    repository.save_training_run(active_run.model_copy(update={"status": RunStatus.COMPLETED}))
    invalid = client.get(f"/api/training-runs/{active_run.run_id}/metrics")

    assert status.json() == {"run": completed_run.model_dump(mode="json")}
    assert metrics.status_code == 200
    assert metrics.json()["total"] == 2
    assert metrics.json()["items"] == metric_rows[:1]
    assert active_partial.json()["items"] == []
    assert invalid.status_code == 409
    assert invalid.json()["error"]["code"] == "training_metrics_invalid"


def test_training_run_detail_restores_config_and_artifact_summary(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    run = TrainingRun(
        run_id="detail-training",
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=17,
        total_timesteps=8,
        status=RunStatus.COMPLETED,
        artifact_directory="runs/detail-training",
    )
    config = MaskablePPOTrainingConfig(
        total_timesteps=8,
        learning_seed=17,
        evaluation_seed=23,
        n_steps=4,
        batch_size=4,
    )
    repository.save_training_config(run, config)
    run_root = repository.data_root / "runs" / run.run_id
    (run_root / "checkpoints").mkdir(parents=True)
    (run_root / "checkpoints" / "checkpoint-4.zip").write_bytes(b"checkpoint")
    (run_root / "model").mkdir(parents=True)
    (run_root / "model" / "final-model.zip").write_bytes(b"model")
    (run_root / "metrics").mkdir(parents=True)
    (run_root / "metrics" / "final-evaluation.json").write_text("{}", encoding="utf-8")
    client = TestClient(create_app(repository=repository))

    response = client.get(f"/api/training-runs/{run.run_id}/detail")

    assert response.status_code == 200
    assert response.json()["run"] == run.model_dump(mode="json")
    assert response.json()["config"]["learning_seed"] == 17
    assert response.json()["checkpoints"] == ["checkpoint-4.zip"]
    assert response.json()["final_model_available"] is True
    assert response.json()["final_evaluation_available"] is True


def test_policy_comparison_persists_selected_same_seed_evaluations(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "data")
    scenario = generate_scenario(seed=1, size="tiny")
    repository.save_scenario(scenario)
    client = TestClient(create_app(repository=repository))
    first = client.post(
        "/api/evaluation-runs",
        json={"scenario_id": scenario.scenario_id, "policy_name": "random_valid", "seed": 17},
    ).json()
    second = client.post(
        "/api/evaluation-runs",
        json={"scenario_id": scenario.scenario_id, "policy_name": "priority_greedy", "seed": 17},
    ).json()
    response = client.post(
        "/api/policy-comparisons",
        json={
            "scenario_id": scenario.scenario_id,
            "seed": 17,
            "evaluation_run_ids": [first["run"]["run_id"], second["run"]["run_id"]],
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert {item["policy_name"] for item in payload["result"]["entries"]} == {
        "random_valid",
        "priority_greedy",
    }
    loaded = client.get(f"/api/policy-comparisons/{payload['comparison']['comparison_id']}")
    assert loaded.json() == payload
