"""FastAPI 요청과 분리해 단일 Maskable PPO 학습 worker를 관리한다."""

from __future__ import annotations

import multiprocessing
from multiprocessing.process import BaseProcess
from pathlib import Path
from threading import Lock

from rl_core.models import EvaluationSummary, MaskablePPOTrainingConfig, RunStatus
from rl_core.optimization import solve_cp_sat_baseline
from rl_core.storage import StorageRepository
from rl_core.training import TrainingStoppedError, train_maskable_ppo


class TrainingWorkerBusyError(RuntimeError):
    """이미 실행 중인 로컬 학습 worker가 있을 때 발생한다."""


class TrainingWorkerStartError(RuntimeError):
    """학습 worker process를 시작하지 못했을 때 발생한다."""


class TrainingWorkerSupervisor:
    """한 Backend process 안에서 동시에 하나의 학습 worker만 시작하게 한다.

    초기 프로토타입은 자원 경합과 여러 PPO 학습의 결과 혼합을 피하기 위해 단일
    worker만 허용한다. DB 상태 복구는 이 supervisor가 worker 부재를 확인한 별도
    시작 절차에서만 수행해야 하며, FastAPI app factory는 이를 임의로 호출하지 않는다.
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self._context = multiprocessing.get_context("spawn")
        self._lock = Lock()
        self._process: BaseProcess | None = None

    def start(self, run_id: str) -> None:
        """저장된 queued run을 별도 non-daemon process에서 시작한다."""

        with self._lock:
            if self._process is not None and self._process.is_alive():
                raise TrainingWorkerBusyError("A local training worker is already running.")
            self._process = None
            process = self._context.Process(
                target=run_training_worker,
                args=(str(self.data_root), run_id),
                daemon=False,
            )
            try:
                process.start()
            except Exception as error:
                raise TrainingWorkerStartError(
                    "Could not start training worker process."
                ) from error
            self._process = process

    def start_cp_sat(self, run_id: str, time_limit_sec: float) -> None:
        """PPO와 같은 단일 실행 슬롯에서 CP-SAT 평가 worker를 시작한다."""

        with self._lock:
            if self._process is not None and self._process.is_alive():
                raise TrainingWorkerBusyError("A local execution worker is already running.")
            process = self._context.Process(
                target=run_cp_sat_worker,
                args=(str(self.data_root), run_id, time_limit_sec),
                daemon=False,
            )
            try:
                process.start()
            except Exception as error:
                raise TrainingWorkerStartError("Could not start CP-SAT worker process.") from error
            self._process = process


def run_training_worker(data_root: str, run_id: str) -> None:
    """저장된 run snapshot을 복원해 PPO 학습을 실행하고 예외 상태를 보존한다."""

    repository = StorageRepository(Path(data_root))
    try:
        run = repository.load_training_run(run_id)
        if run.status is RunStatus.STOPPED:
            return
        if run.status is RunStatus.STOP_REQUESTED:
            repository.save_training_run(run.model_copy(update={"status": RunStatus.STOPPED}))
            return
        if run.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
            raise RuntimeError(f"training run cannot be started from status: {run.status}")
        scenario = repository.load_scenario(run.scenario_id)
        config_payload = repository.load_checked_json_artifact(
            relative_path=Path("runs") / run_id / "config.json",
            artifact_type="training_config",
            owner_type="training_run",
            owner_id=run_id,
        )
        config = MaskablePPOTrainingConfig.model_validate(config_payload)
        train_maskable_ppo(
            scenario,
            config,
            run_id=run_id,
            storage=repository,
            should_stop=lambda: _is_stop_requested(repository, run_id),
        )
    except TrainingStoppedError:
        return
    except Exception as error:
        _record_worker_failure(repository, run_id, str(error))


def _record_worker_failure(repository: StorageRepository, run_id: str, message: str) -> None:
    """trainer 시작 전 실패도 queued/running run에 남겨 polling 시 원인을 보이게 한다."""

    try:
        run = repository.load_training_run(run_id)
        if run.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
            repository.save_training_run(
                run.model_copy(update={"status": RunStatus.FAILED, "error_message": message})
            )
    except Exception:
        # 저장소 손상 때문에 worker가 다시 예외를 내며 traceback만 남기는 것을 막는다.
        return


def _is_stop_requested(repository: StorageRepository, run_id: str) -> bool:
    """callback이 SQLite 상태만 읽어 cooperative cancellation 여부를 판단하게 한다."""

    return repository.load_training_run(run_id).status is RunStatus.STOP_REQUESTED


def run_cp_sat_worker(data_root: str, run_id: str, time_limit_sec: float) -> None:
    """queued CP-SAT EvaluationRun을 풀고 공통 결과 artifact로 색인한다."""

    repository = StorageRepository(Path(data_root))
    try:
        run = repository.load_evaluation_run(run_id)
        if run.status is not RunStatus.QUEUED:
            raise RuntimeError(f"CP-SAT evaluation cannot start from status: {run.status}")
        run = run.model_copy(update={"status": RunStatus.RUNNING})
        repository.save_evaluation_run(run)
        scenario = repository.load_scenario(run.scenario_id)
        result = solve_cp_sat_baseline(scenario, seed=run.seed, time_limit_sec=time_limit_sec)
        if result.status not in {"OPTIMAL", "FEASIBLE"}:
            raise RuntimeError(f"CP-SAT returned no feasible solution: {result.status}")
        captures = len(result.replay.schedule)
        opportunities = {item.opportunity_id: item for item in scenario.opportunities}
        root = Path("evaluations") / run_id
        average_off_nadir = (
            sum(opportunities[item.opportunity_id].off_nadir_deg for item in result.replay.schedule)
            / captures
            if captures
            else 0.0
        )
        summary = EvaluationSummary(
            policy_name=result.replay.policy_name,
            scenario_id=scenario.scenario_id,
            seed=run.seed,
            steps=len(result.replay.steps),
            captures=captures,
            total_return=result.replay.total_return,
            priority_score=sum(item.reward_breakdown.strip_base for item in result.replay.steps),
            angle_bonus=sum(item.reward_breakdown.angle_bonus for item in result.replay.steps),
            missed_penalty=sum(
                item.reward_breakdown.missed_penalty for item in result.replay.steps
            ),
            completed_strips=result.replay.completed_strips,
            completed_orders=result.replay.completed_orders,
            average_off_nadir_deg=average_off_nadir,
            replay_path=(root / "replay.json").as_posix(),
        )
        repository.save_json_artifact(
            artifact_type="optimization_baseline",
            owner_type="evaluation_run",
            owner_id=run_id,
            scenario_id=scenario.scenario_id,
            relative_path=root / "optimization.json",
            payload=result.model_dump(mode="json"),
        )
        repository.save_completed_evaluation(run, summary, result.replay)
    except Exception as error:
        try:
            run = repository.load_evaluation_run(run_id)
            if run.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
                repository.save_evaluation_run(
                    run.model_copy(update={"status": RunStatus.FAILED, "error_message": str(error)})
                )
        except Exception:
            pass
