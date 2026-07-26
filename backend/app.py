"""저장된 시나리오를 조회하는 초기 FastAPI Backend app factory다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from backend.workers import (
    TrainingWorkerBusyError,
    TrainingWorkerStartError,
    TrainingWorkerSupervisor,
)
from rl_core.models import (
    EpisodeReplay,
    EvaluationRun,
    EvaluationSummary,
    MaskablePPOTrainingConfig,
    Opportunity,
    OpportunityKind,
    Order,
    PolicyComparison,
    PolicyComparisonRun,
    Priority,
    ReplayCapture,
    ReplayStep,
    RunStatus,
    Scenario,
    Strip,
    TrainingRun,
)
from rl_core.optimization import CP_SAT_POLICY_NAME
from rl_core.policies import (
    BaselinePolicy,
    EarliestDeadlineFirstPolicy,
    EvaluationResult,
    PriorityEfficiencyGreedyPolicy,
    PriorityGreedyPolicy,
    RandomValidPolicy,
    evaluate_policy,
)
from rl_core.replay import policy_comparison, policy_comparison_entry
from rl_core.storage import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ScenarioSummary,
    ScenarioValidationResult,
    StorageRepository,
)

API_VERSION = "0.1.0"
BASELINE_POLICIES: dict[str, type[BaselinePolicy]] = {
    RandomValidPolicy.name: RandomValidPolicy,
    EarliestDeadlineFirstPolicy.name: EarliestDeadlineFirstPolicy,
    PriorityGreedyPolicy.name: PriorityGreedyPolicy,
    PriorityEfficiencyGreedyPolicy.name: PriorityEfficiencyGreedyPolicy,
}
EVALUATION_EPISODE_ID = "evaluation"


class ErrorDetail(BaseModel):
    """모든 API 오류가 공유하는 기계 판독 가능한 코드와 설명이다."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Frontend가 HTTP 상태와 별개로 오류를 일관되게 표시하기 위한 형식이다."""

    error: ErrorDetail


class ScenarioSummaryResponse(BaseModel):
    """시나리오 목록 항목의 조회 전용 API 표현이다."""

    scenario_id: str
    name: str
    seed: int
    created_at: str
    updated_at: str


class ScenarioListResponse(BaseModel):
    """시나리오 목록 API의 안정적인 최상위 응답 형식이다."""

    items: list[ScenarioSummaryResponse]


class ScenarioValidationIssueResponse(BaseModel):
    """Frontend가 저장된 Scenario의 어느 경로가 잘못됐는지 표시하는 항목이다."""

    code: str
    location: list[str | int]
    message: str


class ScenarioValidationResponse(BaseModel):
    """저장된 Scenario artifact의 해시와 구조 검증 결과다."""

    scenario_id: str
    valid: bool
    issues: list[ScenarioValidationIssueResponse]


class BaselineEvaluationRequest(BaseModel):
    """동기 기준 정책 평가를 시작하기 위한 최소 입력이다."""

    scenario_id: str
    policy_name: str
    seed: int


class TrainingRunRequest(BaseModel):
    """학습 시작 요청에서 서버가 관리하지 않는 artifact 경로를 제외한 설정이다."""

    scenario_id: str
    config: MaskablePPOTrainingConfig


class PolicyComparisonRequest(BaseModel):
    """같은 scenario·seed의 완료 evaluation run만 비교 artifact로 고정한다."""

    scenario_id: str
    seed: int
    evaluation_run_ids: list[str]


class CpSatEvaluationRequest(BaseModel):
    """시간 제한을 둔 CP-SAT 기준해 평가의 입력이다."""

    scenario_id: str
    seed: int
    time_limit_sec: float = 10.0


class EvaluationSummaryResponse(BaseModel):
    """기준 정책과 PPO 평가가 공통으로 제공할 도메인 성능 요약이다."""

    policy_name: str
    scenario_id: str
    seed: int
    steps: int
    captures: int
    total_return: float
    priority_score: float
    angle_bonus: float
    missed_penalty: float
    completed_strips: int
    completed_orders: int
    average_off_nadir_deg: float
    replay_path: str


class BaselineEvaluationResponse(BaseModel):
    """완료된 EvaluationRun과 저장된 replay 요약을 함께 반환하는 응답이다."""

    run: EvaluationRun
    summary: EvaluationSummaryResponse


class EvaluationRunResponse(BaseModel):
    """실행 중인 worker도 polling할 수 있도록 EvaluationRun 상태만 반환한다."""

    run: EvaluationRun


class TrainingRunResponse(BaseModel):
    """비동기 worker에 인계된 학습 run의 초기 상태 응답이다."""

    run: TrainingRun


class TrainingRunDetailResponse(BaseModel):
    """학습 제어 화면이 재시작 후에도 복원할 수 있는 run snapshot과 산출물 요약이다."""

    run: TrainingRun
    config: MaskablePPOTrainingConfig
    checkpoints: list[str]
    final_model_available: bool
    final_evaluation_available: bool


class TrainingMetricEvaluationResponse(BaseModel):
    """학습 곡선에 필요한 고정 시나리오 평가 요약 지표다."""

    policy_name: str
    scenario_id: str
    seed: int
    steps: int
    captures: int
    total_return: float
    priority_score: float
    angle_bonus: float
    missed_penalty: float
    completed_strips: int
    completed_orders: int


class TrainingMetricResponse(BaseModel):
    """training-metrics.jsonl 한 행을 화면용으로 검증·축약한 DTO다."""

    timesteps: int
    evaluation: TrainingMetricEvaluationResponse


class EvaluationResultResponse(BaseModel):
    """완료된 실행의 metadata와 검증된 요약 artifact를 함께 반환한다."""

    run: EvaluationRun
    summary: EvaluationSummaryResponse


class PolicyComparisonResponse(BaseModel):
    """비교 metadata와 검증된 정책 요약 artifact를 함께 반환한다."""

    comparison: PolicyComparisonRun
    result: PolicyComparison


class PaginationResponse(BaseModel):
    """대용량 시나리오 하위 리소스 조회가 공유하는 페이지 정보다."""

    offset: int
    limit: int
    total: int


class TrainingRunListResponse(PaginationResponse):
    """대시보드가 최근 학습 실행 metadata만 페이지로 읽는 응답이다."""

    items: list[TrainingRun]


class EvaluationRunListResponse(PaginationResponse):
    """대시보드와 결과 목록이 최근 평가 실행 metadata를 읽는 응답이다."""

    items: list[EvaluationRun]


class TrainingMetricListResponse(PaginationResponse):
    """현재 저장된 학습 평가 행을 pagination으로 반환하는 응답이다."""

    run: TrainingRun
    items: list[TrainingMetricResponse]


class TimelineCaptureResponse(ReplayCapture):
    """지도·타임라인 화면이 필요한 촬영 schedule 항목만 전달하는 DTO다."""


class TimelineResponse(PaginationResponse):
    """저장된 replay schedule을 시간순으로 나눈 타임라인 응답이다."""

    run: EvaluationRun
    items: list[TimelineCaptureResponse]


class EpisodeSummaryResponse(BaseModel):
    """평가 run에 저장된 재생 episode를 선택하기 위한 가벼운 요약 DTO다."""

    episode_id: str
    policy_name: str
    scenario_id: str
    seed: int
    steps: int
    captures: int
    total_return: float
    completed_strips: int
    completed_orders: int


class EpisodeListResponse(BaseModel):
    """현재 EvaluationRun이 제공하는 재생 episode 목록이다."""

    run: EvaluationRun
    items: list[EpisodeSummaryResponse]


class EpisodeStepListResponse(PaginationResponse):
    """선택한 재생 episode의 상세 step 로그를 나눈 응답이다."""

    run: EvaluationRun
    episode: EpisodeSummaryResponse
    items: list[ReplayStep]


class OrderListItem(Order):
    """주문 목록에서 해당 주문의 하위 데이터 규모를 함께 보여 주는 표현이다."""

    strip_count: int
    opportunity_count: int


class OrderListResponse(PaginationResponse):
    """필터된 주문 목록과 pagination 메타데이터다."""

    items: list[OrderListItem]


class StripListItem(Strip):
    """strip 목록에서 연결된 촬영 후보 수를 함께 제공하는 표현이다."""

    opportunity_count: int


class StripListResponse(PaginationResponse):
    """필터된 strip 목록과 pagination 메타데이터다."""

    items: list[StripListItem]


class OpportunityListItem(BaseModel):
    """Frontend가 자세 계산을 중복하지 않도록 off-nadir 값을 포함한 후보 표현이다."""

    opportunity_id: str
    order_id: str
    strip_id: str
    pass_id: str
    kind: OpportunityKind
    window_start_sec: float
    window_end_sec: float
    capture_time_sec: float
    required_roll_deg: float
    required_tilt_deg: float
    required_pitch_deg: float
    source_access_window_id: str | None
    off_nadir_deg: float


class OpportunityListResponse(PaginationResponse):
    """필터된 촬영 기회 목록과 pagination 메타데이터다."""

    items: list[OpportunityListItem]


class HealthResponse(BaseModel):
    """서버와 SQLite 저장소가 요청을 처리할 준비가 됐는지 나타낸다."""

    status: str = "ok"
    storage_schema_version: int


class VersionResponse(BaseModel):
    """Frontend가 API와 저장소 계약의 버전을 확인하기 위한 응답이다."""

    api_version: str
    storage_schema_version: int


class ApiError(Exception):
    """HTTP 상태와 표준 오류 본문을 함께 전달하는 Backend 내부 예외다."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


def create_app(
    *,
    data_root: Path = Path("data"),
    repository: StorageRepository | None = None,
    training_supervisor: TrainingWorkerSupervisor | None = None,
) -> FastAPI:
    """테스트와 실제 실행에서 저장소를 교체할 수 있는 FastAPI app을 만든다."""

    app = FastAPI(title="NSICPS RL Scheduling API", version=API_VERSION)
    app.state.repository = repository or StorageRepository(data_root)
    app.state.training_supervisor = training_supervisor or TrainingWorkerSupervisor(
        _repository(app).data_root
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorResponse(
                error=ErrorDetail(code=error.code, message=error.message)
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        _: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="request_validation_error",
                    message=str(error),
                )
            ).model_dump(),
        )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """SQLite schema를 읽어 Backend와 저장소의 준비 상태를 확인한다."""

        return HealthResponse(storage_schema_version=_repository(app).schema_version)

    @app.get("/api/version", response_model=VersionResponse)
    def version() -> VersionResponse:
        """API 및 SQLite 계약 버전을 Frontend에 제공한다."""

        return VersionResponse(
            api_version=API_VERSION,
            storage_schema_version=_repository(app).schema_version,
        )

    @app.get("/api/scenarios", response_model=ScenarioListResponse)
    def list_scenarios() -> ScenarioListResponse:
        """시나리오 목록 화면을 위해 가벼운 SQLite 메타데이터만 반환한다."""

        return ScenarioListResponse(
            items=[
                _scenario_summary_response(item)
                for item in _repository(app).list_scenario_summaries()
            ]
        )

    @app.get(
        "/api/scenarios/{scenario_id}",
        response_model=Scenario,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def get_scenario(scenario_id: str) -> Scenario:
        """저장된 Scenario 원본을 다시 검증해 상세 조회 응답으로 반환한다."""

        return _load_scenario_or_api_error(_repository(app), scenario_id)

    @app.get(
        "/api/scenarios/{scenario_id}/validation",
        response_model=ScenarioValidationResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def validate_scenario(scenario_id: str) -> ScenarioValidationResponse:
        """저장된 Scenario의 artifact 무결성과 구조 오류를 읽기 전용으로 보고한다."""

        try:
            return _scenario_validation_response(_repository(app).validate_scenario(scenario_id))
        except KeyError as error:
            raise ApiError(
                404, "scenario_not_found", f"Unknown scenario_id: {scenario_id}"
            ) from error
        except ArtifactNotFoundError as error:
            raise ApiError(
                409,
                "scenario_artifact_missing",
                f"Scenario artifact is missing for scenario_id: {scenario_id}",
            ) from error

    @app.post(
        "/api/evaluation-runs",
        status_code=201,
        response_model=BaselineEvaluationResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def create_baseline_evaluation(
        request: BaselineEvaluationRequest,
    ) -> BaselineEvaluationResponse:
        """기준 정책을 동기 실행하고 EvaluationRun·replay artifact로 보존한다."""

        repository = _repository(app)
        scenario = _load_valid_scenario_or_api_error(repository, request.scenario_id)
        policy = _baseline_policy_or_api_error(request.policy_name)
        run_id = f"evaluation-{uuid4().hex}"
        running_run = EvaluationRun(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            policy_name=policy.name,
            seed=request.seed,
            status=RunStatus.RUNNING,
        )
        repository.save_evaluation_run(running_run)
        replay_path = Path("evaluations") / run_id / "replay.json"
        summary_path = Path("evaluations") / run_id / "summary.json"

        try:
            result = evaluate_policy(policy, scenario, seed=request.seed)
            repository.save_json_artifact(
                artifact_type="episode_replay",
                owner_type="evaluation_run",
                owner_id=run_id,
                scenario_id=scenario.scenario_id,
                relative_path=replay_path,
                payload=result.replay.model_dump(mode="json"),
            )
            summary = _evaluation_summary_response(result, replay_path)
            repository.save_json_artifact(
                artifact_type="evaluation_summary",
                owner_type="evaluation_run",
                owner_id=run_id,
                scenario_id=scenario.scenario_id,
                relative_path=summary_path,
                payload=summary.model_dump(mode="json"),
            )
            completed_run = running_run.model_copy(
                update={"status": RunStatus.COMPLETED, "result_path": summary_path.as_posix()}
            )
            repository.save_evaluation_run(completed_run)
        except Exception as error:
            failed_run = running_run.model_copy(
                update={"status": RunStatus.FAILED, "error_message": str(error)}
            )
            repository.save_evaluation_run(failed_run)
            raise ApiError(
                500,
                "baseline_evaluation_failed",
                "Baseline evaluation failed.",
            ) from error

        return BaselineEvaluationResponse(run=completed_run, summary=summary)

    @app.post("/api/cp-sat-evaluation-runs", status_code=202, response_model=EvaluationRunResponse)
    def create_cp_sat_evaluation(request: CpSatEvaluationRequest) -> EvaluationRunResponse:
        """CP-SAT 평가를 queued 상태로 저장하고 공통 실행 worker에 인계한다."""

        repository = _repository(app)
        scenario = _load_valid_scenario_or_api_error(repository, request.scenario_id)
        run_id = f"evaluation-cp-sat-{uuid4().hex}"
        queued = EvaluationRun(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            policy_name=CP_SAT_POLICY_NAME,
            seed=request.seed,
            status=RunStatus.QUEUED,
        )
        repository.save_evaluation_run(queued)
        try:
            _training_supervisor(app).start_cp_sat(run_id, request.time_limit_sec)
        except TrainingWorkerBusyError as error:
            repository.save_evaluation_run(
                queued.model_copy(update={"status": RunStatus.FAILED, "error_message": str(error)})
            )
            raise ApiError(
                409, "execution_worker_busy", "A local execution worker is already running."
            ) from error
        except TrainingWorkerStartError as error:
            repository.save_evaluation_run(
                queued.model_copy(update={"status": RunStatus.FAILED, "error_message": str(error)})
            )
            raise ApiError(
                500, "execution_worker_start_failed", "Execution worker could not be started."
            ) from error
        return EvaluationRunResponse(run=queued)

    @app.post("/api/policy-comparisons", status_code=201, response_model=PolicyComparisonResponse)
    def create_policy_comparison(request: PolicyComparisonRequest) -> PolicyComparisonResponse:
        """사용자가 고른 완료 결과만 동일 scenario·seed 비교 artifact로 저장한다."""

        repository = _repository(app)
        if not request.evaluation_run_ids:
            raise ApiError(
                422, "comparison_runs_required", "At least one evaluation run is required."
            )
        entries = []
        for run_id in request.evaluation_run_ids:
            run = _load_completed_evaluation_run_or_api_error(repository, run_id)
            if run.scenario_id != request.scenario_id or run.seed != request.seed:
                raise ApiError(
                    422, "comparison_run_mismatch", "All runs must share scenario_id and seed."
                )
            summary = _load_evaluation_summary_or_api_error(repository, run)
            replay = _load_episode_replay_or_api_error(repository, run, summary)
            entries.append(
                policy_comparison_entry(
                    replay=replay,
                    priority_score=summary.priority_score,
                    angle_bonus=summary.angle_bonus,
                    missed_penalty=summary.missed_penalty,
                    replay_path=Path(summary.replay_path),
                    evaluation_run_id=run.run_id,
                )
            )
        result = policy_comparison(entries)
        comparison_id = f"comparison-{uuid4().hex}"
        comparison = PolicyComparisonRun(
            comparison_id=comparison_id,
            scenario_id=request.scenario_id,
            seed=request.seed,
            evaluation_run_ids=request.evaluation_run_ids,
            artifact_path=(Path("comparisons") / f"{comparison_id}.json").as_posix(),
        )
        repository.save_policy_comparison(comparison, result)
        return PolicyComparisonResponse(comparison=comparison, result=result)

    @app.get("/api/policy-comparisons/{comparison_id}", response_model=PolicyComparisonResponse)
    def get_policy_comparison(comparison_id: str) -> PolicyComparisonResponse:
        """색인된 immutable comparison artifact를 무결성 검사 후 반환한다."""

        repository = _repository(app)
        try:
            comparison = repository.load_policy_comparison_run(comparison_id)
        except KeyError as error:
            raise ApiError(
                404, "policy_comparison_not_found", f"Comparison not found: {comparison_id}"
            ) from error
        try:
            result = PolicyComparison.model_validate(
                repository.load_checked_json_artifact(
                    relative_path=Path(comparison.artifact_path),
                    artifact_type="policy_comparison",
                    owner_type="scenario",
                    owner_id=comparison.scenario_id,
                )
            )
        except (ArtifactNotFoundError, ArtifactIntegrityError, ValidationError) as error:
            raise ApiError(
                409, "policy_comparison_invalid", "Comparison artifact is unavailable or invalid."
            ) from error
        return PolicyComparisonResponse(comparison=comparison, result=result)

    @app.post(
        "/api/training-runs",
        status_code=202,
        response_model=TrainingRunResponse,
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def create_training_run(request: TrainingRunRequest) -> TrainingRunResponse:
        """학습 설정 snapshot을 저장하고 별도 worker process에 Maskable PPO 실행을 맡긴다."""

        repository = _repository(app)
        scenario = _load_valid_scenario_or_api_error(repository, request.scenario_id)
        run_id = f"training-{uuid4().hex}"
        config = request.config.model_copy(update={"artifact_root": repository.data_root / "runs"})
        queued_run = TrainingRun(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            algorithm="maskable_ppo",
            seed=config.learning_seed,
            total_timesteps=config.total_timesteps,
            status=RunStatus.QUEUED,
            artifact_directory=(Path("runs") / run_id).as_posix(),
        )
        repository.save_training_config(queued_run, config)
        try:
            _training_supervisor(app).start(run_id)
        except TrainingWorkerBusyError as error:
            repository.save_training_run(
                queued_run.model_copy(
                    update={"status": RunStatus.FAILED, "error_message": str(error)}
                )
            )
            raise ApiError(
                409,
                "training_worker_busy",
                "A local training worker is already running.",
            ) from error
        except TrainingWorkerStartError as error:
            failed_run = queued_run.model_copy(
                update={"status": RunStatus.FAILED, "error_message": str(error)}
            )
            repository.save_training_run(failed_run)
            raise ApiError(
                500,
                "training_worker_start_failed",
                "Training worker could not be started.",
            ) from error
        return TrainingRunResponse(run=queued_run)

    @app.post(
        "/api/training-runs/{run_id}/stop",
        status_code=202,
        response_model=TrainingRunResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def request_training_stop(run_id: str) -> TrainingRunResponse:
        """worker가 다음 PPO step 경계에서 관찰할 cooperative stop 요청을 저장한다."""

        repository = _repository(app)
        run = _load_training_run_or_api_error(repository, run_id)
        if run.status is RunStatus.QUEUED:
            stopped_run = run.model_copy(update={"status": RunStatus.STOPPED})
            repository.save_training_run(stopped_run)
            return TrainingRunResponse(run=stopped_run)
        if run.status is RunStatus.RUNNING:
            requested_run = run.model_copy(update={"status": RunStatus.STOP_REQUESTED})
            repository.save_training_run(requested_run)
            return TrainingRunResponse(run=requested_run)
        if run.status is RunStatus.STOP_REQUESTED:
            return TrainingRunResponse(run=run)
        raise ApiError(
            409,
            "training_run_not_stoppable",
            f"Training run cannot be stopped from status: {run.status}",
        )

    @app.get("/api/training-runs", response_model=TrainingRunListResponse)
    def list_training_runs(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        scenario_id: str | None = None,
        status: RunStatus | None = None,
    ) -> TrainingRunListResponse:
        """대시보드가 artifact 없이 최신순 학습 실행 metadata를 조회하게 한다."""

        items, total = _repository(app).list_training_runs(
            scenario_id=scenario_id, status=status, offset=offset, limit=limit
        )
        return TrainingRunListResponse(items=items, offset=offset, limit=limit, total=total)

    @app.get(
        "/api/training-runs/{run_id}",
        response_model=TrainingRunResponse,
        responses={404: {"model": ErrorResponse}},
    )
    def get_training_run(run_id: str) -> TrainingRunResponse:
        """worker process와 무관하게 SQLite에 저장된 학습 상태를 반환한다."""

        return TrainingRunResponse(run=_load_training_run_or_api_error(_repository(app), run_id))

    @app.get(
        "/api/training-runs/{run_id}/detail",
        response_model=TrainingRunDetailResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def get_training_run_detail(run_id: str) -> TrainingRunDetailResponse:
        """설정 snapshot과 파일 존재 여부만 읽어 학습 제어 화면을 복원한다."""

        repository = _repository(app)
        run = _load_training_run_or_api_error(repository, run_id)
        config_path = Path("runs") / run.run_id / "config.json"
        try:
            config = MaskablePPOTrainingConfig.model_validate(
                repository.load_json_artifact(config_path)
            )
        except (ArtifactNotFoundError, ValidationError) as error:
            raise ApiError(
                409,
                "training_config_invalid",
                f"Training config artifact is unavailable for run_id: {run_id}",
            ) from error
        run_root = repository.data_root / "runs" / run.run_id
        checkpoint_dir = run_root / "checkpoints"
        checkpoints = (
            sorted(item.name for item in checkpoint_dir.glob("checkpoint-*.zip") if item.is_file())
            if checkpoint_dir.is_dir()
            else []
        )
        return TrainingRunDetailResponse(
            run=run,
            config=config,
            checkpoints=checkpoints,
            final_model_available=(run_root / "model" / "final-model.zip").is_file(),
            final_evaluation_available=(run_root / "metrics" / "final-evaluation.json").is_file(),
        )

    @app.get(
        "/api/training-runs/{run_id}/metrics",
        response_model=TrainingMetricListResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def get_training_metrics(
        run_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> TrainingMetricListResponse:
        """append 중인 JSONL에서도 완성된 평가 행만 읽어 학습 곡선을 제공한다."""

        repository = _repository(app)
        run = _load_training_run_or_api_error(repository, run_id)
        metrics = _load_training_metrics_or_api_error(repository, run)
        return TrainingMetricListResponse(
            run=run,
            items=_page(metrics, offset, limit),
            offset=offset,
            limit=limit,
            total=len(metrics),
        )

    @app.get("/api/evaluation-runs", response_model=EvaluationRunListResponse)
    def list_evaluation_runs(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        scenario_id: str | None = None,
        status: RunStatus | None = None,
    ) -> EvaluationRunListResponse:
        """결과 화면이 최신 평가 실행을 발견하도록 metadata 목록만 반환한다."""

        items, total = _repository(app).list_evaluation_runs(
            scenario_id=scenario_id, status=status, offset=offset, limit=limit
        )
        return EvaluationRunListResponse(items=items, offset=offset, limit=limit, total=total)

    @app.get(
        "/api/evaluation-runs/{run_id}",
        response_model=EvaluationRunResponse,
        responses={404: {"model": ErrorResponse}},
    )
    def get_evaluation_run(run_id: str) -> EvaluationRunResponse:
        """결과가 아직 없을 수 있는 EvaluationRun 상태를 polling 용도로 반환한다."""

        return EvaluationRunResponse(
            run=_load_evaluation_run_or_api_error(_repository(app), run_id)
        )

    @app.get(
        "/api/results/{run_id}",
        response_model=EvaluationResultResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def get_evaluation_result(run_id: str) -> EvaluationResultResponse:
        """완료된 실행의 summary artifact를 재계산 없이 검증해 반환한다."""

        repository = _repository(app)
        run = _load_completed_evaluation_run_or_api_error(repository, run_id)
        summary = _load_evaluation_summary_or_api_error(repository, run)
        return EvaluationResultResponse(run=run, summary=summary)

    @app.get(
        "/api/results/{run_id}/timeline",
        response_model=TimelineResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def get_evaluation_timeline(
        run_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> TimelineResponse:
        """완료된 replay의 촬영 schedule만 시간순으로 페이지 처리해 반환한다."""

        repository = _repository(app)
        run = _load_completed_evaluation_run_or_api_error(repository, run_id)
        summary = _load_evaluation_summary_or_api_error(repository, run)
        replay = _load_episode_replay_or_api_error(repository, run, summary)
        items = [
            TimelineCaptureResponse(**capture.model_dump())
            for capture in sorted(
                replay.schedule,
                key=lambda capture: (capture.capture_time_sec, capture.step_index),
            )
        ]
        return TimelineResponse(
            run=run,
            items=_page(items, offset, limit),
            offset=offset,
            limit=limit,
            total=len(items),
        )

    @app.get(
        "/api/results/{run_id}/episodes",
        response_model=EpisodeListResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def list_evaluation_episodes(run_id: str) -> EpisodeListResponse:
        """완료된 평가 run의 저장 replay를 episode 요약으로 반환한다."""

        repository = _repository(app)
        run = _load_completed_evaluation_run_or_api_error(repository, run_id)
        summary = _load_evaluation_summary_or_api_error(repository, run)
        replay = _load_episode_replay_or_api_error(repository, run, summary)
        return EpisodeListResponse(run=run, items=[_episode_summary(replay)])

    @app.get(
        "/api/results/{run_id}/episodes/{episode_id}/steps",
        response_model=EpisodeStepListResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def list_evaluation_episode_steps(
        run_id: str,
        episode_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> EpisodeStepListResponse:
        """선택 당시 후보와 보상을 포함한 저장 replay step을 페이지 처리해 반환한다."""

        repository = _repository(app)
        run = _load_completed_evaluation_run_or_api_error(repository, run_id)
        summary = _load_evaluation_summary_or_api_error(repository, run)
        replay = _load_episode_replay_or_api_error(repository, run, summary)
        if episode_id != EVALUATION_EPISODE_ID:
            raise ApiError(404, "episode_not_found", f"Episode not found: {episode_id}")
        items = sorted(replay.steps, key=lambda step: step.step_index)
        return EpisodeStepListResponse(
            run=run,
            episode=_episode_summary(replay),
            items=_page(items, offset, limit),
            offset=offset,
            limit=limit,
            total=len(items),
        )

    @app.get(
        "/api/results/{run_id}/episodes/{episode_id}/steps/{step_index}",
        response_model=ReplayStep,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def get_evaluation_episode_step(
        run_id: str,
        episode_id: str,
        step_index: int,
    ) -> ReplayStep:
        """선택 capture의 원본 replay step을 index로 직접 조회한다."""

        repository = _repository(app)
        run = _load_completed_evaluation_run_or_api_error(repository, run_id)
        summary = _load_evaluation_summary_or_api_error(repository, run)
        replay = _load_episode_replay_or_api_error(repository, run, summary)
        if episode_id != EVALUATION_EPISODE_ID:
            raise ApiError(404, "episode_not_found", f"Episode not found: {episode_id}")
        step = next((item for item in replay.steps if item.step_index == step_index), None)
        if step is None:
            raise ApiError(404, "episode_step_not_found", f"Step not found: {step_index}")
        return step

    @app.get(
        "/api/scenarios/{scenario_id}/orders",
        response_model=OrderListResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def list_orders(
        scenario_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        priority: Priority | None = None,
    ) -> OrderListResponse:
        """주문별 strip·opportunity 개수와 함께 안정적으로 정렬된 주문 목록을 반환한다."""

        scenario = _load_scenario_or_api_error(_repository(app), scenario_id)
        strip_counts = _counts_by_key(scenario.strips, "order_id")
        opportunity_counts = _counts_by_key(scenario.opportunities, "order_id")
        filtered_orders = [
            order for order in scenario.orders if priority is None or order.priority == priority
        ]
        items = [
            OrderListItem(
                **order.model_dump(),
                strip_count=strip_counts.get(order.order_id, 0),
                opportunity_count=opportunity_counts.get(order.order_id, 0),
            )
            for order in sorted(filtered_orders, key=lambda item: item.order_id)
        ]
        return OrderListResponse(
            items=_page(items, offset, limit), offset=offset, limit=limit, total=len(items)
        )

    @app.get(
        "/api/scenarios/{scenario_id}/strips",
        response_model=StripListResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def list_strips(
        scenario_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        order_id: str | None = None,
    ) -> StripListResponse:
        """주문 필터를 선택적으로 적용한 strip 목록을 순서 번호 기준으로 반환한다."""

        scenario = _load_scenario_or_api_error(_repository(app), scenario_id)
        opportunity_counts = _counts_by_key(scenario.opportunities, "strip_id")
        filtered_strips = [
            strip for strip in scenario.strips if order_id is None or strip.order_id == order_id
        ]
        items = [
            StripListItem(
                **strip.model_dump(),
                opportunity_count=opportunity_counts.get(strip.strip_id, 0),
            )
            for strip in sorted(
                filtered_strips,
                key=lambda item: (item.order_id, item.sequence, item.strip_id),
            )
        ]
        return StripListResponse(
            items=_page(items, offset, limit), offset=offset, limit=limit, total=len(items)
        )

    @app.get(
        "/api/scenarios/{scenario_id}/opportunities",
        response_model=OpportunityListResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def list_opportunities(
        scenario_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        order_id: str | None = None,
        strip_id: str | None = None,
        pass_id: str | None = None,
        kind: OpportunityKind | None = None,
    ) -> OpportunityListResponse:
        """주문·strip·pass·종류 필터를 모두 적용한 촬영 기회 페이지를 반환한다."""

        scenario = _load_scenario_or_api_error(_repository(app), scenario_id)
        filtered_opportunities = [
            opportunity
            for opportunity in scenario.opportunities
            if (order_id is None or opportunity.order_id == order_id)
            and (strip_id is None or opportunity.strip_id == strip_id)
            and (pass_id is None or opportunity.pass_id == pass_id)
            and (kind is None or opportunity.kind == kind)
        ]
        items = [
            OpportunityListItem(
                **opportunity.model_dump(),
                off_nadir_deg=opportunity.off_nadir_deg,
            )
            for opportunity in sorted(
                filtered_opportunities,
                key=lambda item: (item.capture_time_sec, item.opportunity_id),
            )
        ]
        return OpportunityListResponse(
            items=_page(items, offset, limit),
            offset=offset,
            limit=limit,
            total=len(items),
        )

    return app


def _repository(app: FastAPI) -> StorageRepository:
    return cast(StorageRepository, app.state.repository)


def _training_supervisor(app: FastAPI) -> TrainingWorkerSupervisor:
    """app factory가 주입한 supervisor를 training endpoint에서 꺼낸다."""

    return cast(TrainingWorkerSupervisor, app.state.training_supervisor)


def _scenario_summary_response(summary: ScenarioSummary) -> ScenarioSummaryResponse:
    return ScenarioSummaryResponse(
        scenario_id=summary.scenario_id,
        name=summary.name,
        seed=summary.seed,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
    )


def _scenario_validation_response(
    result: ScenarioValidationResult,
) -> ScenarioValidationResponse:
    return ScenarioValidationResponse(
        scenario_id=result.scenario_id,
        valid=result.valid,
        issues=[
            ScenarioValidationIssueResponse(
                code=issue.code,
                location=list(issue.location),
                message=issue.message,
            )
            for issue in result.issues
        ],
    )


def _baseline_policy_or_api_error(policy_name: str) -> BaselinePolicy:
    """허용된 결정론적 기준 정책만 동기 evaluation API에서 실행하게 한다."""

    policy_class = BASELINE_POLICIES.get(policy_name)
    if policy_class is None:
        raise ApiError(
            422,
            "unsupported_baseline_policy",
            f"Unsupported baseline policy: {policy_name}",
        )
    return policy_class()


def _evaluation_summary_response(
    result: EvaluationResult,
    replay_path: Path,
) -> EvaluationSummaryResponse:
    return EvaluationSummaryResponse(
        policy_name=result.policy_name,
        scenario_id=result.scenario_id,
        seed=result.seed,
        steps=result.steps,
        captures=result.captures,
        total_return=result.total_return,
        priority_score=result.priority_score,
        angle_bonus=result.angle_bonus,
        missed_penalty=result.missed_penalty,
        completed_strips=result.completed_strips,
        completed_orders=result.completed_orders,
        average_off_nadir_deg=result.average_off_nadir_deg,
        replay_path=replay_path.as_posix(),
    )


def _summary_from_replay(
    scenario: Scenario, replay: EpisodeReplay, replay_path: Path
) -> EvaluationSummary:
    """solver replay를 기준 정책과 같은 결과 summary 계약으로 변환한다."""

    opportunity_by_id = {item.opportunity_id: item for item in scenario.opportunities}
    captures = len(replay.schedule)
    return EvaluationSummary(
        policy_name=replay.policy_name,
        scenario_id=replay.scenario_id,
        seed=replay.seed,
        steps=len(replay.steps),
        captures=captures,
        total_return=replay.total_return,
        priority_score=sum(item.reward_breakdown.strip_base for item in replay.steps),
        angle_bonus=sum(item.reward_breakdown.angle_bonus for item in replay.steps),
        missed_penalty=sum(item.reward_breakdown.missed_penalty for item in replay.steps),
        completed_strips=replay.completed_strips,
        completed_orders=replay.completed_orders,
        average_off_nadir_deg=(
            sum(opportunity_by_id[item.opportunity_id].off_nadir_deg for item in replay.schedule)
            / captures
            if captures
            else 0.0
        ),
        replay_path=replay_path.as_posix(),
    )


def _load_scenario_or_api_error(repository: StorageRepository, scenario_id: str) -> Scenario:
    """모든 scenario 하위 조회가 같은 오류 형식을 유지하도록 저장소 오류를 변환한다."""

    try:
        return repository.load_scenario(scenario_id)
    except KeyError as error:
        raise ApiError(404, "scenario_not_found", f"Unknown scenario_id: {scenario_id}") from error
    except ArtifactNotFoundError as error:
        raise ApiError(
            409,
            "scenario_artifact_missing",
            f"Scenario artifact is missing for scenario_id: {scenario_id}",
        ) from error
    except ValidationError as error:
        raise ApiError(
            409,
            "scenario_artifact_invalid",
            f"Scenario artifact is invalid for scenario_id: {scenario_id}",
        ) from error


def _load_valid_scenario_or_api_error(repository: StorageRepository, scenario_id: str) -> Scenario:
    """학습·평가 run을 시작하기 전 scenario artifact의 해시·구조까지 재검증한다.

    조회 전용 endpoint는 `_load_scenario_or_api_error`로 충분하지만, 새 run을
    시작하는 endpoint는 저장된 파일이 색인된 SHA-256과 어긋난(조용히 손상된)
    scenario로 학습이 시작되지 않도록 `validate_scenario()`까지 함께 확인한다.
    """

    try:
        result = repository.validate_scenario(scenario_id)
    except KeyError as error:
        raise ApiError(404, "scenario_not_found", f"Unknown scenario_id: {scenario_id}") from error
    except ArtifactNotFoundError as error:
        raise ApiError(
            409,
            "scenario_artifact_missing",
            f"Scenario artifact is missing for scenario_id: {scenario_id}",
        ) from error
    if not result.valid:
        raise ApiError(
            409,
            "scenario_artifact_invalid",
            f"Scenario artifact failed validation for scenario_id: {scenario_id}",
        )
    return repository.load_scenario(scenario_id)


def _load_evaluation_run_or_api_error(repository: StorageRepository, run_id: str) -> EvaluationRun:
    """EvaluationRun 부재를 모든 결과 API의 같은 404 응답으로 변환한다."""

    try:
        return repository.load_evaluation_run(run_id)
    except KeyError as error:
        raise ApiError(
            404, "evaluation_run_not_found", f"Unknown evaluation run_id: {run_id}"
        ) from error


def _load_training_run_or_api_error(repository: StorageRepository, run_id: str) -> TrainingRun:
    """학습 run 부재를 stop API의 구조화된 404 응답으로 변환한다."""

    try:
        return repository.load_training_run(run_id)
    except KeyError as error:
        raise ApiError(
            404, "training_run_not_found", f"Unknown training run_id: {run_id}"
        ) from error


def _load_training_metrics_or_api_error(
    repository: StorageRepository, run: TrainingRun
) -> list[TrainingMetricResponse]:
    """학습 중 append되는 metrics JSONL에서 완성된 행만 안전하게 복원한다."""

    path = repository.data_root / "runs" / run.run_id / "metrics" / "training-metrics.jsonl"
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise ApiError(
            409,
            "training_metrics_invalid",
            f"Training metrics artifact is invalid for run_id: {run.run_id}",
        ) from error

    metrics: list[TrainingMetricResponse] = []
    active_statuses = {RunStatus.RUNNING, RunStatus.STOP_REQUESTED}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        is_unterminated_last_line = index == len(lines) - 1 and not line.endswith("\n")
        if is_unterminated_last_line and run.status in active_statuses:
            continue
        try:
            metrics.append(TrainingMetricResponse.model_validate_json(line))
        except (ValidationError, json.JSONDecodeError) as error:
            raise ApiError(
                409,
                "training_metrics_invalid",
                f"Training metrics artifact is invalid for run_id: {run.run_id}",
            ) from error
    return metrics


def _load_completed_evaluation_run_or_api_error(
    repository: StorageRepository, run_id: str
) -> EvaluationRun:
    """결과 artifact를 읽을 수 있는 completed 실행인지 먼저 확인한다."""

    run = _load_evaluation_run_or_api_error(repository, run_id)
    if run.status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.STOP_REQUESTED}:
        raise ApiError(
            409,
            "evaluation_result_not_ready",
            f"Evaluation run is not completed yet: {run_id}",
        )
    if run.status is not RunStatus.COMPLETED:
        raise ApiError(
            409,
            "evaluation_run_not_completed",
            f"Evaluation run did not complete successfully: {run_id}",
        )
    if run.result_path is None:
        raise ApiError(
            409,
            "evaluation_result_missing",
            f"Completed evaluation run has no result artifact: {run_id}",
        )
    return run


def _load_evaluation_summary_or_api_error(
    repository: StorageRepository, run: EvaluationRun
) -> EvaluationSummaryResponse:
    """summary artifact의 색인·해시·스키마·실행 metadata 일치를 확인한다."""

    assert run.result_path is not None
    try:
        summary = EvaluationSummaryResponse.model_validate(
            repository.load_checked_json_artifact(
                relative_path=Path(run.result_path),
                artifact_type="evaluation_summary",
                owner_type="evaluation_run",
                owner_id=run.run_id,
            )
        )
    except ArtifactNotFoundError as error:
        raise ApiError(
            409,
            "evaluation_artifact_missing",
            f"Result artifact is missing for run_id: {run.run_id}",
        ) from error
    except (ArtifactIntegrityError, ValidationError, ValueError, json.JSONDecodeError) as error:
        raise ApiError(
            409,
            "evaluation_artifact_invalid",
            f"Result artifact is invalid for run_id: {run.run_id}",
        ) from error
    if (
        summary.scenario_id != run.scenario_id
        or summary.policy_name != run.policy_name
        or summary.seed != run.seed
    ):
        raise ApiError(
            409,
            "evaluation_artifact_invalid",
            f"Result artifact does not match run_id: {run.run_id}",
        )
    return summary


def _load_episode_replay_or_api_error(
    repository: StorageRepository,
    run: EvaluationRun,
    summary: EvaluationSummaryResponse,
) -> EpisodeReplay:
    """타임라인에 사용할 replay가 summary 및 EvaluationRun과 같은 평가인지 검증한다."""

    try:
        replay = EpisodeReplay.model_validate(
            repository.load_checked_json_artifact(
                relative_path=Path(summary.replay_path),
                artifact_type="episode_replay",
                owner_type="evaluation_run",
                owner_id=run.run_id,
            )
        )
    except ArtifactNotFoundError as error:
        raise ApiError(
            409,
            "evaluation_artifact_missing",
            f"Replay artifact is missing for run_id: {run.run_id}",
        ) from error
    except (ArtifactIntegrityError, ValidationError, ValueError, json.JSONDecodeError) as error:
        raise ApiError(
            409,
            "evaluation_artifact_invalid",
            f"Replay artifact is invalid for run_id: {run.run_id}",
        ) from error
    if (
        replay.scenario_id != run.scenario_id
        or replay.policy_name != run.policy_name
        or replay.seed != run.seed
    ):
        raise ApiError(
            409,
            "evaluation_artifact_invalid",
            f"Replay artifact does not match run_id: {run.run_id}",
        )
    return replay


def _episode_summary(replay: EpisodeReplay) -> EpisodeSummaryResponse:
    """단일 평가 replay를 확장 가능한 episode 선택용 요약으로 바꾼다."""

    return EpisodeSummaryResponse(
        episode_id=EVALUATION_EPISODE_ID,
        policy_name=replay.policy_name,
        scenario_id=replay.scenario_id,
        seed=replay.seed,
        steps=len(replay.steps),
        captures=len(replay.schedule),
        total_return=replay.total_return,
        completed_strips=replay.completed_strips,
        completed_orders=replay.completed_orders,
    )


def _counts_by_key(
    items: list[Order] | list[Strip] | list[Opportunity], key: str
) -> dict[str, int]:
    """목록 DTO의 하위 항목 수를 만들기 위한 간단한 group-by helper다."""

    counts: dict[str, int] = {}
    for item in items:
        item_key = cast(str, getattr(item, key))
        counts[item_key] = counts.get(item_key, 0) + 1
    return counts


def _page[PageItem](items: list[PageItem], offset: int, limit: int) -> list[PageItem]:
    """offset이 전체 항목 수를 넘는 경우에도 빈 페이지를 일관되게 반환한다."""

    return items[offset : offset + limit]
