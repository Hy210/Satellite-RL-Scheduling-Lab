"""로컬 SQLite와 JSON artifact를 함께 관리하는 저장 계층이다."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ValidationError

from rl_core.models import (
    EpisodeReplay,
    EvaluationRun,
    PolicyComparisonRun,
    RunStatus,
    Scenario,
    TrainingRun,
)

SCHEMA_VERSION = 1
ArtifactOwnerType = Literal["scenario", "training_run", "evaluation_run"]
RunKind = Literal["training", "evaluation"]

_ALLOWED_STATUS_TRANSITIONS = {
    "queued": {"queued", "running", "stopped", "failed"},
    "running": {"running", "stop_requested", "completed", "failed"},
    "stop_requested": {"stop_requested", "stopped", "failed"},
    "completed": {"completed"},
    "stopped": {"stopped"},
    "failed": {"failed"},
}


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """SQLite에 색인된 JSON artifact의 위치와 무결성 요약이다."""

    artifact_type: str
    owner_type: ArtifactOwnerType
    owner_id: str
    scenario_id: str
    relative_path: Path
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    """시나리오 목록 화면이 전체 JSON 없이 표시할 최소 메타데이터다."""

    scenario_id: str
    name: str
    seed: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ScenarioValidationIssue:
    """저장된 Scenario artifact에서 발견한 무결성 또는 구조 검증 문제다."""

    code: str
    location: tuple[str | int, ...]
    message: str


@dataclass(frozen=True, slots=True)
class ScenarioValidationResult:
    """저장된 Scenario 파일의 무결성과 구조 검증 결과다."""

    scenario_id: str
    valid: bool
    issues: tuple[ScenarioValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class RecoveredRun:
    """worker 부재로 terminal 상태로 정리한 실행의 결과다."""

    run_kind: RunKind
    run_id: str
    previous_status: str
    recovered_status: str


class ArtifactNotFoundError(FileNotFoundError):
    """DB에는 존재하지만 artifact 파일을 찾을 수 없을 때 발생한다."""


class ArtifactIntegrityError(ValueError):
    """artifact 색인과 파일 내용 또는 소유 정보가 일치하지 않을 때 발생한다."""


class StorageRepository:
    """SQLite 메타데이터와 data root 아래 JSON artifact를 일관되게 저장한다."""

    def __init__(self, data_root: Path = Path("data")) -> None:
        self.data_root = data_root.resolve()
        self.database_path = self.data_root / "scheduler.sqlite3"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        """필요한 SQLite 테이블을 만들고 현재 schema version을 기록한다."""

        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scenarios (
                    scenario_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    scenario_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL REFERENCES scenarios(scenario_id),
                    algorithm TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    total_timesteps INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    artifact_directory TEXT,
                    error_message TEXT,
                    config_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL REFERENCES scenarios(scenario_id),
                    policy_name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    source_training_run_id TEXT,
                    result_path TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    relative_path TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    owner_type TEXT NOT NULL CHECK (
                        owner_type IN ('scenario', 'training_run', 'evaluation_run')
                    ),
                    owner_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL REFERENCES scenarios(scenario_id),
                    sha256 TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS policy_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL REFERENCES scenarios(scenario_id),
                    seed INTEGER NOT NULL,
                    evaluation_run_ids TEXT NOT NULL,
                    artifact_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT INTO schema_metadata(singleton, schema_version)
                VALUES (1, ?)
                ON CONFLICT(singleton) DO UPDATE SET schema_version = excluded.schema_version
                """,
                (SCHEMA_VERSION,),
            )

    def save_scenario(self, scenario: Scenario) -> Path:
        """Scenario JSON과 메타데이터를 함께 저장하고 상대 artifact 경로를 반환한다."""

        relative_path = Path("scenarios") / scenario.scenario_id / "scenario.json"
        payload = scenario.model_dump(mode="json")
        encoded = _json_bytes(payload)
        self._atomic_write(relative_path, encoded)
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO scenarios(
                    scenario_id, name, seed, content_sha256, scenario_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scenario_id) DO UPDATE SET
                    name = excluded.name,
                    seed = excluded.seed,
                    content_sha256 = excluded.content_sha256,
                    scenario_path = excluded.scenario_path,
                    updated_at = excluded.updated_at
                """,
                (
                    scenario.scenario_id,
                    scenario.name,
                    scenario.seed,
                    _sha256(encoded),
                    relative_path.as_posix(),
                    now,
                    now,
                ),
            )
        self._upsert_artifact(
            artifact_type="scenario",
            owner_type="scenario",
            owner_id=scenario.scenario_id,
            scenario_id=scenario.scenario_id,
            relative_path=relative_path,
            content=encoded,
        )
        return relative_path

    def load_scenario(self, scenario_id: str) -> Scenario:
        """저장된 Scenario JSON을 읽어 Pydantic 계약으로 다시 검증한다."""

        row = self._fetch_one(
            "SELECT scenario_path FROM scenarios WHERE scenario_id = ?", (scenario_id,)
        )
        if row is None:
            raise KeyError(f"unknown scenario_id: {scenario_id}")
        return Scenario.model_validate_json(self._read_artifact(Path(row["scenario_path"])))

    def validate_scenario(self, scenario_id: str) -> ScenarioValidationResult:
        """artifact 해시와 Scenario 구조를 재검증해 상세 문제 목록을 반환한다.

        파일이 없거나 scenario ID가 없으면 검증 대상 자체를 읽을 수 없으므로 예외를
        유지한다. 읽을 수 있는 JSON의 구조 오류는 UI가 표시할 수 있게 결과로 반환한다.
        """

        row = self._fetch_one(
            "SELECT scenario_path, content_sha256 FROM scenarios WHERE scenario_id = ?",
            (scenario_id,),
        )
        if row is None:
            raise KeyError(f"unknown scenario_id: {scenario_id}")
        relative_path = Path(row["scenario_path"])
        path = self.data_root / self._normalize_relative_path(relative_path)
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact file is missing: {relative_path.as_posix()}")

        content = path.read_bytes()
        issues: list[ScenarioValidationIssue] = []
        if _sha256(content) != row["content_sha256"]:
            issues.append(
                ScenarioValidationIssue(
                    code="artifact_checksum_mismatch",
                    location=(),
                    message="Stored scenario artifact does not match its indexed checksum.",
                )
            )
        try:
            Scenario.model_validate_json(content)
        except ValidationError as error:
            issues.extend(_validation_issue(item) for item in error.errors(include_url=False))

        return ScenarioValidationResult(
            scenario_id=scenario_id,
            valid=not issues,
            issues=tuple(issues),
        )

    @property
    def schema_version(self) -> int:
        """현재 SQLite schema version을 health/version API가 조회할 수 있게 한다."""

        row = self._fetch_one("SELECT schema_version FROM schema_metadata WHERE singleton = 1", ())
        if row is None:
            raise RuntimeError("storage schema metadata is missing")
        return cast(int, row["schema_version"])

    def list_scenario_summaries(self) -> list[ScenarioSummary]:
        """대용량 Scenario JSON을 읽지 않고 목록 화면용 메타데이터를 반환한다."""

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT scenario_id, name, seed, created_at, updated_at
                FROM scenarios
                ORDER BY updated_at DESC, scenario_id ASC
                """
            ).fetchall()
        return [
            ScenarioSummary(
                scenario_id=row["scenario_id"],
                name=row["name"],
                seed=row["seed"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def save_training_run(self, run: TrainingRun) -> None:
        """학습 실행 상태를 upsert해 worker 재시작 후에도 조회할 수 있게 한다."""

        self._require_scenario(run.scenario_id)
        self._validate_status_transition("training_runs", run.run_id, run.status.value)
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO training_runs(
                    run_id, scenario_id, algorithm, seed, total_timesteps, status,
                    artifact_directory, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    scenario_id = excluded.scenario_id,
                    algorithm = excluded.algorithm,
                    seed = excluded.seed,
                    total_timesteps = excluded.total_timesteps,
                    status = excluded.status,
                    artifact_directory = excluded.artifact_directory,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    run.run_id,
                    run.scenario_id,
                    run.algorithm,
                    run.seed,
                    run.total_timesteps,
                    run.status.value,
                    run.artifact_directory,
                    run.error_message,
                    now,
                    now,
                ),
            )

    def load_training_run(self, run_id: str) -> TrainingRun:
        """SQLite에 저장한 학습 실행 상태를 현재 Pydantic 계약으로 복원한다."""

        row = self._fetch_one(
            """
            SELECT run_id, scenario_id, algorithm, seed, total_timesteps, status,
                   artifact_directory, error_message
            FROM training_runs WHERE run_id = ?
            """,
            (run_id,),
        )
        if row is None:
            raise KeyError(f"unknown training run_id: {run_id}")
        return TrainingRun.model_validate(dict(row))

    def list_training_runs(
        self,
        *,
        scenario_id: str | None = None,
        status: RunStatus | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[TrainingRun], int]:
        """대시보드가 artifact를 읽지 않고 최근 학습 run을 페이지로 조회하게 한다."""

        where, parameters = _run_list_filters(scenario_id, status)
        return self._list_runs(
            table_name="training_runs",
            columns=(
                "run_id, scenario_id, algorithm, seed, total_timesteps, status, "
                "artifact_directory, error_message"
            ),
            model=TrainingRun,
            where=where,
            parameters=parameters,
            offset=offset,
            limit=limit,
        )

    def save_evaluation_run(self, run: EvaluationRun) -> None:
        """기준 정책·PPO·CP-SAT 평가 실행의 메타데이터를 upsert한다."""

        self._require_scenario(run.scenario_id)
        self._validate_status_transition("evaluation_runs", run.run_id, run.status.value)
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs(
                    run_id, scenario_id, policy_name, seed, status, source_training_run_id,
                    result_path, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    scenario_id = excluded.scenario_id,
                    policy_name = excluded.policy_name,
                    seed = excluded.seed,
                    status = excluded.status,
                    source_training_run_id = excluded.source_training_run_id,
                    result_path = excluded.result_path,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    run.run_id,
                    run.scenario_id,
                    run.policy_name,
                    run.seed,
                    run.status.value,
                    run.source_training_run_id,
                    run.result_path,
                    run.error_message,
                    now,
                    now,
                ),
            )

    def save_completed_evaluation(
        self, run: EvaluationRun, summary: BaseModel, replay: EpisodeReplay
    ) -> EvaluationRun:
        """동일한 summary·replay 계약으로 모든 정책 평가를 완료 상태로 저장한다."""

        if run.status is not RunStatus.RUNNING:
            raise ValueError("completed evaluation must start from running status")
        run_root = Path("evaluations") / run.run_id
        replay_path = run_root / "replay.json"
        summary_path = run_root / "summary.json"
        self.save_json_artifact(
            artifact_type="episode_replay",
            owner_type="evaluation_run",
            owner_id=run.run_id,
            scenario_id=run.scenario_id,
            relative_path=replay_path,
            payload=replay.model_dump(mode="json"),
        )
        self.save_json_artifact(
            artifact_type="evaluation_summary",
            owner_type="evaluation_run",
            owner_id=run.run_id,
            scenario_id=run.scenario_id,
            relative_path=summary_path,
            payload=summary.model_dump(mode="json"),
        )
        completed = run.model_copy(
            update={"status": RunStatus.COMPLETED, "result_path": summary_path.as_posix()}
        )
        self.save_evaluation_run(completed)
        return completed

    def load_evaluation_run(self, run_id: str) -> EvaluationRun:
        """SQLite에 저장한 평가 실행 상태를 현재 Pydantic 계약으로 복원한다."""

        row = self._fetch_one(
            """
            SELECT run_id, scenario_id, policy_name, seed, status, source_training_run_id,
                   result_path, error_message
            FROM evaluation_runs WHERE run_id = ?
            """,
            (run_id,),
        )
        if row is None:
            raise KeyError(f"unknown evaluation run_id: {run_id}")
        return EvaluationRun.model_validate(dict(row))

    def list_evaluation_runs(
        self,
        *,
        scenario_id: str | None = None,
        status: RunStatus | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[EvaluationRun], int]:
        """대시보드와 결과 목록이 최근 평가 run metadata를 조회하게 한다."""

        where, parameters = _run_list_filters(scenario_id, status)
        return self._list_runs(
            table_name="evaluation_runs",
            columns=(
                "run_id, scenario_id, policy_name, seed, status, source_training_run_id, "
                "result_path, error_message"
            ),
            model=EvaluationRun,
            where=where,
            parameters=parameters,
            offset=offset,
            limit=limit,
        )

    def _list_runs[RunModel: TrainingRun | EvaluationRun](
        self,
        *,
        table_name: Literal["training_runs", "evaluation_runs"],
        columns: str,
        model: type[RunModel],
        where: str,
        parameters: list[str],
        offset: int,
        limit: int,
    ) -> tuple[list[RunModel], int]:
        """고정된 run 테이블만 대상으로 최신순 목록과 전체 수를 함께 읽는다."""

        with self._connection() as connection:
            total = cast(
                int,
                connection.execute(
                    f"SELECT COUNT(*) FROM {table_name}{where}", parameters
                ).fetchone()[0],
            )
            rows = connection.execute(
                f"SELECT {columns} FROM {table_name}{where} "
                "ORDER BY updated_at DESC, run_id ASC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchall()
        return [cast(RunModel, model.model_validate(dict(row))) for row in rows], total

    def save_training_config(self, run: TrainingRun, config: BaseModel) -> Path:
        """학습 설정 snapshot을 run artifact로 원자 저장하고 색인한다."""

        self.save_training_run(run)
        return self.save_json_artifact(
            artifact_type="training_config",
            owner_type="training_run",
            owner_id=run.run_id,
            scenario_id=run.scenario_id,
            relative_path=Path("runs") / run.run_id / "config.json",
            payload=config.model_dump(mode="json"),
        )

    def save_policy_comparison(self, comparison: PolicyComparisonRun, payload: BaseModel) -> None:
        """비교 입력 run 집합과 JSON artifact를 scenario 소유물로 원자 저장한다."""

        self._require_scenario(comparison.scenario_id)
        relative_path = self.save_json_artifact(
            artifact_type="policy_comparison",
            owner_type="scenario",
            owner_id=comparison.scenario_id,
            scenario_id=comparison.scenario_id,
            relative_path=Path(comparison.artifact_path),
            payload=payload.model_dump(mode="json"),
        )
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO policy_comparisons("
                "comparison_id, scenario_id, seed, evaluation_run_ids, artifact_path, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(comparison_id) DO UPDATE SET artifact_path = excluded.artifact_path",
                (
                    comparison.comparison_id,
                    comparison.scenario_id,
                    comparison.seed,
                    json.dumps(comparison.evaluation_run_ids),
                    relative_path.as_posix(),
                    _utc_now(),
                ),
            )

    def load_policy_comparison_run(self, comparison_id: str) -> PolicyComparisonRun:
        """비교 metadata를 복원한다."""

        row = self._fetch_one(
            "SELECT comparison_id, scenario_id, seed, evaluation_run_ids, artifact_path "
            "FROM policy_comparisons WHERE comparison_id = ?",
            (comparison_id,),
        )
        if row is None:
            raise KeyError(comparison_id)
        return PolicyComparisonRun(
            **{**dict(row), "evaluation_run_ids": json.loads(row["evaluation_run_ids"])}
        )

    def save_json_artifact(
        self,
        *,
        artifact_type: str,
        owner_type: ArtifactOwnerType,
        owner_id: str,
        scenario_id: str,
        relative_path: Path,
        payload: Any,
    ) -> Path:
        """검증된 JSON payload를 원자 저장하고 SQLite artifact 색인에 반영한다."""

        self._require_owner(owner_type, owner_id, scenario_id)
        encoded = _json_bytes(payload)
        normalized_path = self._normalize_relative_path(relative_path)
        self._atomic_write(normalized_path, encoded)
        self._upsert_artifact(
            artifact_type=artifact_type,
            owner_type=owner_type,
            owner_id=owner_id,
            scenario_id=scenario_id,
            relative_path=normalized_path,
            content=encoded,
        )
        return normalized_path

    def load_json_artifact(self, relative_path: Path) -> Any:
        """artifact 파일을 읽고 유효한 JSON 값으로 복원한다."""

        return json.loads(self._read_artifact(self._normalize_relative_path(relative_path)))

    def load_checked_json_artifact(
        self,
        *,
        relative_path: Path,
        artifact_type: str,
        owner_type: ArtifactOwnerType,
        owner_id: str,
    ) -> Any:
        """색인된 소유자·종류·SHA-256이 모두 일치하는 JSON artifact만 읽는다.

        결과 조회는 파일 경로만 신뢰하지 않는다. 실행 metadata가 다른 실행의 파일을
        가리키거나 저장 뒤 파일이 바뀐 경우에도 화면에 잘못된 결과를 보이지 않게 한다.
        """

        normalized_path = self._normalize_relative_path(relative_path)
        row = self._fetch_one(
            """
            SELECT artifact_type, owner_type, owner_id, sha256, byte_size
            FROM artifacts WHERE relative_path = ?
            """,
            (normalized_path.as_posix(),),
        )
        if row is None:
            raise ArtifactIntegrityError(f"artifact is not indexed: {normalized_path.as_posix()}")
        if (
            row["artifact_type"] != artifact_type
            or row["owner_type"] != owner_type
            or row["owner_id"] != owner_id
        ):
            raise ArtifactIntegrityError(
                f"artifact ownership does not match: {normalized_path.as_posix()}"
            )

        path = self.data_root / normalized_path
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact file is missing: {normalized_path.as_posix()}")
        content = path.read_bytes()
        if len(content) != row["byte_size"] or _sha256(content) != row["sha256"]:
            raise ArtifactIntegrityError(
                f"artifact checksum does not match: {normalized_path.as_posix()}"
            )
        return json.loads(content)

    def register_existing_artifact(
        self,
        *,
        artifact_type: str,
        owner_type: ArtifactOwnerType,
        owner_id: str,
        scenario_id: str,
        relative_path: Path,
    ) -> Path:
        """학습 모델처럼 별도 과정이 만든 파일을 무결성 정보와 함께 색인한다."""

        self._require_owner(owner_type, owner_id, scenario_id)
        normalized_path = self._normalize_relative_path(relative_path)
        path = self.data_root / normalized_path
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact file is missing: {normalized_path.as_posix()}")
        self._upsert_artifact(
            artifact_type=artifact_type,
            owner_type=owner_type,
            owner_id=owner_id,
            scenario_id=scenario_id,
            relative_path=normalized_path,
            content=path.read_bytes(),
        )
        return normalized_path

    def list_missing_artifacts(self) -> list[StoredArtifact]:
        """DB 색인과 실제 파일이 어긋난 artifact를 복구 작업용으로 반환한다."""

        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM artifacts ORDER BY relative_path").fetchall()
        return [
            _stored_artifact(row)
            for row in rows
            if not (self.data_root / Path(row["relative_path"])).is_file()
        ]

    def recover_interrupted_runs(self) -> tuple[RecoveredRun, ...]:
        """중단된 단일 worker의 running 상태를 안전한 terminal 상태로 정리한다.

        Backend 재시작만으로 호출하면 살아 있는 worker를 잘못 실패 처리할 수 있다.
        따라서 worker supervisor가 기존 worker가 없음을 확인한 시작 시점에만 호출한다.
        """

        recovered: list[RecoveredRun] = []
        recovery_rules = (
            ("training", "training_runs"),
            ("evaluation", "evaluation_runs"),
        )
        with self._connection() as connection:
            for run_kind, table_name in recovery_rules:
                rows = connection.execute(
                    f"SELECT run_id, status, error_message FROM {table_name} "
                    "WHERE status IN ('running', 'stop_requested') ORDER BY run_id"
                ).fetchall()
                for row in rows:
                    previous_status = row["status"]
                    recovered_status = "failed" if previous_status == "running" else "stopped"
                    message = _recovery_message(previous_status, row["error_message"])
                    connection.execute(
                        f"UPDATE {table_name} SET status = ?, error_message = ?, updated_at = ? "
                        "WHERE run_id = ?",
                        (recovered_status, message, _utc_now(), row["run_id"]),
                    )
                    recovered.append(
                        RecoveredRun(
                            run_kind=cast(RunKind, run_kind),
                            run_id=row["run_id"],
                            previous_status=previous_status,
                            recovered_status=recovered_status,
                        )
                    )
        return tuple(recovered)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _fetch_one(self, query: str, parameters: tuple[str, ...]) -> sqlite3.Row | None:
        with self._connection() as connection:
            return cast(sqlite3.Row | None, connection.execute(query, parameters).fetchone())

    def _require_scenario(self, scenario_id: str) -> None:
        if self._fetch_one("SELECT 1 FROM scenarios WHERE scenario_id = ?", (scenario_id,)) is None:
            raise KeyError(f"unknown scenario_id: {scenario_id}")

    def _validate_status_transition(
        self,
        table_name: Literal["training_runs", "evaluation_runs"],
        run_id: str,
        new_status: str,
    ) -> None:
        row = self._fetch_one(f"SELECT status FROM {table_name} WHERE run_id = ?", (run_id,))
        if row is None:
            return
        current_status = row["status"]
        if new_status not in _ALLOWED_STATUS_TRANSITIONS[current_status]:
            raise ValueError(f"invalid run status transition: {current_status} -> {new_status}")

    def _require_owner(
        self,
        owner_type: ArtifactOwnerType,
        owner_id: str,
        scenario_id: str,
    ) -> None:
        if owner_type == "scenario":
            if owner_id != scenario_id:
                raise ValueError("scenario artifact owner_id must match scenario_id")
            self._require_scenario(scenario_id)
            return
        table_name = "training_runs" if owner_type == "training_run" else "evaluation_runs"
        row = self._fetch_one(f"SELECT scenario_id FROM {table_name} WHERE run_id = ?", (owner_id,))
        if row is None:
            raise KeyError(f"unknown {owner_type} owner_id: {owner_id}")
        if row["scenario_id"] != scenario_id:
            raise ValueError("artifact scenario_id must match its owner")

    def _upsert_artifact(
        self,
        *,
        artifact_type: str,
        owner_type: ArtifactOwnerType,
        owner_id: str,
        scenario_id: str,
        relative_path: Path,
        content: bytes,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(
                    relative_path, artifact_type, owner_type, owner_id, scenario_id,
                    sha256, byte_size, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    artifact_type = excluded.artifact_type,
                    owner_type = excluded.owner_type,
                    owner_id = excluded.owner_id,
                    scenario_id = excluded.scenario_id,
                    sha256 = excluded.sha256,
                    byte_size = excluded.byte_size,
                    created_at = excluded.created_at
                """,
                (
                    relative_path.as_posix(),
                    artifact_type,
                    owner_type,
                    owner_id,
                    scenario_id,
                    _sha256(content),
                    len(content),
                    _utc_now(),
                ),
            )

    def _normalize_relative_path(self, relative_path: Path) -> Path:
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("artifact path must stay inside data_root")
        return relative_path

    def _atomic_write(self, relative_path: Path, content: bytes) -> None:
        target = self.data_root / self._normalize_relative_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _read_artifact(self, relative_path: Path) -> str:
        path = self.data_root / self._normalize_relative_path(relative_path)
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact file is missing: {relative_path.as_posix()}")
        return path.read_text(encoding="utf-8")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run_list_filters(scenario_id: str | None, status: RunStatus | None) -> tuple[str, list[str]]:
    """두 run 테이블이 공통으로 사용하는 선택 조건을 안전한 고정 SQL로 만든다."""

    clauses: list[str] = []
    parameters: list[str] = []
    if scenario_id is not None:
        clauses.append("scenario_id = ?")
        parameters.append(scenario_id)
    if status is not None:
        clauses.append("status = ?")
        parameters.append(status.value)
    return (f" WHERE {' AND '.join(clauses)}" if clauses else ""), parameters


def _recovery_message(previous_status: str, previous_message: str | None) -> str:
    recovery_note = (
        "worker process was unavailable during recovery"
        if previous_status == "running"
        else "stop request was recovered after worker process ended"
    )
    return f"{previous_message}\n{recovery_note}" if previous_message else recovery_note


def _stored_artifact(row: sqlite3.Row) -> StoredArtifact:
    return StoredArtifact(
        artifact_type=row["artifact_type"],
        owner_type=row["owner_type"],
        owner_id=row["owner_id"],
        scenario_id=row["scenario_id"],
        relative_path=Path(row["relative_path"]),
        sha256=row["sha256"],
        byte_size=row["byte_size"],
    )


def _validation_issue(error: Mapping[str, Any]) -> ScenarioValidationIssue:
    return ScenarioValidationIssue(
        code=str(error["type"]),
        location=tuple(cast(str | int, item) for item in error["loc"]),
        message=str(error["msg"]),
    )
