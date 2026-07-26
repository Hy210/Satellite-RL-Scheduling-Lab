# 데이터 형식

## 1. 목적과 적용 범위

이 문서는 현재 구현된 RL core의 데이터 계약, 단위 및 검증 경계를 기록한다. Python 모델의 기준 구현은 `rl_core/models.py`다.

첫 구현에서는 Pydantic 모델과 JSON을 사용한다. Backend API가 추가될 때도 같은 의미와 단위를 유지한다.

## 2. 공통 규칙

- ID는 비어 있지 않은 문자열이다.
- 모든 시각은 시나리오 시작을 `0`으로 하는 초 단위 실수다.
- Scenario 저장 메타데이터의 `created_at`/`updated_at`은 위 시나리오 시각과 다른 범주로, UTC 기준 ISO 8601 문자열이다.
- 시간 구간은 `start < end`를 만족한다.
- 각도 단위는 degree다.
- 각속도 단위는 degree/second다.
- 위도와 경도 단위는 degree다.
- 알 수 없는 JSON 필드는 오류로 처리한다.
- pitch는 초기 프로토타입에서 `0 deg`만 허용한다.
- 최대 시나리오 시간은 기본 `86,400초`다.

## 3. Geometry

초기 geometry는 날짜변경선을 넘지 않는 직사각형 경계 또는 단순 polygon으로 표현한다.

```text
Rectangle
  min_lat: -90..90
  min_lon: -180..180
  max_lat: -90..90
  max_lon: -180..180
```

다음 조건을 만족해야 한다.

```text
min_lat < max_lat
min_lon < max_lon
```

복잡한 polygon과 날짜변경선 교차는 초기 범위에서 제외한다.

`Polygon`은 최소 3개의 꼭짓점을 가진다.

```text
Polygon
  vertices[]:
    lat: -90..90
    lon: -180..180
```

polygon은 면적이 0이면 거부한다. 현재 생성기는 strip과 footprint를 4개 꼭짓점의 회전 polygon으로 만든다.

## 4. 주요 모델

### Scenario

- `scenario_id`
- `name`
- `seed`
- `satellite`
- `environment`
- `reward`
- `passes[]`
- `orders[]`
- `strips[]`
- `opportunities[]`
- `ground_track_points[]`
- `footprint_samples[]`
- `access_windows[]`

### SatelliteConfig

- 초기 roll/tilt/pitch
- 축별 자세 제한
- 결합 off-nadir 제한
- roll/tilt 각속도
- 안정화 시간

### EnvironmentConfig

- 시뮬레이션 기간
- strip 촬영시간
- 최소 촬영 간격
- 최대 strip 수
- 최대 현재 행동 후보 수

### RewardConfig

- 각도 보너스 계수
- 미완료 패널티 계수

### OrbitPass

- pass ID와 순서
- 시작 및 종료 시각

`OrbitPass` 자체는 시간 구간만 가진다. 같은 pass ID에 연결되는 ground track과 footprint 샘플은 `ground_track_points[]`와 `footprint_samples[]`에 별도로 저장한다.

### Order

- order ID와 이름
- `red`, `blue`, `background` 우선순위
- 공통 촬영 요구 시작 및 종료 시각
- 주문 직사각형 geometry
- 주문이 허용하는 roll/tilt 범위

### Strip

- strip ID
- 소유 order ID
- 주문 안에서의 순서
- pass 진행 방향에 맞춘 polygon geometry

촬영시간은 strip별 필드로 중복 저장하지 않고 환경 전체의 `imaging_duration_sec`를 사용한다.

### Opportunity

- opportunity, order, strip 및 pass ID
- `early`, `min_off_nadir`, `late` 종류
- 접근 구간 시작과 종료 시각
- 이산화된 촬영 시각
- 요구 roll/tilt/pitch
- 파생 source access window ID

Opportunity는 실제 또는 가상 footprint가 strip과 교차한 접근 가능 구간에서 생성되어야 한다. 구현된 가상 생성기는 `source_access_window_id`로 opportunity가 어떤 access window에서 파생됐는지 추적한다.

### GroundTrackPoint

실제 궤도 데이터가 연결되기 전까지 seed 기반 가상 생성기가 pass별 ground track과 footprint를 만든다. 이 데이터는 RL 정책 입력의 핵심 특성이 아니라 시나리오 검증과 지도 시각화를 위한 근거 데이터다.

- ground track point ID
- pass ID
- sample index
- 시나리오 시작 기준 시각
- 위성 지상점 위도/경도

### FootprintSample

- footprint sample ID
- pass ID
- 참조 ground track point ID
- sample index
- 시나리오 시작 기준 시각
- footprint 중심 위도/경도
- footprint 또는 swath 근사 polygon geometry

현재 구현은 footprint를 pass 진행 방향에 맞춘 회전 polygon으로 저장한다. 실제 궤도 전파 결과를 연결할 때는 같은 의미를 유지하되 geometry 표현을 확장할 수 있다.

### AccessWindow

- access window ID
- pass ID
- order ID와 strip ID
- 교차 시작/종료 시각
- 최소 off-nadir 시각
- 교차 근거 footprint ID 목록
- 파생 opportunity ID 목록

### TrainingRun 및 EvaluationRun

향후 worker와 Backend가 사용할 최소 실행 메타데이터다. 실행 ID, 시나리오 ID, seed, 상태와 artifact 위치를 가진다.

실행 상태는 다음 값을 사용한다.

```text
queued
running
stop_requested
completed
stopped
failed
```

### MaskablePPOTrainingConfig

Maskable PPO 학습을 재현하기 위한 설정이다.

- `total_timesteps`: 학습에 사용할 전체 transition 수
- `learning_seed`: 학습 환경과 정책 초기화 seed
- `evaluation_seed`: 고정 평가 episode seed
- `n_steps`, `batch_size`, `n_epochs`: PPO rollout과 업데이트 크기
- `learning_rate`, `gamma`: PPO 최적화 파라미터
- `checkpoint_interval`: checkpoint 저장 timestep 간격
- `evaluation_interval`: 고정 시나리오 평가 timestep 간격
- `deterministic_eval`: 평가 시 deterministic action 사용 여부
- `artifact_root`: 학습 산출물 루트 경로, 기본값은 `data/runs`

초기 단일 환경 trainer에서는 `batch_size <= n_steps`를 요구한다. 이 제한은 rollout buffer보다 큰 batch를 요청하는 잘못된 smoke 학습 설정을 조기에 거부하기 위한 구현 경계다.

## 5. 참조 무결성

Scenario 로드 시 다음을 검증한다.

- pass, order, strip 및 opportunity ID가 각각 고유하다.
- 모든 주문은 하나 이상의 strip을 가진다.
- strip이 존재하는 order를 참조한다.
- opportunity가 존재하는 order, strip 및 pass를 참조한다.
- opportunity의 order와 해당 strip의 소유 order가 같다.
- pass, 주문 및 촬영이 시나리오 시간 안에 들어간다.
- opportunity가 전체 5초 촬영시간을 접근 구간 안에 담을 수 있다.
- strip 수가 설정된 최대값을 넘지 않는다.

ground track과 footprint 데이터에 대해 다음 검증도 수행한다.

- ground track과 footprint 샘플이 존재하는 pass를 참조한다.
- 샘플 시각이 참조 pass의 시간 구간 안에 있다.
- access window가 존재하는 strip과 pass를 참조한다.
- opportunity의 접근 구간이 참조 access window 안에 들어간다.
- access window는 footprint와 strip의 교차에서 생성된 근거를 가진다.
- opportunity가 참조하는 access window의 파생 opportunity 목록에 포함되어야 한다.

## 6. 구조 검증과 Action mask의 경계

구조적으로 모순된 데이터는 Scenario 로드 시 거부한다. 반면 특정 state에서 실행할 수 없는 opportunity는 데이터에서 삭제하거나 Scenario 전체를 거부하지 않고 simulator가 action mask로 판정한다.

예를 들면 다음과 같다.

| 조건 | 처리 위치 |
|---|---|
| 존재하지 않는 strip 참조 | 데이터 검증 오류 |
| 촬영시간이 접근 구간에 들어가지 않음 | 데이터 검증 오류 |
| 중복 opportunity ID | 데이터 검증 오류 |
| 현재 자세에서 전환시간 부족 | Action mask |
| 이미 촬영한 strip | Action mask |
| 위성 또는 주문 자세 제한 위반 | Action mask |
| 주문 마감 후 후보 | Action mask 및 만료 처리 |

이 경계를 통해 잘못된 파일은 조기에 거부하면서도, 실행 시점에 달라지는 제약과 마스킹 사유는 분석 로그로 보존한다.

## 7. 저장과 복원

`Scenario.to_json()`과 `Scenario.from_json()`은 문자열 직렬화를 담당한다. `Scenario.save(path)`와 `Scenario.load(path)`는 UTF-8 JSON 파일 저장과 복원을 담당한다.

동일한 Scenario를 저장하고 다시 읽었을 때 의미적으로 같은 모델이어야 한다. seed 기반 생성기는 동일 seed와 크기에서 동일한 JSON을 생성해야 한다.

Maskable PPO 학습 산출물은 `data/runs/<run-id>/` 아래에 저장한다.

```text
data/runs/<run-id>/
|-- config.json
|-- run.json
|-- checkpoints/
|   +-- checkpoint-<timesteps>.zip
|-- metrics/
|   |-- training-metrics.jsonl
|   |-- final-evaluation.json
|   +-- replay.json
+-- model/
    +-- final-model.zip
```

`training-metrics.jsonl`의 각 줄은 특정 timestep에서의 고정 시나리오 평가 결과를 담는다. `final-evaluation.json`은 최종 모델의 reward breakdown, 완료 strip 및 주문 수, step별 선택 요약과 replay를 포함한다. `replay.json`은 같은 replay만 별도 저장한 파일이다.

단계 6 성능 검증 CLI는 여러 학습 run을 하나로 묶어 `data/runs/stage6-benchmark-<timestamp>/summary.json`을 저장한다. 이 파일은 benchmark 설정, Random valid 반복 평가 결과, Maskable PPO 반복 학습 결과, 평균/중앙값 요약과 단계 6 통과 판정을 포함한다. 개별 PPO run은 같은 디렉터리의 `ppo-runs/<run-id>/` 아래에 위 artifact 구조로 저장한다.

### EpisodeReplay

평가 episode의 재생 로그는 `EpisodeReplay` 형식으로 저장한다. 같은 형식은 기준 정책 평가와 Maskable PPO 평가가 함께 사용한다.

- `policy_name`, `scenario_id`, `seed`
- `steps[]`: step별 재생 로그
- `schedule[]`: 최종 촬영 스케줄 요약
- `total_return`, `completed_strips`, `completed_orders`

`steps[]`의 각 항목은 다음 값을 가진다.

- `step_index`
- `state_before`, `state_after`: 선택 전후 시각, roll, tilt, 완료 strip/order 수
- `candidates[]`: 선택 당시 후보 slot, opportunity/order/strip/pass ID, 촬영 시각, 요구 자세, valid 여부와 `mask_reasons[]`
- `action`, `selected_opportunity_id`, `expired_order_ids[]`
- `reward`, `reward_breakdown`, `cumulative_return`

`schedule[]`은 실제 촬영된 action만 모은 목록이며 step index, opportunity/order/strip/pass ID, 촬영 시각, 촬영 자세와 해당 step reward를 가진다.

평가 결과 조회에서 `GET /api/results/{run_id}/episodes`는 현재 `EpisodeReplay` 하나를 `episode_id: "evaluation"`으로 노출한다. `GET /api/results/{run_id}/episodes/evaluation/steps?offset&limit`의 `items[]`는 위 `steps[]` 항목을 그대로 반환하며, 응답에는 선택한 `episode` 요약과 `offset`, `limit`, `total`도 포함한다. 이 endpoint는 저장 artifact의 소유자, 타입, SHA-256 및 `EpisodeReplay` 구조와 run metadata를 검증한 뒤에만 응답한다.

### PolicyComparison

동일 시나리오의 여러 정책 결과는 `PolicyComparison` 형식으로 저장한다.

- `scenario_id`
- `entries[]`: 정책별 비교 행
- `best_policy_name`: 총 return, 완료 주문 수, 완료 strip 수, 촬영 수 순서로 고른 최고 정책 이름

`entries[]`의 각 항목은 정책 이름, seed, 총 return, reward breakdown 합계, 완료 strip/order 수, 촬영 수, step 수와 선택적으로 해당 정책의 `replay_path`를 가진다. Backend에서 완료 `EvaluationRun`을 선택해 만들 때는 `evaluation_run_id`도 저장해 비교 행이 정확한 결과와 replay 화면으로 이동할 수 있게 한다. 이 artifact는 단계 13의 정책 비교 화면과 단계 8 저장 계층에서 재사용한다.

### OptimizationBaselineResult

CP-SAT 같은 최적화 solver 기준해는 `OptimizationBaselineResult` 형식으로 저장한다. 이 artifact는 solver 실행 자체의 메타데이터와 선택된 opportunity 목록을 보존하고, 실제 정책 비교 점수는 선택 결과를 simulator로 다시 평가해 만든 내장 `EpisodeReplay`와 `PolicyComparison`을 기준으로 한다.

- `solver_name`: 현재 구현값은 `ortools_cp_sat`
- `scenario_id`
- `seed`
- `status`: OR-Tools CP-SAT status 이름, 예를 들어 `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `UNKNOWN`
- `objective_value`: solver 모델의 목적함수 값, 해가 없으면 null
- `best_objective_bound`: solver가 제공하는 최선 bound, 해가 없으면 null
- `optimality_gap`: `(best_objective_bound - objective_value) / max(abs(objective_value), 1)` 형태의 상대 gap, 계산할 수 없으면 null
- `time_limit_sec`: solver 실행 제한 시간
- `selected_opportunity_ids[]`: solver가 선택한 opportunity ID 목록
- `replay`: 선택 결과를 simulator 평가기로 실행해 만든 `EpisodeReplay`

`selected_opportunity_ids[]`는 시간순으로 정렬해 저장한다. `INFEASIBLE` 또는 `UNKNOWN`처럼 해가 없는 상태에서는 목록이 비고 objective/bound/gap은 `null`이며, replay는 선택 없이 simulator를 끝까지 진행한 기록이다. `UNKNOWN`은 time limit처럼 해를 찾기 전 종료된 경우를 포함하므로 `status`와 `time_limit_sec`를 함께 확인한다. 저장된 목록을 다시 평가할 때 simulator가 시간, 자세, 마감 및 중복 촬영 제약을 위반한다고 판정하면 해당 solver artifact는 유효한 기준해로 사용하지 않고 검증 오류로 처리한다. `replay.policy_name`은 정책 비교 호환성을 위해 `cp_sat_baseline`을 사용한다.

## 8. 저장 계층 계약

`StorageRepository`는 `data/scheduler.sqlite3`의 메타데이터와 data root 아래 artifact 파일을 함께 관리한다.

- SQLite에는 scenario ID, run ID, 상태, 상대 artifact 경로, SHA-256, byte 크기 및 생성 시각을 저장한다.
- `Scenario`, 학습 설정, `EpisodeReplay`, `PolicyComparison`, `OptimizationBaselineResult` 같은 JSON 원본은 파일로 저장한다.
- 모델처럼 바이너리인 artifact도 파일로 저장하고, DB에는 경로와 무결성 요약만 기록한다.
- 모든 경로는 data root 기준 상대 경로이며 `..` 또는 절대 경로는 거부한다.
- JSON 파일은 같은 디렉터리의 임시 파일을 fsync한 뒤 원자 교체한다. 교체 실패 시 이전 완성 파일을 유지한다.
- DB에는 있지만 파일이 없는 artifact는 `ArtifactNotFoundError` 또는 `list_missing_artifacts()`로 식별한다.
- 실행 상태 전이는 `queued → running → completed/failed`, `running → stop_requested → stopped/failed`만 허용한다. `completed`, `stopped`, `failed`는 terminal 상태다.
- worker supervisor의 `recover_interrupted_runs()`는 worker 부재가 확인된 시작 시점에만 `running`을 `failed`로, `stop_requested`를 `stopped`로 바꾼다. 이 복구는 artifact를 삭제하거나 학습을 자동 재개하지 않는다.

## 9. 초기 Backend API 계약

초기 FastAPI Backend는 `backend.app.create_app()`으로 만들며, 테스트에서는 독립된 `StorageRepository`를 주입할 수 있다.

- `GET /api/health`: `{ "status": "ok", "storage_schema_version" }`
- `GET /api/version`: `{ "api_version", "storage_schema_version" }`
- `GET /api/scenarios`: `items[]`에 scenario ID, 이름, seed, 생성·수정 시각을 담은 목록
- `GET /api/scenarios/{scenario_id}`: 검증된 전체 `Scenario` JSON
- `GET /api/scenarios/{scenario_id}/orders`: priority 필터가 가능한 주문 목록
- `GET /api/scenarios/{scenario_id}/strips`: order ID 필터가 가능한 strip 목록
- `GET /api/scenarios/{scenario_id}/opportunities`: order/strip/pass/kind 필터가 가능한 촬영 기회 목록
- `GET /api/scenarios/{scenario_id}/validation`: artifact 무결성과 구조 검증 결과
- `POST /api/evaluation-runs`: 기준 정책을 실행하고 평가 artifact를 저장
- `POST /api/cp-sat-evaluation-runs`: CP-SAT 평가 run을 queued 상태로 저장하고 worker에 인계
- `POST /api/training-runs`: Maskable PPO 학습 run을 queued 상태로 생성하고 worker에 인계
- `POST /api/training-runs/{run_id}/stop`: 학습 run의 cooperative stop 요청
- `GET /api/training-runs/{run_id}`: 학습 실행 상태 metadata
- `GET /api/training-runs/{run_id}/detail`: 설정 snapshot과 checkpoint/final artifact 존재 여부
- `GET /api/training-runs/{run_id}/metrics`: pagination된 학습 평가 요약 로그
- `GET /api/training-runs`: 최신 학습 run metadata 목록 (`scenario_id`, `status`, pagination 필터)
- `GET /api/evaluation-runs`: 최신 평가 run metadata 목록 (`scenario_id`, `status`, pagination 필터)
- `GET /api/evaluation-runs/{run_id}`: 평가 실행 상태 metadata
- `GET /api/results/{run_id}`: 검증된 평가 요약 결과
- `GET /api/results/{run_id}/timeline`: replay schedule 기반 촬영 타임라인
- `GET /api/results/{run_id}/episodes/{episode_id}/steps/{step_index}`: 선택 capture의 원본 replay step
- `POST /api/policy-comparisons`: 선택한 완료 evaluation run 집합의 비교 artifact 생성
- `GET /api/policy-comparisons/{comparison_id}`: 검증된 비교 artifact 조회

현재 scenario, order, environment 및 reward 설정 API는 읽기 전용이다. `POST`/`PATCH`/`DELETE` 형태의 변경 계약은 아직 정의하지 않았다. 후속 변경 API는 기존 scenario를 덮어쓰지 않는 version 또는 복제본 생성, 변경값 검증, strip/opportunity 등 파생 artifact 재생성 및 training/evaluation run의 snapshot 참조 규칙을 함께 정의해야 한다.

하위 목록 API는 `{ "items", "offset", "limit", "total" }` 형식을 사용한다. 기본 offset은 0, 기본 limit은 100, 허용 limit 범위는 1~500이다. orders에는 `strip_count`, `opportunity_count`, strips에는 `opportunity_count`, opportunities에는 계산된 `off_nadir_deg`를 추가한다.

training/evaluation run 목록도 같은 pagination 형식을 사용하며 `TrainingRun` 또는 `EvaluationRun` metadata만 반환한다. summary, replay, metrics artifact는 목록에 포함하지 않고 각 run의 기존 상세 endpoint에서 검증해 조회한다.

API 오류는 다음 공통 형식을 사용한다.

```json
{
  "error": {
    "code": "scenario_not_found",
    "message": "Unknown scenario_id: example"
  }
}
```

없는 시나리오는 404 `scenario_not_found`, 색인된 Scenario artifact가 없거나 검증에 실패한 경우는 409 `scenario_artifact_missing` 또는 `scenario_artifact_invalid`으로 반환한다.

validation API는 존재하고 읽을 수 있는 artifact의 hash 불일치 또는 구조 오류를 200 응답으로 반환한다.

```json
{
  "scenario_id": "synthetic-tiny-20260707",
  "valid": false,
  "issues": [
    {
      "code": "artifact_checksum_mismatch",
      "location": [],
      "message": "Stored scenario artifact does not match its indexed checksum."
    }
  ]
}
```

각 issue는 `code`, JSON 경로를 나타내는 `location[]`, `message`를 가진다. action mask와 simulator 실행 시점 제약은 구조 오류로 넣지 않는다.

기준 정책 평가 요청은 다음 형식을 사용한다.

```json
{
  "scenario_id": "synthetic-tiny-20260707",
  "policy_name": "priority_greedy",
  "seed": 17
}
```

현재 허용되는 동기 기준 정책 `policy_name`은 `random_valid`, `earliest_deadline_first`, `priority_greedy`, `priority_efficiency_greedy`다. 성공 응답은 `run`과 `summary`를 가지며, summary에는 성능 지표와 `replay_path`가 포함된다. replay와 summary는 각각 `data/evaluations/<run-id>/replay.json`, `summary.json`으로 저장되고 SQLite에는 `EvaluationRun.result_path`와 artifact 색인이 남는다. PPO 최종 평가도 같은 형식의 완료 `EvaluationRun`으로 저장하며 `source_training_run_id`로 원본 학습 run을 연결한다.

CP-SAT 요청은 `{ "scenario_id", "seed", "time_limit_sec" }`를 받고 `202`와 queued `EvaluationRun`을 반환한다. CP-SAT은 PPO 학습과 같은 단일 로컬 실행 슬롯을 공유하는 별도 worker에서 실행한다. `OPTIMAL` 또는 해를 찾은 `FEASIBLE` 결과만 completed evaluation으로 summary·replay·solver artifact를 저장한다. `INFEASIBLE`·해가 없는 `UNKNOWN` 및 worker 예외는 failed 상태와 오류 메시지를 남기며 정책 비교 대상에 포함하지 않는다.

결과 조회 API는 `GET /api/evaluation-runs/{run_id}`, `GET /api/results/{run_id}`, `GET /api/results/{run_id}/timeline`, `GET /api/results/{run_id}/episodes`, `GET /api/results/{run_id}/episodes/{episode_id}/steps`, `GET /api/results/{run_id}/episodes/{episode_id}/steps/{step_index}`을 사용한다. run 상태 조회는 `{ "run": EvaluationRun }`을 반환한다. 결과 조회는 `{ "run", "summary" }`을 반환하고, 타임라인은 `{ "run", "items", "offset", "limit", "total" }` 형식에서 `EpisodeReplay.schedule`의 `ReplayCapture` 항목만 반환한다. episode 목록은 현재 단일 replay의 `evaluation` 요약을 반환하며, step 목록은 `{ "run", "episode", "items", "offset", "limit", "total" }` 형식에서 원본 `ReplayStep`을 반환한다. 단일 step endpoint는 같은 검증 경계 안에서 해당 `step_index`의 `ReplayStep`을 직접 반환하며, 없으면 `404 episode_step_not_found`이다. summary와 replay는 artifact 색인의 owner/type/SHA-256 및 Pydantic 계약을 통과하고 scenario ID, policy name, seed가 해당 `EvaluationRun`과 일치해야 한다. 알 수 없는 episode ID는 `404 episode_not_found`이다.

학습 시작 요청은 `{ "scenario_id", "config" }` 형식이며 `config`는 `MaskablePPOTrainingConfig`의 학습 파라미터를 사용한다. `artifact_root`가 전달돼도 Backend는 이를 무시하고 data root 아래 `runs`로 교체한다. 성공 응답은 `202`와 `{ "run": TrainingRun }`이고 초기 상태는 `queued`다. `runs/<run_id>/config.json`은 `training_config` artifact로 색인한 뒤 worker가 다시 검증해 읽는다. 실행 중인 단일 worker가 있으면 `409 training_worker_busy`, process 시작 실패는 `500 training_worker_start_failed`로 응답하며 두 경우 모두 생성한 run은 `failed` 상태로 남긴다.

중지 요청은 `POST /api/training-runs/{run_id}/stop`이며 `{ "run": TrainingRun }`을 `202`로 반환한다. `queued → stopped`, `running → stop_requested → stopped` 전이를 사용하고, 이미 `stop_requested`인 요청은 같은 상태를 반환한다. callback이 중지 신호를 감지하면 현재 timestep의 checkpoint를 저장한다. `stopped` run에는 final model, final evaluation, replay artifact가 없을 수 있으며 이는 artifact 누락 오류가 아니라 의도된 중지 결과다.

학습 상태 조회는 `{ "run": TrainingRun }`을 반환한다. `GET /api/training-runs/{run_id}/detail`은 같은 `run`, 검증된 `MaskablePPOTrainingConfig` snapshot, checkpoint 파일명 목록, `final_model_available`, `final_evaluation_available`을 반환한다. checkpoint와 final artifact는 현재 다운로드 API가 아닌 상태 표시용 존재 여부이며, config artifact가 없거나 구조가 잘못되면 `409 training_config_invalid`이다. metrics 조회는 `{ "run", "items", "offset", "limit", "total" }` 형식이며 각 item은 `timesteps`와 평가 요약(`policy_name`, scenario/seed, return 및 reward breakdown, 완료 strip/order 수, captures, steps)을 가진다. metrics는 `runs/<run_id>/metrics/training-metrics.jsonl`의 append-only JSONL이며 metrics 행에는 대용량 replay와 decision 목록을 넣지 않는다. 파일 부재는 빈 목록이고, 실행 중 마지막 줄의 미완성 JSON은 일시적으로 제외한다. 그 밖의 UTF-8·JSON·필수 필드 오류는 `409 training_metrics_invalid`이다.

## 10. Gymnasium 관측 및 행동 계약

`SatelliteSchedulingEnv`는 모든 값을 `numpy.float32` 고정 배열로 반환한다.

| 관측 key | shape | 의미 |
|---|---:|---|
| `global` | `(7,)` | 시간, 현재 자세, 이전 촬영과 전체 완료율 |
| `strips` | `(2000, 8)` | strip 및 소유 주문의 현재·미래 요약 |
| `strip_presence` | `(2000,)` | 실제 strip 행은 1, padding은 0 |
| `candidates` | `(128, 10)` | 현재 action 후보의 정규화된 특징 |
| `candidate_presence` | `(128,)` | 실제 후보 행은 1, padding은 0 |

행동 공간은 `Discrete(129)`다.

```text
0     = skip
1~128 = 현재 candidate slot
```

`action_masks()`는 같은 길이의 `numpy.bool_` 배열을 반환한다. `0=skip`은 항상 true다. 후보가 없는 slot과 실행 불가능한 후보는 false다.

모든 시간 특징은 episode 길이를 기준으로, roll/tilt는 축별 제한을 기준으로, 우선순위는 red 점수 5를 기준으로 정규화한다. 결합각은 off-nadir 제한을 기준으로 정규화한다. Presence와 완료 여부는 `0` 또는 `1`을 사용한다.

`info`에는 최소한 다음 실행 정보가 포함된다.

- 시나리오 ID와 누적 return
- 완료 strip 및 주문 수
- 요청 action과 실제 실행 action
- 마스킹된 외부 action 여부
- 선택 opportunity ID
- 만료 order ID
- 기본 보상, 각도 보너스 및 미완료 패널티 분해
