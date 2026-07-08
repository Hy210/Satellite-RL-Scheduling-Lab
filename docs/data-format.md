# 데이터 형식

## 1. 목적과 적용 범위

이 문서는 현재 구현된 RL core의 데이터 계약, 단위 및 검증 경계를 기록한다. Python 모델의 기준 구현은 `rl_core/models.py`다.

첫 구현에서는 Pydantic 모델과 JSON을 사용한다. Backend API가 추가될 때도 같은 의미와 단위를 유지한다.

## 2. 공통 규칙

- ID는 비어 있지 않은 문자열이다.
- 모든 시각은 시나리오 시작을 `0`으로 하는 초 단위 실수다.
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
|   +-- final-evaluation.json
+-- model/
    +-- final-model.zip
```

`training-metrics.jsonl`의 각 줄은 특정 timestep에서의 고정 시나리오 평가 결과를 담는다. `final-evaluation.json`은 최종 모델의 reward breakdown, 완료 strip 및 주문 수, step별 선택 요약을 포함한다.

단계 6 성능 검증 CLI는 여러 학습 run을 하나로 묶어 `data/runs/stage6-benchmark-<timestamp>/summary.json`을 저장한다. 이 파일은 benchmark 설정, Random valid 반복 평가 결과, Maskable PPO 반복 학습 결과, 평균/중앙값 요약과 단계 6 통과 판정을 포함한다. 개별 PPO run은 같은 디렉터리의 `ppo-runs/<run-id>/` 아래에 위 artifact 구조로 저장한다.

## 8. Gymnasium 관측 및 행동 계약

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
