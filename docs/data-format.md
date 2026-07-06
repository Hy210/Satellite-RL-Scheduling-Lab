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

초기 geometry는 날짜변경선을 넘지 않는 직사각형 경계로 표현한다.

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
- 직사각형 geometry

촬영시간은 strip별 필드로 중복 저장하지 않고 환경 전체의 `imaging_duration_sec`를 사용한다.

### Opportunity

- opportunity, order, strip 및 pass ID
- `early`, `min_off_nadir`, `late` 종류
- 접근 구간 시작과 종료 시각
- 이산화된 촬영 시각
- 요구 roll/tilt/pitch

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
