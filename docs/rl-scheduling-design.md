# 단일 위성 촬영 스케줄링 RL 프로토타입 설계

## 1. 문서 목적

이 문서는 강화학습을 이용해 단일 위성의 촬영 일정을 결정하는 학습용 프로토타입의 확정 설계를 기록한다.

결과 확인, 시나리오 편집 및 학습 제어를 위한 웹 애플리케이션 설계는 [web-application-design.md](web-application-design.md)를 참고한다.
단계별 구현 순서와 각 단계의 완료 조건은 [implementation-plan.md](implementation-plan.md)를 따른다.
구현된 입력 데이터의 필드와 단위는 [data-format.md](data-format.md)를 따른다.

프로젝트의 우선 목표는 실제 위성 운영 시스템을 완전하게 재현하는 것이 아니라 다음을 달성하는 것이다.

- 강화학습의 상태, 행동, 보상, 에피소드 개념을 학습한다.
- 제한된 촬영 기회 중 가치 있는 주문과 strip을 선택하는 정책을 학습한다.
- 물리 및 운영 제약은 단순하지만 명시적인 인터페이스로 제공한다.
- 향후 다중 위성, 구름, 전력, 정밀 궤도 전파 등의 기능을 확장할 수 있게 한다.

초기 버전은 위성 한 대만 지원한다. 향후 위성 두 대까지 확장할 계획이지만 현재 설계 범위에는 포함하지 않는다.

## 2. 문제 정의

한 대의 위성이 고정된 1일 동안 30개의 연속된 orbit/pass를 비행한다. 전 세계에 위치한 주문 영역이 에피소드 시작 전에 모두 주어지며, 에이전트는 가능한 촬영 기회 중 일부를 선택해 누적 return을 최대화한다.

모든 주문을 반드시 완료할 필요는 없다. 높은 우선순위 주문을 우선 처리하면서 가능한 한 많은 주문과 strip을 촬영하는 것이 목적이다.

## 3. 시나리오와 에피소드

### 3.1 시나리오 규모

- 시뮬레이션 기간: 1일(86,400초)
- 위성 수: 1대
- orbit/pass 수: 30개
- 주문 수: 100개
- 주문당 평균 strip 수: 10개 이내
- 최대 strip padding 크기: 2,000개
- 현재 이벤트의 최대 행동 후보 수: 128개

### 3.2 고정 시나리오

첫 프로토타입은 하나의 고정 시나리오에서 좋은 스케줄을 학습한다. 매 에피소드에서 주문, 궤도와 촬영 기회는 완전히 동일하다.

초기 고정 시나리오는 실제 위경도를 가지는 가상 주문 100개를 전 세계에 분포하도록 생성한다. 시나리오 생성기는 seed를 입력받으며, 동일한 seed에서는 항상 동일한 주문, strip, 가상 ground track, footprint 및 촬영 기회 데이터를 생성해야 한다.

### 3.3 에피소드와 step

에피소드 하나는 1일 전체 촬영 스케줄이다.

```text
s1 -> a1 -> r1 -> s2
s2 -> a2 -> r2 -> s3
...
sn -> an -> rn -> 종료
```

`(state, action)` 전이 하나는 step이고, 모든 step을 포함한 하루 전체가 episode다. 에피소드 return은 해당 하루 동안 받은 모든 보상의 합이다.

다음 조건 중 하나를 만족하면 에피소드를 종료한다.

- 24시간에 도달한다.
- 모든 strip을 촬영한다.
- 유효한 촬영 기회가 더 이상 남아 있지 않다.

## 4. 데이터 모델

### 4.1 주문(Order)

주문은 실제 위경도를 가진 하나의 지리 영역이며 다음 정보를 포함한다.

- 주문 ID
- 우선순위: `red`, `blue`, `background`
- 촬영 요구 시작 시각과 종료 시각
- 주문 영역 geometry
- 주문에 속한 strip 목록
- 허용 roll/tilt 범위

주문의 촬영 요구 기간은 주문에 속한 모든 strip에 동일하게 적용한다. 에피소드 진행 중 신규 주문은 추가되지 않는다.

### 4.2 Strip

주문 영역은 위성의 1회 촬영 단위인 strip polygon으로 분할한다.

- 주문마다 strip 수가 다를 수 있다.
- 프로젝트 전체의 모든 strip은 크기와 촬영 소요시간이 동일하다.
- strip 촬영 소요시간은 5초다.
- seed 기반 가상 생성기의 strip은 선택된 pass의 진행 방향에 맞춘 회전 polygon으로 생성한다.
- 같은 주문에 속한 모든 strip의 가치는 동일하다.
- 구름이나 strip별 품질 차이는 첫 버전에서 사용하지 않는다.
- 모든 strip을 촬영해야 주문이 완전히 완료된다.
- 일부 strip만 촬영한 경우 촬영 면적 비율에 따른 부분 성공으로 인정한다.

```text
completion_ratio = 촬영 완료 strip 수 / 전체 strip 수
```

한 strip을 성공적으로 촬영하면 해당 strip의 다른 모든 촬영 기회는 마스킹한다. 주문의 모든 strip을 촬영하면 주문 전체를 완료 처리한다.

첫 프로토타입의 RL core는 주문 polygon을 strip으로 분할하지 않는다. 이미 분할된 strip 데이터를 시나리오 입력으로 받는다. 주문 geometry의 자동 strip 분할은 별도의 전처리 또는 향후 확장 기능으로 둔다.

### 4.3 촬영 기회(Opportunity)

궤도 전파와 가시성 계산은 RL 환경 밖에서 수행하며, 환경에는 사전 계산된 촬영 기회를 제공한다.

최종 방향은 외부 궤도 계산기 또는 실제 운영 데이터가 촬영 기회를 제공하는 것이다. 하지만 현재는 실제 ground track과 footprint 데이터가 없으므로 seed 기반 가상 생성기가 다음 중간 산출물을 함께 만든다.

1. pass별 시간 샘플
2. 각 시간 샘플의 가상 ground track 좌표
3. 센서 footprint 또는 swath 근사 영역
4. footprint와 strip geometry의 교차로부터 얻은 접근 가능 구간
5. 접근 가능 구간을 이산화한 촬영 기회

이 중간 산출물은 실제 궤도 물리를 정밀하게 재현하기 위한 것이 아니라, `이 strip이 왜 이 pass에서 유효한 후보인지`를 지도에서 검수할 수 있게 하는 가상 근거 데이터다. 실제 궤도 데이터가 연결되면 같은 인터페이스에 실제 ground track, footprint 및 접근 가능 구간을 공급하고 seed 기반 생성기는 테스트와 예제 데이터 용도로 축소한다.

RL 환경은 촬영 기회의 시각, pass, 접근 구간 및 요구 자세를 직접 계산하지 않고 입력 데이터를 검증하고 소비한다. 다만 가상 시나리오 생성기는 RL 환경 밖의 전처리 계층으로서 ground track과 footprint를 만든 뒤 opportunity를 생성해야 한다.

촬영 기회는 최소한 다음 정보를 포함한다.

- opportunity ID
- order ID
- strip ID
- orbit/pass ID
- 촬영 시각
- 요구 roll
- 요구 tilt
- 요구 pitch

같은 strip은 여러 pass에서 촬영할 수 있고, 동일 pass에서도 여러 촬영 후보를 가질 수 있다.

하나의 연속 접근 가능 구간은 최대 세 후보로 이산화한다.

1. 초반 후보: 접근 구간 시작 시각
2. 최소각 후보: 접근 구간 안에서 `sqrt(roll^2 + tilt^2)`가 최소인 시각
3. 후반 후보: 접근 구간 종료 시각에서 촬영시간 5초를 뺀 시각

같은 시각으로 생성되는 중복 후보는 하나로 합친다.

seed 기반 가상 생성기에서는 경쟁 가능한 촬영 후보가 생기도록 접근 구간 시작 시각을 10초 grid로 양자화한다. 연속 난수 시각만 사용하면 대부분의 이벤트에 유효 후보가 하나뿐이어서 정책별 선택 차이가 사라질 수 있기 때문이다. 이 grid는 실제 위성 운용 규칙이 아니라 학습용 가상 시나리오의 임시 가정이며, 실제 궤도 데이터 연결 시 재검토한다.

이전 생성기는 pass 시간 구간과 무작위 접근 구간으로 opportunity를 만들었기 때문에, strip과 궤도 footprint의 공간적 관계를 시각적으로 증명하지 못했다. 현재 가상 생성기는 pass별 ground track과 회전 footprint polygon을 만들고, pass 진행 방향에 맞춘 strip polygon과의 교차를 access window로 병합한 뒤 opportunity를 파생해 공간적 근거를 추적한다.

## 5. 위성 자세와 촬영 제약

### 5.1 초기 자세와 허용 범위

초기 자세는 다음과 같다.

```text
roll  = 0 deg
tilt  = 0 deg
pitch = 0 deg
```

첫 버전에서 pitch는 인터페이스에 유지하지만 `0 deg`로 고정한다.

자세 제한은 다음과 같다.

```text
-30 deg <= roll <= 30 deg
-30 deg <= tilt <= 30 deg
sqrt(roll^2 + tilt^2) <= 30 deg
```

촬영 기회는 위성의 물리 범위와 주문의 허용 각도 범위를 모두 만족해야 한다.

### 5.2 자세 전환

roll과 tilt 축은 동시에 회전할 수 없다고 가정한다. 각도 차이에 따라 자세 전환시간을 계산한다.

```text
delta_roll = abs(target_roll - current_roll)
delta_tilt = abs(target_tilt - current_tilt)

slew_time =
    delta_roll / roll_rate
  + delta_tilt / tilt_rate
  + settling_time
```

기본 파라미터는 다음과 같다.

```text
roll_rate     = 5 deg/s
tilt_rate     = 5 deg/s
settling_time = 1 s
```

현재 자세와 요구 자세가 완전히 같으면 `slew_time = 0`으로 처리하며 안정화 시간도 부여하지 않는다. 촬영 후에는 해당 촬영 자세를 유지한다.

별도의 자세 전환 action은 만들지 않는다. 이전 촬영과 후보 촬영 사이의 시간에 필요한 자세 전환을 수행할 수 있다고 가정한다.

### 5.3 연속 촬영

촬영 종료 후 다음 촬영까지의 최소 간격은 환경 파라미터이며 초기값은 5초다.

```text
earliest_next_start =
    previous_imaging_end + max(minimum_interval, slew_time)
```

후보 촬영 시각이 `earliest_next_start`보다 빠르면 해당 행동을 마스킹한다.

## 6. 이벤트 기반 환경

모든 촬영 후보를 시각순으로 정렬한다. 환경은 매초 진행하지 않고 의미 있는 촬영 이벤트에서만 에이전트에게 결정을 요청한다.

각 step에서 에이전트는 현재 시각에 실행 가능한 후보 중 하나를 선택하거나 `skip`한다.

- 촬영을 선택하면 5초 동안 촬영하고 시간과 자세를 갱신한다.
- 촬영과 자세 전환 때문에 이미 지나간 후보는 만료 처리한다.
- `skip`하면 보상 없이 현재 시각의 모든 후보를 포기하고 다음 이벤트로 이동한다.
- 동시에 여러 strip이 촬영 가능해도 한 step에서는 하나만 선택한다.

촬영 시작 시각은 기회 데이터에 이산화된 시각으로 지정한다. 시간과 자세 제약을 만족하지 못하는 후보는 선택할 수 없다.

## 7. 행동 공간

행동은 현재 이벤트에 배치된 촬영 기회 하나를 선택하는 것이다.

```text
0     = skip
1~128 = 현재 이벤트의 촬영 후보
```

행동의 실제 의미는 다음과 같다.

```text
action = (strip_id, opportunity_id)
```

전체 촬영 기회를 하나의 거대한 고정 행동 공간으로 만들지 않는다. 현재 이벤트의 후보만 최대 128개까지 padding하고 action mask를 적용한다.

현재 후보가 128개를 초과하면 다음 순서로 정렬한다.

1. 우선순위가 높은 후보
2. 주문 마감이 가까운 후보
3. off-nadir가 낮은 후보

가능하면 시나리오 생성 단계에서 동시 후보가 128개를 넘지 않게 한다.

## 8. Action masking

다음 조건 중 하나라도 해당하면 촬영 행동을 마스킹한다.

- 주문 요구 기간 밖이다.
- 촬영 기회의 유효 시각이 아니다.
- strip이 이미 촬영되었다.
- 주문이 이미 완료되었다.
- 요구 자세가 위성의 물리 범위를 벗어난다.
- 요구 자세가 주문의 허용 각도 범위를 벗어난다.
- 이전 촬영 후 최소 간격을 만족하지 못한다.
- 요구 자세로 전환할 시간이 부족하다.
- 다른 촬영과 시간이 겹친다.
- 촬영 종료가 접근 구간 또는 에피소드 종료 시각을 넘는다.
- 주문의 마감 시각이 지났다.

하드 제약 위반은 음의 보상으로 학습시키지 않고 action mask로 차단한다. 마스킹 사유는 디버깅 정보로 기록한다.

Gym wrapper의 `action_masks()`는 `skip + 128개 후보`에 대응하는 길이 129의 boolean 배열을 반환한다. Maskable PPO는 이 배열을 이용해 마스킹된 action의 선택 확률을 0으로 만든다.

Action mask를 사용하지 않는 일반 Gym 검사나 외부 호출이 마스킹된 action을 전달하면 실제 촬영을 실행하지 않고 `skip`으로 안전하게 변환한다. 해당 사실과 요청·실행 action은 `info`에 기록하며 별도 패널티는 주지 않는다.

## 9. 관측 상태

에이전트는 현재 상태와 미래 촬영 기회를 고려할 수 있다. 다만 모든 원시 기회를 행동으로 펼치지 않고 strip별 요약 정보로 제공한다.

관측에는 다음 정보가 포함된다.

### 9.1 위성 및 시간 정보

- 현재 시각
- 현재 roll/tilt
- 마지막 촬영 종료 시각
- 최소 촬영 간격

### 9.2 주문 및 strip 정보

- 주문 우선순위
- 주문 마감까지 남은 시간
- 주문 완료율
- strip 완료 여부
- 다음 촬영 기회 시각
- 그다음 촬영 기회 시각
- 남은 촬영 기회 수
- 가장 좋은 미래 off-nadir
- 현재 후보의 요구 roll/tilt
- 현재 후보의 실행 가능 여부

최대 2,000개 strip까지 padding하며 padding 영역은 mask로 구분한다.

구현된 Gymnasium 관측은 다음 고정 크기 Dict로 구성한다.

```text
global             : (7,)
strips             : (2000, 8)
strip_presence     : (2000,)
candidates         : (128, 10)
candidate_presence : (128,)
```

- `global`: 현재 시각, 자세, 이전 촬영 여부와 경과시간, strip·주문 완료율
- `strips`: 우선순위, 마감, 주문 완료율, strip 완료, 다음 기회, 남은 기회와 최저 미래각
- `strip_presence`: 실제 strip 행과 padding 행 구분
- `candidates`: 현재 후보의 우선순위, 자세, 각도, 기동시간, 종류와 유효성
- `candidate_presence`: 실제 후보 행과 padding 행 구분

Action mask는 관측 Dict와 별도로 제공한다. Presence 배열은 데이터가 없는 padding 행을 구분하고, action mask는 실제 후보 중 현재 state에서 실행할 수 있는 행동을 구분한다.

### 9.3 정규화

신경망 입력은 다음 기준으로 정규화한다.

```text
시간          = 하루 86,400초 기준 0~1
roll/tilt     = 30 deg 기준 -1~1
우선순위      = red 1.0, blue 0.6, background 0.2
완료율        = 0~1
남은 기회 수 = 설정된 최대값 기준 0~1
```

실제 위경도 geometry는 데이터와 시각화를 위해 보존한다. 첫 정책에는 궤도 계산 결과인 시간과 자세 정보를 직접 제공하며 위경도를 주요 학습 특성으로 사용하지 않는다.

## 10. 보상 함수

### 10.1 우선순위

주문 전체의 기본 우선순위 점수는 다음과 같다.

```text
red        = 5
blue       = 3
background = 1
```

우선순위 점수를 `P`, 주문의 전체 strip 수를 `N`이라고 하면 strip 하나의 기본 보상은 다음과 같다.

```text
strip_base_reward = P / N
```

따라서 주문 크기와 관계없이 모든 strip을 촬영했을 때 기본 보상 합계는 항상 해당 주문의 우선순위 점수 `P`가 된다.

별도의 주문 완료 보너스는 사용하지 않는다. `skip` 보상은 0이다.

### 10.2 각도 선호 보상

off-nadir의 단순 근사값은 다음과 같다.

```text
off_nadir = sqrt(roll^2 + tilt^2)
```

```text
angle_quality =
    1 - clamp(off_nadir / max_off_nadir, 0, 1)

angle_bonus =
    (P / N) * angle_bonus_weight * angle_quality
```

초기값은 다음과 같다.

```text
max_off_nadir     = 30 deg
angle_bonus_weight = 0.1
```

최종 촬영 보상은 다음과 같다.

```text
capture_reward = strip_base_reward + angle_bonus
```

각도는 우선순위를 뒤집는 강한 목표가 아니라 같은 조건에서 낮은 각도를 선호하게 하는 보조 조건이다.

### 10.3 미완료 패널티

각 주문의 마감 시각에 미완료 비율에 따른 패널티를 한 번 적용한다.

```text
miss_penalty =
    -penalty_weight * P * (1 - completion_ratio)
```

초기값은 다음과 같다.

```text
penalty_weight = 0.5
```

패널티를 적용한 뒤 해당 주문의 남은 strip과 촬영 기회는 만료 처리한다.

## 11. 평가 기준

학습 보상과 별도로 다음 지표를 기록한다.

- 에피소드 총 return
- 획득한 기본 우선순위 점수
- 완전히 완료한 주문 수
- 우선순위별 주문 완료율
- 전체 strip 촬영률
- 주문별 부분 완료율
- 평균 off-nadir
- 총 자세 전환시간
- 전체 촬영 횟수
- 미완료 패널티 합계
- 유효 촬영 기회 사용률

별도의 완료 보너스는 두지 않는다. 완료 주문 수는 평가 지표로 사용한다. 정책이 주문 완료를 회피하고 부분 촬영만 지나치게 분산하는 현상이 관찰될 때만 완료 보너스 도입을 재검토한다.

## 12. 비교 정책

RL 정책은 다음 기준 정책과 비교한다.

1. Random valid: 마스킹되지 않은 행동 중 무작위 선택
2. Earliest deadline first: 마감이 가까운 주문 우선
3. Priority greedy: `red`, `blue`, `background` 순서 우선
4. Priority-efficiency greedy: 우선순위 대비 촬영, 대기 및 자세 전환 비용이 낮은 후보 우선

```text
efficiency =
    priority_score / (imaging_time + waiting_time + slew_time)
```

축소된 소규모 시나리오에서는 전수조사, 정수계획 또는 CP-SAT 등으로 최적해를 구하고 RL의 optimality gap을 측정한다.

```text
optimality_gap = (optimal_score - rl_score) / optimal_score
```

큰 시나리오에서는 주요 greedy 정책보다 높은 점수를 얻는 것을 목표로 하고, 작은 시나리오에서는 계산된 최적해에 최대한 근접하는 것을 목표로 한다. RL 정책이 항상 최적해를 보장할 필요는 없다.

## 13. 초기 알고리즘 방향

초기 알고리즘은 이산 행동과 action masking을 지원하는 Maskable PPO를 우선 검토한다.

알고리즘과 하이퍼파라미터는 구현 단계에서 확정한다. 환경을 먼저 구성한 뒤 기준 정책, 환경 검증, RL 학습 순으로 진행한다.

구현된 초기 trainer는 `MaskablePPOTrainingConfig`로 학습 seed와 평가 seed를 분리한다. 학습 중에는 일정 timestep마다 같은 고정 시나리오를 평가해 `training-metrics.jsonl`에 기록하고, checkpoint와 최종 모델을 `data/runs/<run-id>/` 아래에 저장한다.

단계 6 검증은 tiny 시나리오에서 저장한 모델을 다시 불러와 동일한 방식으로 평가할 수 있는지 확인하고, 별도의 반복 seed benchmark로 Random valid보다 일관되게 우수한지 비교한다. 2026-07-08 `synthetic-tiny-20260707` 엄격 검증에서는 Maskable PPO가 Random valid median return을 근소하게 넘었고 skip 반복이나 특정 action slot 고착 기준도 통과했다. 다만 tiny 문제에서는 완료 strip/order 수가 같았으므로 small/full 시나리오에서 성능 의미를 다시 검토한다.

## 14. 초기 범위에서 제외하는 기능

- 다중 위성 공동 스케줄링
- 실제 날짜별 궤도 변화
- RL 환경 내부의 정밀 궤도 전파
- RL core 내부의 주문 polygon 자동 strip 분할
- 배터리와 전력 소비
- 저장장치 및 지상국 다운로드
- 구름과 기상
- strip별 품질 및 가치 차이
- pitch 기동
- 진행 중 신규 주문 유입
- 연속적인 촬영 시각 및 자세 제어 action

실제 궤도 전파는 제외하지만, 실제 데이터가 없는 동안에는 시각 검수 가능한 가상 ground track과 footprint 생성기를 제공한다. 이 생성기는 정밀 궤도 모델이 아니라 opportunity 생성 근거를 보존하기 위한 전처리 도구다.

위 기능은 초기 프로토타입의 학습과 평가가 완료된 후 확장한다.

## 15. 확정 파라미터 요약

| 항목 | 값 |
|---|---:|
| 위성 수 | 1 |
| 에피소드 길이 | 1일 |
| orbit/pass 수 | 30 |
| 주문 수 | 100 |
| 주문당 평균 strip 수 | 10개 이내 |
| 최대 strip padding | 2,000 |
| 현재 행동 후보 padding | 128 |
| strip 촬영시간 | 5초 |
| 최소 촬영 간격 | 5초 |
| roll 범위 | -30~30 deg |
| tilt 범위 | -30~30 deg |
| 최대 결합 off-nadir | 30 deg |
| pitch | 0 deg 고정 |
| roll rate | 5 deg/s |
| tilt rate | 5 deg/s |
| 자세 안정화 시간 | 1초 |
| 접근 구간당 후보 | 최대 3개 |
| 우선순위 점수 | 5 / 3 / 1 |
| 각도 보너스 계수 | 0.1 |
| 미완료 패널티 계수 | 0.5 |
