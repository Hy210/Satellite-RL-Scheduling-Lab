# 프로젝트 지식 베이스

## 1. 문서 목적

이 문서는 강화학습 기반 위성 촬영 스케줄링 프로젝트를 수행하면서 축적한 도메인 지식, 개념, 설계 근거, 관찰 및 미해결 질문을 장기적으로 보존한다.

문서별 책임은 다음과 같이 구분한다.

- 확정된 RL 환경 동작: [RL 스케줄링 설계](rl-scheduling-design.md)
- 웹 시스템 구조와 화면: [웹 애플리케이션 설계](web-application-design.md)
- 구현 순서와 완료 조건: [단계별 구현 계획](implementation-plan.md)
- 사용자의 개인 RL 공부: [개인 RL 학습 노트](rl-study-notes.md)
- 프로젝트 전반에서 축적한 지식과 근거: 이 문서

자동 갱신 규칙은 프로젝트 루트의 [AGENTS.md](../AGENTS.md)를 따른다.

## 2. 지식 상태

| 상태 | 의미 |
|---|---|
| 확정 | 프로젝트에서 검증했거나 사용자와 합의한 지식 |
| 가정 | 프로토타입 단순화를 위해 임시로 채택한 내용 |
| 관찰 | 구현 또는 실험에서 확인했지만 추가 일반화가 필요한 내용 |
| 검토 필요 | 근거 확인이나 추가 결정이 필요한 내용 |
| 폐기 | 이전에 사용했지만 현재는 유효하지 않은 내용 |

## 3. 핵심 도메인 모델

### 3.1 주문 영역과 strip

**상태:** 확정
**마지막 갱신:** 2026-07-06

하나의 주문은 실제 위경도를 가진 지리 영역이며 위성의 1회 촬영 단위인 여러 직사각형 strip으로 구성된다. 주문마다 strip 수가 다를 수 있지만 첫 프로토타입에서 모든 strip의 크기, 촬영시간 및 주문 내부 가치는 동일하다.

주문을 완전히 완료하려면 모든 strip을 촬영해야 한다. 일부만 촬영하면 촬영한 strip 비율을 주문 면적의 부분 완료율로 간주한다.

첫 프로토타입의 RL core는 polygon을 strip으로 분할하지 않는다. 분할된 strip은 외부 전처리 또는 가상 시나리오 생성기의 결과로 제공된다.

관련 설계: [RL 스케줄링 설계 - 데이터 모델](rl-scheduling-design.md#4-데이터-모델)

### 3.2 Orbit, pass와 촬영 기회

**상태:** 확정과 가정 혼합  
**마지막 갱신:** 2026-07-07

이 프로젝트에서 여러 orbit/pass는 위성 한 대가 하루 동안 연속해서 지나는 궤도 구간을 의미한다. 초기 시나리오는 30개 pass를 사용한다.

동일 strip은 여러 pass에서 촬영할 수 있다. 동일 pass 안에서도 위성이 시각에 따라 자세를 달리하면 같은 strip을 다른 시각과 각도로 촬영할 수 있다. 초기 프로토타입은 연속적인 접근 가능 구간을 그대로 제어하지 않고 다음 최대 세 개 후보로 이산화한다.

- 접근 구간 초반
- 결합 off-nadir가 최소인 시점
- 접근 구간 후반

실제 궤도 전파와 가시성 계산은 RL core의 책임이 아니다. 촬영 기회는 외부 계산기 또는 seed 기반 가상 생성기가 사전 계산해 제공한다.

2026-07-07 설계 검토에서 현재 생성기가 pass 시간 구간과 무작위 접근 구간만으로 opportunity를 만들기 때문에, strip이 특정 pass의 궤도와 footprint 관점에서 왜 유효한지 시각적으로 확인할 수 없다는 허점이 확인됐다. 실제 궤도 데이터가 아직 없으므로 중간 단계로 가상 ground track과 footprint 생성기를 추가한다.

가상 생성기는 pass별 ground track 좌표, footprint 또는 swath 근사 영역, footprint-strip 교차로부터 access window를 만든 뒤 opportunity를 파생해야 한다. 이는 정밀 궤도 모델이 아니라 실제 데이터가 들어오기 전까지 opportunity의 공간적 근거를 보존하고 지도 검수를 가능하게 하는 전처리 계층이다.

구현된 가상 생성기는 20초 간격의 pass 샘플, pass 진행 방향에 맞춘 strip polygon, 회전 footprint polygon, footprint-strip 교차 기반 access window 및 `source_access_window_id`를 가진 opportunity를 생성한다. 이는 공간적 근거 추적을 위한 가상 데이터이며 실제 궤도 전파나 센서 물리 모델 검증을 대체하지 않는다.

pitch를 0으로 고정하더라도 strip이 위경도 축 정렬 사각형이어야 하는 것은 아니다. 이 프로젝트의 가상 모델에서는 pitch 0을 along-track 방향으로 앞뒤를 기울여 보지 않는다는 의미로 해석하고, strip의 긴 축은 pass의 ground track 진행 방향을 따르게 한다. roll/tilt는 여전히 요구 자세와 off-nadir 근사를 표현하는 값으로 사용한다.

세 후보 이산화는 실제 위성 운용의 일반 법칙이 아니라 행동 공간을 제한하기 위한 프로젝트 가정이다.

### 3.3 자세와 off-nadir 근사

**상태:** 가정  
**마지막 갱신:** 2026-07-26

프로토타입은 pitch를 `0 deg`로 고정하고 roll과 tilt만 사용한다. 결합 기울기는 다음 값으로 단순화한다.

```text
off_nadir = sqrt(roll^2 + tilt^2)
```

이 값은 두 방향의 기울기를 하나의 품질 선호도로 표현하기 위한 근사이며 정밀한 3차원 자세 또는 센서 시선 계산식으로 취급하지 않는다.

물리적으로 지나치게 큰 자세를 차단하기 위해 축별 `-30~30 deg`와 결합 off-nadir `30 deg` 제한을 사용한다. 이 제한도 초기 학습 환경을 위한 프로젝트 파라미터다.

2026-07-26 결과·재생 지도에 실제 조준점을 시각화하는 기능(`rl_core.generator.resolve_attitude_look_point`)을 추가하며 `_attitude_for_time`이 쓰는 `ATTITUDE_DEG_PER_GROUND_DEG = 8.0`(자세 각도 1도당 지상 거리 1/8도로 환산하는 상수)의 근거를 git 이력에서 확인했다. 이 상수는 2026-07-07 가상 ground track/footprint 생성기 도입 커밋(`03e8d65`)에서 `FOOTPRINT_HALF_ALONG_DEG`, `STRIP_ALONG_DEG` 등 다른 가상 지오메트리 상수와 함께 처음 등장했으며, 코드 주석이나 설계 문서 어디에도 8.0이라는 값의 물리적 근거(위성 고도, 센서 화각 등)는 기록되어 있지 않다. 즉 이 값은 "±30도 자세 제한 안에서 그럴듯한 각도가 나오게" 맞춘 임의 상수이며, 실제 위성 물리에서 유도된 값이 아니다.

이 상수는 strip 크기(`STRIP_ALONG_DEG=0.45도`)에 비해 상대적으로 작은 각도(1도 미만)도 무시할 수 없는 지상 거리(1도당 약 13.9km)로 환산해, 지도 시각화에서 "각도는 작은데 strip 대비 상당히 벗어나 보이는" 결과를 만들 수 있다는 것을 확인했다(관찰: 2026-07-26 attitude-target 시각화 작업 중 `order-018-strip-03` 사례). 정밀 자세 모델로 교체할 때는 이 상수와 그 각도-거리 환산 비율도 함께 재검토해야 한다 — 단순 시각화 문제가 아니라 opportunity별 roll/tilt 분포(±27도 clamp 도달 빈도, 평균 off-nadir 등) 자체를 바꾸므로 PPO 재학습 필요 여부도 같이 판단해야 한다.

### 3.4 자세 전환과 연속 촬영

**상태:** 가정  
**마지막 갱신:** 2026-07-06

roll과 tilt는 동시에 회전하지 않는 것으로 단순화하므로 전환시간은 축별 시간을 합산한다.

```text
slew_time =
    abs(target_roll - current_roll) / roll_rate
  + abs(target_tilt - current_tilt) / tilt_rate
  + settling_time
```

요구 자세가 현재 자세와 같으면 전환시간은 0이다. 촬영 후에는 마지막 촬영 자세를 유지한다.

다음 촬영은 이전 촬영 종료 후 최소 촬영 간격과 자세 전환시간을 모두 만족해야 한다.

```text
earliest_next_start =
    previous_end + max(minimum_interval, slew_time)
```

현재 각속도, 안정화 시간 및 최소 간격은 실제 기체 제원을 나타내는 값이 아니라 학습용 환경 파라미터다.

## 4. 스케줄링 문제에 관한 지식

### 4.1 행동 대상은 주문보다 촬영 기회에 가깝다

**상태:** 확정  
**마지막 갱신:** 2026-07-06

같은 strip에도 여러 시각과 자세의 촬영 기회가 있으므로 `strip을 촬영한다`만으로는 행동을 완전히 표현할 수 없다. 에이전트의 실제 선택은 `(strip, opportunity)`다.

그러나 하루의 모든 opportunity를 하나의 고정 행동 공간으로 펼치면 행동 수가 지나치게 커진다. 따라서 현재 이벤트의 후보만 최대 128개까지 제공하고, 미래 정보는 strip별 요약 관측으로 전달한다.

### 4.2 실행 불가능과 좋지 않은 선택은 구분해야 한다

**상태:** 확정  
**마지막 갱신:** 2026-07-06

시간, 자세, 중복 촬영과 같은 하드 제약 위반은 음의 보상으로 학습시키지 않고 action mask로 차단한다. 에이전트는 실행 가능한 행동 중 어떤 선택이 장기적으로 더 좋은지를 학습한다.

이 구분은 물리법칙을 시행착오로 발견하게 하는 대신 스케줄링 전략 학습에 용량을 집중시키려는 설계다.

데이터 검증과 action mask도 구분한다. 존재하지 않는 ID 참조나 촬영시간이 접근 구간에 들어가지 않는 구조적 모순은 시나리오 로드 시 거부한다. 현재 자세에서 전환시간이 부족하거나 이미 촬영한 strip처럼 state에 따라 달라지는 실행 가능성은 시뮬레이터가 마스킹한다.

이 경계 덕분에 잘못된 입력 파일은 조기에 발견하면서도, 실행 불가능한 기회와 그 사유를 정책 분석에 남길 수 있다. 구현된 계약은 [데이터 형식](data-format.md)을 참고한다.

### 4.3 주문 크기에 따른 보상 왜곡

**상태:** 확정  
**마지막 갱신:** 2026-07-06

strip마다 주문 우선순위 전체 점수를 주면 strip이 많은 주문이 의도보다 큰 가치를 갖게 된다. 이를 방지하기 위해 주문의 우선순위 점수 `P`를 전체 strip 수 `N`으로 나눈다.

```text
strip_base_reward = P / N
```

따라서 주문 크기와 관계없이 모든 strip을 촬영했을 때 기본 보상 합계는 `red=5`, `blue=3`, `background=1`을 유지한다.

## 5. RL 문제 구성에 관한 지식

### 5.1 Episode와 event 기반 step

**상태:** 확정  
**마지막 갱신:** 2026-07-06

에피소드 하나는 고정된 시나리오의 1일 전체 일정이다. `(state, action)` 한 번은 에피소드가 아니라 하나의 step이다.

매초 의사결정하면 대부분의 step이 기다림이 되므로 촬영과 관련된 의미 있는 시각에서만 결정하는 event 기반 환경을 사용한다.

### 5.2 고정 시나리오 학습의 의미

**상태:** 확정  
**마지막 갱신:** 2026-07-06

초기 목표는 처음 보는 시나리오에 일반화하는 정책이 아니라 동일한 하루 시나리오에서 좋은 스케줄을 찾는 것이다. 따라서 매 에피소드의 주문, 궤도 및 촬영 기회가 동일해도 현재 목표와 모순되지 않는다.

다만 이 방식에서 높은 성능은 일반적인 위성 스케줄링 능력보다 해당 시나리오에 대한 과적합 또는 반복 최적화 결과일 수 있다. 향후 여러 시나리오로 확장할 때 별도의 일반화 평가가 필요하다.

### 5.3 RL 결과에는 기준점이 필요하다

**상태:** 확정
**마지막 갱신:** 2026-07-13

학습 return만으로 정책의 품질을 판단하기 어렵다. Random valid, Earliest deadline first, Priority greedy 및 Priority-efficiency greedy와 동일 시나리오에서 비교해야 한다.

축소된 문제에서는 전수조사나 수리 최적화 결과와 비교해 optimality gap을 측정할 수 있다. 큰 문제에서는 최적성 보장보다 주요 휴리스틱 대비 개선을 평가한다.

2026-07-13 설계 구체화에서 CP-SAT은 RL 결과를 고치는 후처리기가 아니라 `tiny`/`small` 시나리오의 정교한 기준해를 만드는 solver baseline으로 정의했다. 각 opportunity의 선택 여부를 0/1 변수로 두고, 같은 strip 중복, 시간 겹침, 최소 촬영 간격 및 자세 전환 불가능 후보 쌍을 제약으로 차단한다. CP-SAT 선택 결과는 기존 simulator로 다시 평가해 `EpisodeReplay`와 `PolicyComparison`에 포함해야 한다.

2026-07-13 1차 구현에서 `rl_core/optimization.py`는 tiny 시나리오에 대해 OR-Tools CP-SAT 기준해를 생성하고, 선택된 opportunity ID 목록을 simulator replay로 재검증한다. 이 구현은 CP-SAT 목적함수와 공식 simulator return을 분리해 기록하므로, solver 모델의 선형 근사와 실제 reward breakdown의 차이를 추적할 수 있다.

이 결정의 의미는 Maskable PPO가 Random valid보다 나은지뿐 아니라, 최적화 solver 기준해에 얼마나 가까운지 볼 수 있게 하는 것이다. 다만 CP-SAT baseline의 신뢰도는 solver 모델이 simulator 제약과 reward 의미를 얼마나 정확히 반영하는지에 달려 있다.

### 5.4 초기 알고리즘 선택

**상태:** 확정  
**마지막 갱신:** 2026-07-06

초기 RL 알고리즘은 Maskable PPO를 사용한다. 현재 문제는 이산 행동을 사용하고 각 step마다 하드 제약으로 선택 가능한 후보가 달라지므로 action masking을 직접 적용할 수 있는 정책이 적합하다.

Maskable PPO 선택은 이 알고리즘이 최적해를 보장하기 때문이 아니라, 실행 불가능한 행동을 제거한 상태에서 비교적 명확한 구현 경로로 policy gradient 학습을 시작하기 위한 것이다. 알고리즘 비교보다 환경, 보상 및 마스킹의 정확성을 먼저 검증한다.

Random 및 greedy 정책은 baseline으로, CP-SAT이나 정수계획은 축소 문제의 최적성 비교 도구로 구분한다.

## 6. 시스템 구조에 관한 지식

### 6.1 RL core와 웹의 분리

**상태:** 확정  
**마지막 갱신:** 2026-07-06

RL core는 웹 프레임워크에 의존하지 않는 독립 Python 모듈이어야 한다. 웹은 시나리오 수정, 학습 제어 및 결과 시각화를 담당하며 보상과 action mask 같은 도메인 규칙을 다시 구현하지 않는다.

이 분리는 같은 환경을 테스트, CLI, Backend 및 향후 다른 UI에서 재사용하기 위한 것이다.

### 6.2 학습 worker 분리

**상태:** 확정  
**마지막 갱신:** 2026-07-06

RL 학습은 오래 실행되고 CPU 또는 GPU를 지속적으로 사용할 수 있으므로 웹 요청 처리 프로세스와 분리한다. GUI 연결이 끊기거나 새로 고쳐져도 학습 실행은 계속되어야 한다.

초기에는 로컬 단일 worker로 시작하고 분산 작업 큐는 필요할 때 검토한다.

### 6.3 SQLite 메타데이터와 파일 artifact의 분리

**상태:** 확정
**마지막 갱신:** 2026-07-19

SQLite는 시나리오와 실행의 조회·상태 복구에 필요한 작은 메타데이터와 artifact의 상대 경로·해시만 저장하고, Scenario JSON, replay, 비교 결과, 모델 파일 같은 대용량 원본은 로컬 파일로 유지한다. 이 분리는 SQLite에 대형 step 로그를 중복 저장하지 않으면서도 Backend가 파일 누락과 손상을 확인할 수 있게 한다.

파일과 DB는 하나의 원자 transaction으로 묶을 수 없으므로, JSON은 임시 파일을 fsync한 뒤 원자 교체하고 DB 색인은 그 다음에 반영한다. 교체 실패 시 기존 완성 파일은 유지하며, DB에만 남은 누락 artifact는 복구 대상으로 명시적으로 탐지한다.

실행 상태 복구는 Backend가 아니라 worker supervisor의 책임이다. Backend만 재시작해도 별도 worker가 계속 실행될 수 있으므로, worker 부재를 확인한 supervisor 시작 시점에만 `running`과 `stop_requested` 상태를 terminal 상태로 정리한다. 초기 버전에서는 불확실한 자동 재개보다 checkpoint를 보존하고 사용자가 새 run으로 명시적으로 재시작하는 정책을 택한다.

초기 Backend 조회는 SQLite 메타데이터 목록과 검증된 Scenario 원본 조회를 분리한다. 목록에서 대용량 geometry와 opportunity를 반복 전송하지 않아 대시보드·목록 화면의 기본 조회 비용을 제한하고, 상세 화면에서만 전체 Scenario를 읽는다. 이 원칙은 이후 주문·strip·opportunity 전용 조회 API를 추가할 때도 유지한다.

Scenario 하위 목록은 초기 규모에서 전체 Scenario를 메모리로 읽어 필터·정렬한 뒤 pagination 한다. full 가상 시나리오의 opportunity 수는 수천 개 수준이라 조회 전용 초기 Backend에는 충분하지만, 실제 데이터가 커지면 opportunities를 SQLite에서 직접 filter/paginate할 수 있도록 정규화해야 한다.

저장된 Scenario의 재검증은 구조 검증과 artifact 무결성을 함께 확인해야 한다. Pydantic 구조 검증만으로는 외부에서 파일을 유효한 JSON으로 바꾼 경우를 찾지 못하므로, 저장 시 기록한 SHA-256과 실제 파일의 hash도 비교한다. 반면 action mask 조건은 scenario 파일이 아니라 simulator state에 의존하므로 validation API의 오류 목록에 포함하지 않는다.

기준 정책 평가는 학습 worker보다 먼저 동기 API로 연결할 수 있다. 결정론적 휴리스틱은 짧은 시간에 종료되고 `EpisodeReplay`를 바로 남기므로, EvaluationRun 상태 전이·artifact 저장·오류 반환을 검증하는 안전한 첫 실행 경로가 된다. 장시간 학습이나 solver 실행을 같은 HTTP 요청에서 처리하면 서버 응답성과 worker 분리 원칙을 해치므로 후속 단계에서 분리한다.

### 6.4 초기 개발 기반과 코어 구현

**상태:** 확정  
**마지막 갱신:** 2026-07-06

초기 개발 기반은 Python 3.12, Pydantic 2, Pytest, Ruff 및 Mypy를 사용한다. Frontend 기반은 Node.js 22, React 19, TypeScript 6 및 Vite 8로 구성했다.

결정론적 시뮬레이션 코어는 Gymnasium이나 웹 프레임워크에 의존하지 않는다. 데이터 계약, seed 기반 시나리오 생성기와 이벤트 시뮬레이터를 먼저 독립적으로 검증했으며, 이후 기준 정책과 Gymnasium wrapper가 이 코어를 재사용한다.

### 6.5 Full 가상 시나리오 초기 관찰

**상태:** 관찰  
**마지막 갱신:** 2026-07-07

최초 생성기는 접근 구간 시작을 연속 난수로 선택했다. 이 방식의 seed `20260706` full 생성 결과는 주문 100개, strip 543개와 opportunity 3,195개였고 정상 종료했지만, 동일 시각의 경쟁 후보가 거의 없어 네 기준 정책이 같은 결과를 냈다.

이를 보완해 접근 구간 시작을 10초 grid로 양자화했다. 변경 후 seed `20260707`은 strip 572개, opportunity 3,447개를 생성했으며 정책별 episode에서 54~57개의 경쟁 step이 관찰됐다. 네 정책의 return과 완료 주문 수가 서로 달라져 비교 기준으로 기능했다.

이 값들은 생성기 동작 확인을 위한 단일 seed 관찰이며 일반적인 분포나 성능 기준으로 간주하지 않는다. 후보 128개 제한과 10초 양자화의 영향은 여러 seed 및 향후 실제 궤도 데이터에서 재검토해야 한다.

### 6.6 기준 정책 구현에서 얻은 지식

**상태:** 확정과 관찰 혼합  
**마지막 갱신:** 2026-07-07

기준 정책은 동일한 simulator와 action mask를 사용해야 공정하게 비교할 수 있다. 정책마다 별도 제약 판정을 구현하지 않고, 현재 유효 slot 중 선택 기준만 달리한다.

- Random valid는 seed가 고정된 난수로 유효 후보를 선택한다.
- Earliest deadline first는 주문 마감이 가장 가까운 후보를 선택한다.
- Priority greedy는 주문 우선순위 점수가 가장 높은 후보를 선택한다.
- Priority-efficiency greedy는 `priority / (imaging time + slew time)`이 가장 높은 후보를 선택한다.

정책의 절대 점수보다 동일 시나리오와 seed에서 결과가 재현되고, 마스킹된 행동을 선택하지 않으며, 서로 다른 선택 기준이 실제로 다른 스케줄을 만드는지가 먼저 검증되어야 한다.

### 6.6 Gymnasium wrapper의 책임 경계

**상태:** 확정  
**마지막 갱신:** 2026-07-07

Gymnasium wrapper는 simulator의 보상과 제약을 다시 구현하지 않고 RL 라이브러리가 요구하는 고정 observation, action space, 반환 tuple과 action mask로 변환하는 adapter다.

가변 strip과 후보는 최대 2,000개와 128개로 padding하고 presence 배열을 별도로 제공한다. Presence는 실제 데이터와 padding을 구분하며, action mask는 현재 실행 가능한 행동을 구분한다. 두 개념을 하나의 mask로 혼합하지 않는다.

초기 호환 조합은 Gymnasium 1.3, Stable-Baselines3 2.9 및 sb3-contrib 2.9다. `MultiInputPolicy` 예측과 짧은 Maskable PPO rollout을 통해 Dict 관측과 환경 내부 `action_masks()` 연동을 확인했다.

일반 Gym checker처럼 action mask를 인식하지 않는 호출이 invalid action을 전달하면 wrapper는 촬영을 실행하지 않고 skip으로 변환한다. 이를 `info`에 남겨 하드 제약을 지키면서도 외부 호출 오류를 관찰할 수 있게 한다.

### 6.7 학습 산출물과 평가 분리

**상태:** 확정
**마지막 갱신:** 2026-07-07

Maskable PPO 학습은 학습 중 rollout return만으로 판단하지 않고, 별도의 평가 seed로 고정 시나리오를 주기적으로 실행해 metric을 남긴다. 이렇게 하면 정책 업데이트 과정의 noisy한 학습 로그와 실제 비교용 episode 결과를 구분할 수 있다.

초기 artifact 구조는 `data/runs/<run-id>/` 아래에 설정, run 상태, checkpoint, metric 및 최종 모델을 함께 둔다. 이 구조는 나중에 Backend와 웹이 실행 상태와 결과 파일을 추적하기 위한 최소 단위가 된다.

현재 구현은 단일 Gym 환경에서 시작하므로 `batch_size <= n_steps`를 요구한다. 병렬 환경 또는 더 큰 rollout으로 확장할 때 이 제약과 하이퍼파라미터 기본값은 다시 검토한다.

### 6.8 단계 6 성능 검증 관찰

**상태:** 관찰
**마지막 갱신:** 2026-07-08

`synthetic-tiny-20260707`에서 Maskable PPO 5개 학습 seed와 Random valid 5개 평가 seed를 비교했다. Maskable PPO는 50,000 timestep, `n_steps=256`, `batch_size=64`, `n_epochs=5` 설정으로 학습했고, 단계 6 통과 기준을 모두 만족했다.

PPO median return은 `5.326453530248241`, Random valid median return은 `5.325396139404043`으로 PPO가 근소하게 높았다. 두 정책 모두 median completed strips는 `9`였고, PPO의 median skip ratio는 `0.1`, non-skip action concentration은 약 `0.556`으로 skip 반복이나 단일 action slot 고착은 관찰되지 않았다.

이 결과는 tiny 시나리오에서 저장, 평가, action mask 및 반복 seed 검증 파이프라인이 작동하고 Random valid 기준선을 넘었다는 근거다. 다만 문제 규모가 작아 두 정책이 같은 수의 strip/order를 완료했고 차이는 주로 angle bonus에서 발생했다. 따라서 이 결과를 일반적인 스케줄링 성능 향상으로 확대 해석하지 않고, small/full 시나리오와 greedy 기준 정책 비교에서 다시 검증해야 한다.

### 6.9 Episode replay 로그의 역할

**상태:** 확정
**마지막 갱신:** 2026-07-08

평가 episode 재생은 사후에 simulator 규칙을 다시 실행해 추정하지 않고, 선택 당시의 state, 후보, action mask 사유, 선택 action, reward breakdown 및 선택 후 state를 `EpisodeReplay`로 저장한다. 이렇게 해야 나중에 웹 화면이나 분석 도구가 정책이 실제로 본 후보와 마스킹 사유를 그대로 확인할 수 있다.

기준 정책과 Maskable PPO 평가는 같은 replay 계약을 사용한다. 따라서 정책 비교 화면은 서로 다른 정책 결과를 같은 step 로그 구조로 읽을 수 있고, 보상 합계와 누적 return 일치도 파일만으로 검증할 수 있다.

`PolicyComparison`은 같은 시나리오의 여러 replay를 요약 지표와 함께 묶는다. 최고 정책은 총 return을 우선으로 고르고, 동률일 때 완료 주문 수, 완료 strip 수, 촬영 수를 차례로 비교한다. 이 기준은 화면 정렬과 빠른 요약을 위한 artifact 규칙이며, 도메인 최적성 보장을 의미하지 않는다.

### 6.10 단계 14 1차 검증에서 확인한 지식

**상태:** 확정
**마지막 갱신:** 2026-07-26

학습·평가 run 생성 API(`POST /api/training-runs`, `POST /api/evaluation-runs`, `POST /api/cp-sat-evaluation-runs`)가 공유하던 `_load_scenario_or_api_error`는 scenario JSON을 Pydantic으로 다시 파싱만 할 뿐 저장된 SHA-256과 비교하지 않았다. 즉 파일이 외부에서 손상되거나 수정됐지만 구조상 여전히 유효한 `Scenario`로 파싱되는 경우, 학습·평가가 조용히 시작될 수 있었다. `repository.validate_scenario()`는 이미 해시·구조를 모두 검증하는 기능을 갖고 있었지만 조회 API(`GET /api/scenarios/{id}/validation`)에서만 쓰이고 있었다. 이 gap은 코드 검토로만 드러났고 기존 테스트에는 없었다 — write endpoint에 대한 무결성 검증은 read endpoint와 별개로 명시적으로 확인해야 한다는 재사용 가능한 교훈이다. write endpoint에는 `validate_scenario()`까지 통과해야 하는 `_load_valid_scenario_or_api_error`를 별도로 두고, 읽기 전용 조회 API는 기존 동작을 유지했다(`docs/web-application-design.md` 참고).

모델·설정의 추적 가능성은 `EvaluationRun.source_training_run_id`부터 `GET /api/training-runs/{run_id}/detail`의 config snapshot·checkpoint 목록까지는 API로 완전히 보장되지만, 실제 모델 파일(`.zip`)을 가리키는 API는 의도적으로 두지 않았다. 이 프로젝트는 로컬 단일 사용자 프로토타입이라 운영자가 `data/` 파일시스템에 직접 접근할 수 있으므로, `TrainingRun.artifact_directory` + 고정된 저장 규칙(`model/final-model.zip`)을 문서화하는 것으로 추적 가능성 요건을 충족한다고 판단했다. 다중 사용자·원격 배포로 확장하면 이 가정을 재검토해야 한다.

### 6.11 full 규모 Maskable PPO 학습의 처리량·메모리 관찰

**상태:** 관찰
**마지막 갱신:** 2026-07-26

`tools/stage14_scale_benchmark.py`로 seed `20260707`의 tiny/small/full 시나리오를 같은 하이퍼파라미터(`n_steps=256, batch_size=64, n_epochs=5`, CPU 고정)로 측정한 결과, **메모리는 규모에 따라 뚜렷하게 늘지 않았지만(peak RSS가 tiny 733MB, small 381MB, full 811MB로 규모와 단조 증가하지 않음 — 최대 7% 차이), 처리량은 full이 tiny 대비 약 17배 느렸다**(본측정 기준 tiny 493.9 steps/sec, full 29.3 steps/sec; 짧은 calibration 단발 측정에서도 16.6배로 일관되게 재현됨).

관측 공간이 시나리오 규모와 무관하게 고정 크기(strip 2,000칸, 후보 128칸 padding, `rl_core/models.py:163-164`)로 padding되므로 신경망 forward/backward 비용은 규모와 거의 무관할 것으로 예상되고, 실제로 메모리는 그 예상과 일치했다. 반면 처리량 저하는 신경망이 아니라 **시뮬레이터 쪽 병목**(매 step마다 실제 strip/opportunity 수에 비례해 유효 후보를 계산하는 비용, 또는 학습 종료마다 1회 필수로 도는 전체 episode 평가가 full 규모에서 훨씬 오래 걸리는 것)일 가능성이 높다고 본다. 정확한 원인은 profiling이 더 필요해 이번 측정 범위 밖으로 남겨둔다.

방법론적 한계: 256-step 단발 calibration은 학습 호출 끝에 항상 붙는 고정 비용(최종 평가 1회)의 비중이 짧은 실행일수록 커서 실제 처리량을 상당히 과소평가한다(tiny 기준 calibration 148.9 vs 본측정 493.9 steps/sec). 적응형 예산 배분이 이 과소평가된 처리량으로 timesteps를 계산해, 실제로는 25분 예산 중 12.6분만 쓰고 끝났다 — 데이터 자체는 유효하지만 향후 같은 스크립트를 재사용할 때 감안해야 한다.

실무적 함의: full 규모에서 `tools/stage6_benchmark.py` 수준(50,000 timesteps) 학습 1회는 대략 28~30분이 걸린다(29.3 steps/sec 기준). "RL과 모든 기준 정책 비교"(단계 14 나머지 항목)에서 full 규모 학습 시간 예산을 잡을 때 이 수치를 근거로 쓴다.

### 6.12 full 규모 RL·모든 기준 정책 비교 관찰

**상태:** 관찰
**마지막 갱신:** 2026-07-26

`tools/stage14_full_scale_direction_check.py`로 seed `20260707`의 full 시나리오에서 Maskable PPO(2 seed, 각 30,000 timesteps)를 4개 baseline 휴리스틱과 CP-SAT(`time_limit_sec=900`) 전부와 비교했다 — full 규모에서 이 비교가 이뤄진 것은 이번이 처음이다. total return 기준 순위:

| 정책 | total_return |
|---|---|
| CP-SAT (`OPTIMAL`, gap 0.0) | 104.59 |
| Priority-efficiency greedy | 101.44 |
| Priority greedy / Earliest deadline first (정확히 동률) | 100.35 |
| **Maskable PPO (2 seed median)** | **98.42** |
| Random valid (3 seed median) | 97.30 |

**PPO는 Random valid보다는 일관되게 나았지만, 세 휴리스틱과 CP-SAT 최적해에는 못 미쳤다.** 두 seed의 학습 곡선(`evaluation_interval=2,000`, 15개 지점)이 서로 다른 양상을 보였다 — seed 23은 81.3(초반)에서 101.5(22,000~24,000 timesteps 부근, 이 시점엔 Priority-efficiency greedy를 일시적으로 앞섰음)까지 뚜렷하게 상승했다가 후반(28,000~30,000)에 99.3으로 다소 내려오며 마무리됐다. seed 11은 초반부터 이미 baseline 근처(97.5)에서 시작해 30,000 timesteps 내내 큰 변화 없이 평평했다. 즉 **학습이 방향성을 보인다는 신호는 뚜렷하지만(seed 23), seed에 따라 편차가 크고 30,000 timesteps로는 아직 최상위 휴리스틱·CP-SAT을 안정적으로 넘어서지 못한다.**

CP-SAT은 우려와 달리 15분 제한 훨씬 이전에(스모크 테스트에서 5초 제한으로도) `OPTIMAL`·gap 0.0에 도달했다 — full 규모(opportunity 약 3,447개)에서도 조합폭발로 시간 제한에 걸릴 것이라는 사전 우려는 이번 시나리오·seed 조합에서는 근거가 없었다. 다만 이는 단일 seed·단일 시나리오 관찰이라 일반화하지 않는다.

Priority greedy와 Earliest deadline first가 이 시나리오에서 total_return까지 정확히 동일하게 나온 것은 우연이거나(두 기준이 이 시나리오에서 같은 선택 순서를 만든 경우), 시나리오 구조상 두 정렬 기준이 자주 일치하기 때문일 수 있다 — 근거 확인이 더 필요해 결론 내리지 않는다.

이 결과로 `docs/implementation-plan.md` 단계 14 "RL과 모든 기준 정책 비교" 완료 조건("RL 정책이 최소한 Random valid와 정량적으로 비교된다")을 충족했다 — PPO가 모든 기준을 이겨야 한다는 합격선은 두지 않았고, 실제 정량 비교가 처음으로 이뤄졌다는 사실 자체가 완료 조건이다.

### 6.13 order 간 공간적 겹침 엔지니어링

**상태:** 확정
**마지막 갱신:** 2026-07-27

6.12절 결과를 검토하던 중 "촬영이 겹치는 상황에서 무엇을 먼저 찍는 게 최선인가"를 생성기가 실제로 테스트하는지 의문이 제기됐다. `generate_scenario()`가 order마다 `rng.choice(footprint_samples)`로 완전히 독립적인 위치를 뽑는지 읽기 전용으로 실측한 결과(seed `20260707`, order 쌍 전수 검사):

| 규모 | order 쌍 수 | bbox 겹침 | 포함관계 | 실제 strip 겹침 |
|---|---|---|---|---|
| tiny | 10 | 0 (0%) | 0 | 0 |
| small | 190 | 1 (0.53%) | 0 | 1 |
| full | 4,950 | 1 (0.02%) | 0 | 1 |

포함관계(큰 order 안에 작은 order)는 세 규모 모두 0건이었다. order 크기(strip 0.45°×0.12°)에 비해 전 세계 분포 범위가 훨씬 넓어, 독립 균등 랜덤으로는 겹침이 통계적으로 거의 발생하지 않는다는 것이 원인이다. 기존 "의미 있는 선택" 대응(10초 시간 grid 양자화, 6.10절 이전 및 `rl-study-notes.md`)은 시간 축 경쟁만 다뤘고 공간적 order 겹침은 다루지 않았다.

이를 보완해 `rl_core/generator.py`에 order 일부(약 30%, `OVERLAP_ORDER_FRACTION`)를 인접 index 쌍으로 묶어 `partial`(부분 겹침)/`full`(거의 완전 겹침)/`containment`(포함) 세 종류를 순환 배정하는 겹침 엔지니어링을 추가했다. 겹치는 쌍은 같은 anchor footprint(같은 pass)를 공유하고 request 기간도 서로 겹치도록 계산해, 공간적 겹침이 실제 시간적 경쟁으로 이어지게 했다. 나머지 다수 order는 기존과 동일한 독립 랜덤 배치로 남겼다. `Order`/`Strip` 스키마에 새 필드는 추가하지 않았다 — 겹침 쌍 여부는 인접 order index 규칙으로만 구분한다.

구현 후 같은 seed로 재측정한 결과:

| 규모 | order 쌍 수 | bbox 겹침 | 포함관계 | 실제 strip 겹침 |
|---|---|---|---|---|
| tiny | 10 | 1 (10.0%) | 1 | 1 |
| small | 190 | 4 (2.11%) | 2 | 4 |
| full | 4,950 | 15 (0.30%) | 11 | 15 |

세 규모 모두 겹치는 쌍(부분/완전/포함 혼재)이 최소 1건 이상 생기면서도, 대다수 order 쌍(90~99.7%)은 여전히 분리된 상태로 남아 "겹치는 구간과 안 겹치는 구간의 공존"을 확인했다. `tests/test_generator.py::test_generator_creates_mixed_overlapping_and_separated_orders`가 이 혼재 조건(겹침 존재, 포함관계 존재, 분리 쌍이 과반, 겹치는 쌍의 request 기간도 실제로 겹침)을 회귀 테스트로 고정한다.

참고: `full` 겹침의 상당수(15건 중 11건)가 진단 스크립트의 "포함관계" 판정(비엄격 부등식 `<=`/`>=`)에 함께 잡혔는데, 이는 `full` 종류(거의 동일한 좌표로 배치)가 수학적으로 상호 포함(mutual containment)의 특수 사례이기 때문이다 — 별도 결함이 아니라 겹침 판정 방식의 자연스러운 결과다.

### 6.14 겹침 엔지니어링 이후 full 규모 8-seed PPO 재검증

**상태:** 관찰
**마지막 갱신:** 2026-07-28

6.13절 겹침 엔지니어링 이후 시나리오 데이터가 바뀌었으므로, 6.12절의 2-seed 비교는 더 이상 유효한 비교 기준이 아니다. 또한 seed 2개는 "PPO가 안정적으로 학습되는가, seed 운인가"를 판단하기에 통계적으로 부족하다고 판단해(표준오차가 seed 수의 제곱근에 반비례하는 반면 비용은 선형으로 늘어 8개 이후로는 효율이 급격히 나빠짐), `tools/stage14_full_scale_direction_check.py`에 `--ppo-learning-seeds` CLI 인자를 추가해 학습 seed를 8개(`11,23,37,41,53,67,79,89`)로 늘려 재실행했다.

full 규모 8-seed 결과(seed `20260707`, `scenario_id: synthetic-full-20260707`, 각 30,000 timesteps):

| 정책 | total_return |
|---|---|
| CP-SAT (`OPTIMAL`, gap 0.0) | 65.50 |
| **Maskable PPO (8 seed median)** | **64.23** |
| Priority greedy | 62.33 |
| Priority-efficiency greedy | 62.27 |
| Earliest deadline first | 62.32 |
| Random valid (3 seed median) | 61.43 |

seed별 final total_return(오름차순): 62.28(seed 37), 63.79(89), 63.90(53), 64.13(79), 64.32(41), 64.41(23), 64.64(11), 64.65(67) — median 64.23, stdev 0.77.

**6.12절과 달리 이번에는 PPO가 세 휴리스틱 전부를 안정적으로 넘어섰다** — 8개 seed 중 7개가 세 휴리스틱 전부(62.27~62.33)보다 높았고, 유일하게 못 넘은 seed 37도 사실상 동률(62.28)이었다. CP-SAT 최적해(65.50)에는 여전히 못 미쳤지만, 겹침 엔지니어링으로 시나리오에 실제 선택 문제(공간+시간 경쟁)가 늘어난 것이 PPO가 단순 휴리스틱보다 나은 정책을 학습할 여지를 만들어준 것으로 보인다 — 다만 시나리오가 동시에 바뀌었으므로 "겹침 엔지니어링이 원인"이라고 단정하지는 않는다(seed 수 증가와 시나리오 변경이 함께 일어났다).

**6.12절에서 관찰한 "학습 후반 불안정성(상승 후 하락)"이 이번에도 재현됐다** — seed 37은 4,000 timesteps 지점에서 이미 64.29(다른 seed들과 동등한 수준)에 도달했지만, 이후 30,000 timesteps까지 계속 학습하며 62.28로 후퇴했다. 반면 나머지 7개 seed는 peak 대비 final 하락폭이 작았다(최대 0.45, 대부분 0.1 이내). 즉 이 불안정성은 seed에 따라 발생 여부와 정도가 다른 실제 현상이며, seed를 늘린다고 해결되지는 않는다 — 이전에 검토했던 학습 후반 안정화(learning rate 감쇠, best-checkpoint 저장/사용)가 여전히 유효한 다음 개선 방향이다.

**부수적으로 발견한 재현성 위험과 대응:** 이 실행 도중 컴퓨터가 다시 예기치 않게 재부팅되어(2026-07-26에 이어 두 번째) 중단됐다. 8개 중 5개 seed(11, 23, 37, 41, 53)는 이미 `metrics/final-evaluation.json`까지 완료된 상태였고, seed 67은 checkpoint 20,000/30,000에서 끊겼으며 79/89는 시작 전이었다. 이 문제가 반복될 것으로 보여 `tools/stage14_full_scale_direction_check.py`에 `--resume-from <이전 benchmark_root>` 옵션을 추가했다 — 완료된 seed(`final-evaluation.json` 존재)는 재학습 없이 그대로 재사용하고, 미완료 seed(체크포인트만 있어도)는 이어서가 아니라 처음부터 다시 학습한다(체크포인트 재개는 optimizer·RNG 상태 불일치 위험이 있어 선택하지 않았다). baseline 4종과 CP-SAT은 재계산 비용이 낮아 매번 새로 계산한다. 이 기능으로 seed 67/79/89 3개만 재학습(~51분)하면 됐고, 전체를 처음부터 다시 돌리지 않아도 됐다.

**범위에 대한 명확화(2026-07-28):** 이 8개 `learning_seed`는 전부 **동일한 시나리오**(`generate_scenario(seed=20260707, size="full")`)를 학습한 것이다 — 바뀐 건 신경망 초기화와 탐색 난수뿐이다. 즉 이 절의 "7/8 seed가 휴리스틱을 넘었다"는 결과는 "이 하나의 고정 시나리오를 안정적으로 잘 푸는가"를 확인한 것이지, 서로 다른 시나리오에 대한 일반화 능력을 확인한 게 아니다. 개념적 배경은 `docs/rl-study-notes.md`의 [학습 seed 다양화와 시나리오 일반화는 다른 문제다](rl-study-notes.md#학습-seed-다양화와-시나리오-일반화는-다른-문제다) 참고. 또한 이 표의 CP-SAT `65.50`을 "진짜 최적 상한"으로 읽지 않도록 주의한다 — 6.15절 참고.

### 6.15 CP-SAT의 "OPTIMAL"은 total_return 기준 최적이 아니다

**상태:** 확정(목적함수가 미완료 패널티를 제외한다는 사실) / 검토 필요(실제 개선 여지 크기)
**마지막 갱신:** 2026-07-28

6.14절에서 PPO가 CP-SAT을 못 넘는 이유를 검토하던 중, CP-SAT이 실제로 무엇을 maximize하는지 다시 확인했다. `rl_core/optimization.py`의 `_set_objective`/`_opportunity_reward`는 선택한 opportunity의 `strip_base + angle_bonus`만 합산하며, `missed_penalty`(주문 마감 시 미완료 비율에 비례해 깎이는 패널티, `RewardConfig.missed_penalty_weight` 기본값 0.5)는 목적함수에 전혀 들어가지 않는다. 이 설계 자체는 `docs/rl-scheduling-design.md` 12.1절에 "첫 버전" 단순화로 이미 문서화돼 있었다 — 새로 확인한 것은 그 단순화가 만드는 **실제 크기**와 **해석상 함의**다.

full 시나리오(seed 20260707)로 재확인한 수치:

- solver가 보고하는 `objective_value`: 119.091969
- replay에서 직접 합산한 `priority_score + angle_bonus`: 110.810 + 8.282 = 119.092 (오차 4e-6, 사실상 동일 — objective가 정확히 이 두 항만의 합임을 확인)
- replay의 `missed_penalty`: -53.595
- 최종 `total_return`(= 위 세 값의 합): 65.497

즉 CP-SAT은 "이 선택을 하면 몇 개 order를 완전히 포기하게 되는가"를 전혀 고려하지 않고 raw capture value만 최대화한다. `missed_penalty`는 선택 변수에 대해 선형식(주문별 완료 비율의 1차식)이라 목적함수에 추가하는 것 자체는 어렵지 않다. 목적함수를 `total_return`(= priority_score + angle_bonus + missed_penalty) 기준으로 바꿔 다시 풀면, 현재 해가 여전히 실행 가능한 후보로 남아있으므로 새 최적값은 수학적으로 **65.497 이상**이어야 한다(내려갈 수 없음) — 다만 얼마나 오를지는 재풀이 전에는 알 수 없다. `missed_penalty` 중 상당 부분이 애초에 물리적으로 완료 불가능한 order(접근 기회·자세 전환 제약상 실현 불가) 때문일 수 있어, 그런 부분은 목적함수를 고쳐도 줄지 않는다 — 실제 개선 여지는 재실행해봐야 안다.

실무적 함의: PPO가 CP-SAT의 65.50을 못 넘는다고 해서 "진짜 최적"에 못 미친다고 단정할 수 없다 — CP-SAT 자신도 `total_return` 기준으로는 최적이라는 보장이 없는 값을 내고 있기 때문이다. CP-SAT을 진짜 상한선으로 쓰려면 목적함수에 `missed_penalty` 항을 추가하는 수정이 먼저 필요하다.

### 6.16 학습 시나리오 하나에 대한 PPO 정책의 zero-shot 전이 확인

**상태:** 관찰
**마지막 갱신:** 2026-07-28

6.14절의 8-seed 실행은 전부 동일 시나리오(seed 20260707)를 학습한 것이었다(6.14절 명확화 참고). "이 정책이 다른(학습에 쓰이지 않은) 시나리오에서도 휴리스틱을 이길 수 있는가"를 직접 확인하기 위해, `tools/stage14_zero_shot_transfer_check.py`를 새로 만들어 가장 안정적이었던 학습 seed 11 모델(처음부터 227 strip/23 order를 유지)을 재학습 없이 unseen full 규모 시나리오 5개(seed 1001~1005)에 그대로 평가했다.

| unseen scenario seed | PPO zero-shot | 휴리스틱 최저 | random_valid median | 휴리스틱 전부 이김 |
|---|---|---|---|---|
| 1001 | 65.21 | 64.26 | 61.63 | Yes |
| 1002 | 59.39 | 64.07 | 59.16 | No |
| 1003 | 66.74 | 78.36 | 65.99 | No |
| 1004 | 61.48 | 68.90 | 58.98 | No |
| 1005 | 64.06 | 66.78 | 58.80 | No |

**random_valid는 5/5 시나리오에서 이겼지만(순수 무작위보다는 확실히 낫다), 세 휴리스틱 전부를 이긴 건 1/5뿐이었다.** 학습 시나리오(seed 20260707)에서는 8-seed 중 7/8이 세 휴리스틱을 전부 넘었던 것(6.14절)과 뚜렷하게 대비된다. 즉 이 정책이 학습 시나리오에서 보인 "휴리스틱보다 낫다"는 능력의 상당 부분은 그 시나리오 배치에 대한 과적합이고, 일반적으로 전이되는 전략(예: 우선순위·마감 고려)은 무작위보다 나은 수준까지만 학습된 것으로 보인다.

이 결과는 `docs/rl-study-notes.md`의 [학습 seed 다양화와 시나리오 일반화는 다른 문제다](rl-study-notes.md#학습-seed-다양화와-시나리오-일반화는-다른-문제다)에서 이론적으로 예상한 바를 실측으로 확인한 것이다. 향후 여러 시나리오에 대해 일반화하는 정책이 필요하면, 학습 중 `SCENARIO_SEED` 자체를 다양화하는 절차(domain randomization)가 필요하며, 그 경우 학습 시나리오 하나에 대한 최고 성능(지금의 64.23)은 오히려 낮아질 수 있다는 trade-off를 감안해야 한다.

### 6.17 Domain randomization 학습으로 zero-shot 일반화 개선 확인

**상태:** 관찰
**마지막 갱신:** 2026-07-29

6.16절에서 확인한 낮은 일반화 성능(단일 시나리오 학습 모델은 unseen 5개 중 1개만 세 휴리스틱을 이김)을 해결하기 위해, 시간이 걸려도 괜찮다는 사용자 판단에 따라 domain randomization 학습을 구현했다. `rl_core/gym_env.py::SatelliteSchedulingEnv`에 `scenario_pool` 옵션을 추가해(하위 호환 유지) episode마다 pool에서 시나리오를 다시 뽑고, `rl_core/training.py::train_maskable_ppo_with_scenario_pool()`을 새로 만들어(`storage`/`TrainingRun` 연동 없이) 여러 시나리오를 겪으며 학습하되 진행 상황은 pool에 없는 held-out 시나리오로 추적하게 했다. reward 스케일이 시나리오마다 다르다는 문제(6.16절)에는 `VecNormalize(norm_reward=True)`로 학습 시에만 대응했다. `tools/stage14_domain_randomization_check.py`로 학습→held-out 5개(1001~1005, 6.16절과 동일) zero-shot 비교를 자동화했다.

learning rate 감쇠 등 다른 안정화 기법은 "여러 시나리오 학습 자체가 되는지" 확인이 먼저라는 판단 아래 이번 범위에서 의도적으로 제외했다.

**tiny 파일럿(2,048 timesteps)**: `MaskablePPO`의 action masking이 `DummyVecEnv`+`VecNormalize` 조합을 통과하는지가 가장 우려했던 지점이었는데, 정상 동작을 확인했다(별도 코드 수정 없이 통과).

**small 파일럿(102,400 timesteps)**: held-out 5개 전부(5/5)가 세 휴리스틱을 이겼지만 마진이 매우 얇았고(예: 36.723 vs 36.715) 시나리오 하나는 random_valid에도 졌다 — small 규모는 문제 자체가 단순해 거의 모든 정책이 비슷한 상한에 수렴하는 것으로 보여, 이 결과만으로 결론 내리지 않았다.

**full 규모 본실행(500,000 timesteps, pool 20개 seed 2001~2020, held-out 5개는 6.16절과 동일)** — 진짜 검증 대상:

| unseen seed | 단일 시나리오 PPO(6.16절) | domain randomization PPO | 휴리스틱 최저 | 개선폭 |
|---|---|---|---|---|
| 1001 | 65.21 | 65.32 | 64.26 | +0.11 |
| 1002 | 59.39 | 63.20 | 64.07 | +3.81 |
| 1003 | 66.74 | 79.76 | 78.36 | +13.02 |
| 1004 | 61.48 | 68.09 | 68.90 | +6.61 |
| 1005 | 64.06 | 64.65 | 66.78 | +0.59 |

**5개 unseen 시나리오 전부에서 domain randomization 모델의 raw return이 단일 시나리오 모델보다 높았다** — 개선 방향 자체는 명확하고 일관적이다. 세 휴리스틱 전부를 이긴 시나리오 수는 1/5 → 2/5로 늘었고(random_valid는 여전히 5/5), 아직 3개 시나리오(1002, 1004, 1005)에서는 휴리스틱 최저점을 근소한 차이로 못 넘었다 — 다만 "이긴다/못 이긴다"는 이산적 지표라 못 넘은 3개도 실제로는 크게(4~7점) 좋아졌다는 걸 놓칠 수 있다. 즉 domain randomization은 "일반화 문제를 완전히 해결"하지는 않았지만, 방향성 있고 일관된 개선을 만들었다는 게 이번 실측의 핵심 결론이다.

held-out 학습 곡선(seed 1001, 48개 지점)도 이를 뒷받침한다 — 첫 1/4 구간 평균 63.38에서 마지막 1/4 구간 평균 65.11로 뚜렷하게 상승했고(등락은 있지만 6.14절 seed 37처럼 한 번 찾은 좋은 지점을 완전히 잃어버리는 붕괴는 없었다), small 파일럿에서 본 "초반에 평평해지는" 패턴과 달리 500,000 timestep 동안 계속 실질적으로 학습이 진행됐다.

**다음 후보(사용자 확인 필요)**: (1) 세 시나리오가 근소하게 못 넘은 만큼, 더 큰 timesteps 예산이나 더 큰 pool로 재시도, (2) 이번에 의도적으로 제외한 learning rate 감쇠를 domain randomization 학습에도 적용, (3) pool 크기·held-out 개수를 늘려 이 결과가 우연이 아닌지 재확인.

## 7. 용어집

| 용어 | 프로젝트에서의 의미 |
|---|---|
| Order | 공통 우선순위와 요구 기간을 가진 지리적 주문 영역 |
| Strip | 위성의 1회 촬영 단위인 직사각형 영역 |
| Opportunity | 특정 strip을 특정 pass, 시각 및 자세로 촬영할 수 있는 후보 |
| Pass | 위성 한 대의 연속 궤도 중 하나의 접근 구간 |
| Ground track | pass 중 위성의 지상점이 시간에 따라 이동한 가상 또는 실제 궤적 |
| Footprint | 특정 시각과 자세에서 센서가 지면에서 덮는 가상 또는 실제 영역 |
| Access window | footprint가 특정 strip과 교차해 촬영 후보를 만들 수 있는 연속 시간 구간 |
| Off-nadir | 정면 관측 방향에서 벗어난 정도를 나타내는 값 |
| Slew | 현재 자세에서 다음 촬영 자세로 전환하는 기동 |
| Settling time | 자세 전환 후 촬영 안정화를 위해 필요한 시간 |
| Action mask | 실행 불가능한 행동이 정책에서 선택되지 않도록 차단하는 값 |
| Baseline | RL 정책의 성능을 판단하기 위한 비교 정책 |
| CP-SAT baseline | 축소 시나리오에서 최적화 solver로 만든 비교용 기준해 |

## 8. 미해결 질문과 향후 검토

| 주제 | 상태 | 내용 |
|---|---|---|
| 실제 궤도 인터페이스 | 검토 필요 | 가상 opportunity 형식을 실제 궤도 전파 결과와 연결할 때 데이터 계약 검토 필요 |
| 가상 footprint 생성기 | 관찰 | 20초 샘플 ground track, 회전 strip/footprint polygon 및 access window 기반 opportunity 생성은 구현됨. 브라우저 지도에서 pass 기울기와 strip 방향 재확인 필요 |
| 정밀 자세 모델 | 검토 필요 | 현재 off-nadir 및 축별 순차 기동은 단순 근사이므로 실제 기체 적용 전 교체 필요. `ATTITUDE_DEG_PER_GROUND_DEG=8.0`은 물리적 근거 없는 임의 상수이며, 교체 시 opportunity의 roll/tilt 분포가 바뀌어 PPO 재학습 필요 여부도 함께 판단해야 한다 |
| 다중 위성 | 검토 필요 | 위성 두 대 확장 시 충돌, 주문 공유 및 행동 공간 설계 필요 |
| 구름과 영상 품질 | 검토 필요 | strip별 품질과 불확실성을 관측 및 보상에 반영하는 방법 필요 |
| 일반화 평가 | 검토 필요 | 여러 시나리오 학습 시 훈련/검증/평가 seed 분리 필요 |
| 후보 128개 제한 | 검토 필요 | 실제 시나리오에서 동시 후보 분포와 잘림 영향 측정 필요 |
| tiny 성능 개선 폭 | 관찰 | 단계 6에서는 Random valid보다 근소하게 높았지만 완료 strip/order 수가 같아 larger scenario에서 재검증 필요 |
| CP-SAT baseline 정합성 | 관찰 | tiny와 small 시나리오의 CP-SAT 선택 결과는 simulator replay 검증을 통과했다. `INFEASIBLE`·`UNKNOWN` artifact와 전체 정책 비교도 테스트했지만, 실제 대규모 입력에서 time limit이 발생했을 때의 성능·bound 해석은 후속 분석이 필요하다. |

## 9. 변경 기록

- 2026-07-26: roll/tilt 시각화 기능 구현 중 `ATTITUDE_DEG_PER_GROUND_DEG=8.0`이 물리적 근거 없이 2026-07-07 가상 생성기 도입 시 정해진 임의 상수임을 git 이력으로 확인했고, strip 크기 대비 작지 않은 지상 거리로 환산된다는 것을 관찰해 기록했다.
- 2026-07-26: 학습·평가 run 생성 API가 scenario artifact의 SHA-256 불일치를 검사하지 않던 gap을 발견해 `_load_valid_scenario_or_api_error`로 수정했고, 모델 파일 추적은 다운로드 API 대신 로컬 경로 규칙 문서화로 충분하다는 판단을 기록했다.
- 2026-07-23: 정책 비교는 화면이 최신 실행을 추정해 합치는 방식 대신 사용자가 같은 scenario·seed의 완료 EvaluationRun을 명시적으로 선택해 immutable artifact로 고정한다. 각 비교 행에 evaluation run ID를 보존하면 동일 정책을 여러 번 실행했어도 결과와 replay 링크가 다른 실행으로 연결되지 않는다.
- 2026-07-23: episode 재생은 전체 replay를 브라우저에 다시 계산하거나 한꺼번에 적재하지 않고, episode 요약의 step 수와 direct step 조회를 이용해 현재 step만 읽도록 구성했다. 이 방식은 후보와 action mask 상세는 유지하면서 큰 replay의 초기 화면 비용을 제한한다.
- 2026-07-22: 학습 제어 UI는 worker의 메모리 상태가 아니라 저장된 `TrainingRun`, config snapshot 및 append-only metrics만 polling해야 새로고침과 Backend 재시작 뒤에도 같은 run을 복구할 수 있다. `stop_requested`는 완료 상태가 아니라 cooperative cancellation이 checkpoint를 보존 중인 중간 상태이므로 별도 표시한다.
- 2026-07-22: 평가 지도에서 strip은 schedule의 capture 유무로 이진 완료 상태를 판단하고, 부분 완료는 여러 strip을 가진 order에만 적용했다. 우선순위와 실행 상태를 각각 테두리·채움으로 분리하면 중요한 주문과 실제 수행 결과를 같은 지도에서 혼동 없이 볼 수 있다. 선택 capture의 상세 설명은 schedule을 확장하지 않고, `step_index`로 검증된 replay step을 직접 조회해 제공한다.
- 2026-07-22: 지도는 Leaflet으로 시작하고, 주문 윤곽만 기본 렌더링한 뒤 선택 pass 또는 strip의 상세 레이어를 추가하는 방식으로 정했다. full 규모에서 모든 strip·footprint를 동시에 SVG로 만들지 않아 기본 탐색 비용을 제한한다. 결과 선택은 `captureId` URL query로 지도와 24시간 타임라인이 공유하며, reward/action mask의 완전한 설명은 schedule API가 아닌 episode step 로그의 책임으로 유지한다.
- 2026-07-20: 대시보드의 run 목록은 SQLite metadata만 읽고 artifact summary/replay는 선택한 결과 상세에서만 검증한다. 이 경계는 목록을 빠르게 유지하고 손상된 한 artifact가 최근 실행 목록 전체를 사용할 수 없게 하는 것을 막는다.
- 2026-07-20: 시나리오 상세 탐색은 전체 Scenario로 상단 설정·개수를 한 번 읽고, 주문·strip·촬영 기회는 전용 pagination API로 별도 요청하도록 구성했다. 이는 대용량 목록 렌더링을 개요 조회와 분리하며, URL query에 선택·필터 상태를 보존해 다음 지도 및 재생 화면에서도 같은 탐색 위치를 재사용할 수 있게 한다.
- 2026-07-20: 읽기 전용 Frontend의 첫 단위에서는 BrowserRouter와 공통 API client를 분리했다. UI는 HTTP 상태만으로 오류를 표시하지 않고 Backend가 제공한 구조화된 오류 code/message를 함께 사용하며, 개발 환경의 `/api` proxy와 배포 환경의 `VITE_API_BASE_URL`을 분리해 화면 코드에 서버 주소를 고정하지 않는다.
- 2026-07-20: 단계 10 Frontend는 시나리오·주문·strip·opportunity와 검증 결과를 읽기 전용으로 탐색한다. 현재 변경 API 없이 UI만 먼저 열면 실행 이력과 파생 artifact의 정합성을 깨뜨릴 수 있으므로, 수정 기능은 scenario version 또는 복제본, 검증, 파생 데이터 재생성, run snapshot 정책을 함께 확정한 뒤 추가한다.
- 2026-07-20: 평가 replay는 현재 실행마다 하나이지만 Backend에서는 예약된 `evaluation` episode ID로 노출한다. 이 방식은 현재의 단일 replay 파일과 무결성 검증 경계를 유지하면서도, 후속 다중 평가 episode 확장 시 URL 계약을 바꾸지 않게 한다. 상세 step은 후보와 action mask 사유를 포함하므로 타임라인과 분리해 pagination한다.

- 2026-07-08: 기준 정책과 Maskable PPO 평가가 공유하는 `EpisodeReplay` 로그 계약과 `PolicyComparison` 비교 artifact를 기록했다.
- 2026-07-13: CP-SAT을 후처리가 아닌 축소 시나리오용 최적화 기준해 baseline으로 정의하고, simulator replay로 재검증해야 한다는 설계 근거를 기록했다.
- 2026-07-13: OR-Tools CP-SAT 기반 tiny baseline 구현과 simulator replay 검증 결과를 기록했다.
- 2026-07-19: CP-SAT의 해 없음 artifact 경로와 RL·휴리스틱·CP-SAT 통합 비교를 검증하고, small 시나리오에서 최적해와 replay 정합성을 확인했다.
- 2026-07-20: 결과 조회는 재계산 대신 artifact 색인의 소유자·종류·SHA-256과 Pydantic 계약, 실행 metadata 일치를 함께 검증해야 한다는 경계를 기록했다. 이 방식은 유효한 JSON으로 외부 변경된 평가 결과나 다른 실행의 artifact 참조를 화면에 노출하지 않는다.
- 2026-07-20: 장시간 PPO 학습은 HTTP 요청에서 직접 실행하지 않고, 저장된 run·scenario·config snapshot을 다시 읽는 별도 worker process로 분리했다. 초기 단일 worker 제한은 GPU/CPU 경합과 artifact 혼합을 피하는 범위 제약이며, Backend 재시작과 worker 장애 복구를 같은 사건으로 취급하지 않는다.
- 2026-07-20: 학습 중지는 process 강제 종료 대신 PPO callback의 step 경계에서 상태를 확인하는 cooperative cancellation으로 처리한다. 이 경계에서 checkpoint를 보존하면 모델 파일 쓰기나 rollout 처리 중단으로 인한 손상 위험을 줄일 수 있으며, 중지된 model은 최종 평가 결과로 취급하지 않는다.
- 2026-07-20: 실행 중인 학습 metrics는 계속 append되므로 고정 artifact의 SHA-256 방식으로 검증하지 않는다. 대신 각 JSONL 행의 구조를 검증하고, active run의 마지막 미완성 행만 polling 시 일시적으로 제외한다. 학습 곡선에는 replay를 중복 저장하지 않아 결과 artifact와 로그의 책임을 분리한다.
- 2026-07-08: 단계 6 Maskable PPO 반복 seed 성능 검증 결과와 tiny 시나리오 해석상 주의점을 기록했다.
- 2026-07-07: Maskable PPO 학습 산출물 구조와 학습/평가 seed 분리의 의미를 기록했다.
- 2026-07-07: strip과 footprint를 pass 진행 방향에 맞춘 polygon으로 바꾸고 pitch 0 해석을 기록했다.
- 2026-07-07: 가상 ground track, footprint, access window 생성기와 opportunity 근거 추적 구현 결과를 기록했다.
- 2026-07-07: opportunity 생성의 공간적 근거가 부족한 설계 허점을 확인하고 가상 ground track/footprint 생성기 필요성을 기록했다.
- 2026-07-07: Gymnasium Dict 관측, padding/presence와 action mask의 책임 경계를 기록했다.
- 2026-07-07: 기준 정책 구현과 시각 양자화를 통해 학습 환경에 경쟁 가능한 행동이 필요함을 기록했다.
- 2026-07-06: 단계 0~3 구현에서 확정된 개발 기반과 full 시나리오 스모크 실행 관찰을 기록했다.
- 2026-07-06: 구조적 데이터 오류와 state 의존적 action mask의 검증 책임 경계를 명확히 했다.
- 2026-07-06: 프로젝트 지식 베이스를 생성하고 현재까지 합의한 핵심 도메인 및 설계 지식을 정리했다.
