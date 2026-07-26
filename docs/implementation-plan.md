# 위성 촬영 스케줄링 프로토타입 구현 계획

## 1. 문서 목적

이 문서는 전체 시스템을 한 번에 구현하면서 발생할 수 있는 누락과 오류를 줄이기 위해 단계별 구현 순서, 산출물, 검증 항목 및 다음 단계 진입 조건을 정의한다.

도메인 규칙은 [rl-scheduling-design.md](rl-scheduling-design.md), 웹 구조는 [web-application-design.md](web-application-design.md)를 기준으로 한다. 서로 충돌하는 구현이 발견되면 코드를 임의로 확장하지 않고 설계 문서를 먼저 검토한다.

## 2. 진행 원칙

- 아래 단계의 순서를 지킨다.
- 각 단계의 테스트와 완료 조건을 충족한 후 다음 단계로 이동한다.
- RL 학습 전에 결정론적 시뮬레이터와 기준 정책을 검증한다.
- Backend 연결 전에 RL core를 독립적으로 실행할 수 있게 한다.
- Frontend는 검증된 API와 결과 데이터만 사용한다.
- 한 단계에서 발견한 설계 변경은 관련 문서에 반영한다.
- 실행 결과에는 시나리오, 설정, seed와 코드 버전을 함께 남긴다.
- 범위 밖 기능은 구현 중 편의상 추가하지 않는다.

## 3. 전체 단계

```text
0. 개발 기반
1. 데이터 계약
2. 가상 시나리오 생성
2A. 가상 ground track 및 footprint 생성기 보완
3. 시뮬레이션 코어
4. 기준 정책
5. Gymnasium 환경
6. RL 학습
7. 결과 및 재생 로그
7A. CP-SAT 최적화 기준해
8. 저장 계층
9. FastAPI Backend
10. React Frontend 기본 화면
11. 지도와 타임라인
12. 학습 제어와 실시간 상태
13. Episode 재생 및 정책 비교
14. 통합 검증과 문서 정리
```

## 4. 단계별 계획

### 단계 0. 개발 기반 구성

#### 작업

- [x] Python 및 Node.js 개발 버전 결정
- [x] Python 패키지 및 가상환경 관리 방식 결정
- [x] Frontend 패키지 관리 방식 결정
- [x] 프로젝트 디렉터리 구성
- [x] 테스트, lint 및 format 도구 구성
- [x] 환경별 설정 파일과 로컬 데이터 경로 정책 정의

권장 최상위 구조는 다음과 같다.

```text
backend/
frontend/
rl_core/
tests/
data/
docs/
```

#### 완료 조건

- [x] Python 및 Frontend의 최소 실행 명령이 동작한다.
- [x] 테스트를 실행할 수 있다.
- [x] 생성 파일과 로컬 학습 산출물의 버전 관리 제외 정책이 정해져 있다.

---

### 단계 1. 데이터 계약 정의

#### 작업

- [x] Scenario 모델
- [x] Satellite 및 자세 파라미터 모델
- [x] Orbit/Pass 모델
- [x] Order 모델
- [x] Strip 모델
- [x] Opportunity 모델
- [x] 환경 및 보상 설정 모델
- [x] Episode/Step 결과 모델
- [x] Training/Evaluation run 모델
- [x] JSON 직렬화 및 역직렬화 형식
- [x] ID, 시각, 각도 및 geometry 단위 정의

모든 시각은 시나리오 시작 기준 초 또는 명시적인 UTC datetime 중 하나로 통일한다. 내부 계산 단위와 API 표현 단위가 다르면 변환 경계를 문서화한다.

#### 필수 검증

- [x] 잘못된 주문 기간 거부
- [x] strip이 없는 주문 거부
- [x] 중복 ID 거부
- [x] 잘못된 초기 위성 자세 거부
- [x] 시뮬레이션 기간 밖 기회 거부
- [x] 촬영시간을 담지 못하는 기회 거부

Opportunity의 자세가 특정 state에서 허용되는지는 데이터 로드 오류로 처리하지 않고 action mask로 판정한다. 구조 검증과 실행 가능성의 경계는 [데이터 형식](data-format.md#6-구조-검증과-action-mask의-경계)을 따른다.

#### 완료 조건

- [x] 하나의 작은 시나리오를 파일로 저장하고 같은 값으로 다시 읽을 수 있다.
- [x] 잘못된 예제 데이터가 예상한 검증 오류를 발생시킨다.

---

### 단계 2. seed 기반 가상 시나리오 생성기

#### 작업

- [x] 전 세계에 가상 주문 geometry 생성
- [x] red/blue/background 우선순위 배정
- [x] 주문별 동일 요구 기간 생성
- [x] 사전 분할된 직사각형 strip 생성
- [x] 30개 orbit/pass 생성
- [x] strip별 접근 가능 구간 생성
- [x] 초반/최소각/후반 opportunity 생성
- [x] 중복 opportunity 병합
- [x] 자세 및 주문 기간 제약에 맞지 않는 후보 제거
- [x] 동일 seed 재현성 보장

현재 완료된 생성기는 pass 시간 구간과 무작위 접근 구간으로 opportunity를 생성한다. 이 방식은 RL 환경 검증에는 충분했지만, strip이 궤도와 footprint 관점에서 왜 유효한지 지도에서 확인할 수 있는 근거 데이터가 부족하다. 실제 궤도 데이터가 들어오기 전까지는 아래 단계 2A를 통해 가상 ground track과 footprint 생성기를 보완한다.

#### 단계적 시나리오 크기

```text
tiny   : 주문 5개
small  : 주문 20개
full   : 주문 100개, pass 30개
```

#### 완료 조건

- [x] 같은 seed에서 byte 단위 또는 의미적으로 동일한 시나리오가 생성된다.
- [x] 다른 seed에서는 다른 주문과 기회가 생성된다.
- [x] 모든 opportunity가 유효한 order, strip 및 pass를 참조한다.
- [x] 최대 strip 2,000개 제한을 검사한다.

---

### 단계 2A. 가상 ground track 및 footprint 생성기 보완

이 단계는 실제 궤도 전파 데이터가 없는 현재 상태에서 opportunity의 공간적 근거를 만들기 위한 보완 단계다. 정밀 궤도 물리를 구현하는 것이 아니라, pass별 가상 궤적과 센서 footprint를 만들고 strip과의 교차로 access window를 생성해 지도에서 검수 가능하게 한다.

#### 작업

- [x] pass별 시간 샘플 간격 결정
- [x] seed 기반 가상 ground track 생성
- [x] 촬영 가능 폭 또는 footprint 근사 파라미터 정의
- [x] 시간 샘플별 회전 footprint polygon 생성
- [x] pass 진행 방향에 맞춘 strip polygon 생성
- [x] footprint와 strip polygon 교차 판정
- [x] 연속 교차 샘플을 access window로 병합
- [x] access window에서 early/min_off_nadir/late opportunity 생성
- [x] opportunity가 참조 access window에서 파생됐음을 추적
- [x] pass, ground track, footprint, strip 및 opportunity를 지도에서 확인하는 임시 HTML 또는 개발용 뷰어 작성
- [x] 실제 궤도 데이터 연결 시 교체할 입력 경계 문서화

#### 완료 조건

- [x] 같은 seed에서 ground track, footprint, access window 및 opportunity가 재현된다.
- [x] 모든 opportunity가 footprint-strip 교차 근거를 가진다.
- [x] 지도에서 선택한 pass의 ground track, footprint, strip 및 opportunity를 함께 확인할 수 있다.
- [x] 공간적으로 교차하지 않는 strip에는 해당 pass의 opportunity가 생성되지 않는다.
- [x] 기존 simulator, 기준 정책 및 Gym 환경이 새 opportunity 입력을 그대로 소비할 수 있다.

임시 지도 뷰어는 `tools/scenario_viewer.html`로 추가했다. 브라우저에서 지도 타일 렌더링, pass별 ground track, 회전 footprint polygon, pass 진행 방향에 맞춘 strip polygon 및 opportunity 표시를 확인했다.

---

### 단계 3. 결정론적 시뮬레이션 코어

이 단계에서는 Gymnasium과 RL 라이브러리를 사용하지 않는다.

#### 작업

- [x] 시뮬레이션 clock
- [x] opportunity 시간순 이벤트 처리
- [x] 주문 및 strip 진행 상태
- [x] 자세 전환시간 계산
- [x] 최소 촬영 간격 계산
- [x] 촬영 가능 여부 판정
- [x] action 후보 최대 128개 구성
- [x] action mask와 마스킹 사유 생성
- [x] 촬영 실행 및 자세 갱신
- [x] skip 및 다음 이벤트 이동
- [x] 주문 마감과 부분 미완료 패널티
- [x] 에피소드 종료 판정
- [x] 보상 요소별 계산

#### 필수 단위 테스트

- [x] 같은 자세의 전환시간은 0초다.
- [x] roll과 tilt 전환시간은 순차 합산된다.
- [x] 안정화 시간은 자세가 바뀔 때 한 번 적용된다.
- [x] 최소 촬영 간격 5초가 적용된다.
- [x] roll/tilt `-30~30 deg` 범위를 검사한다.
- [x] 결합 off-nadir `30 deg` 한계를 검사한다.
- [x] 이미 완료한 strip은 다시 촬영할 수 없다.
- [x] strip 완료 시 관련 opportunity가 모두 차단된다.
- [x] 주문 마감 후 남은 opportunity가 만료된다.
- [x] 부분 완료율에 따라 패널티가 감소한다.
- [x] 동시에 겹치는 촬영 중 하나만 실행된다.
- [x] skip은 보상 0으로 다음 이벤트에 이동한다.
- [x] 24시간, 전체 완료 또는 기회 소진 시 종료한다.
- [x] 동일 입력 action 열은 같은 결과를 생성한다.

#### 완료 조건

- [x] 결정론적인 유효 action 열로 tiny 및 full 시나리오를 끝까지 실행할 수 있다.
- [x] 모든 실행 촬영이 시간 및 자세 제약을 만족한다.
- [x] reward 합계와 기록된 return이 일치한다.
- [x] 필수 단위 테스트를 모두 통과한다.

---

### 단계 4. 기준 정책

#### 작업

- [x] Random valid
- [x] Earliest deadline first
- [x] Priority greedy
- [x] Priority-efficiency greedy
- [x] 정책 공통 인터페이스
- [x] 정책별 결정 로그
- [x] 동일 시나리오 반복 평가

#### 완료 조건

- [x] 모든 정책이 마스킹된 action을 선택하지 않는다.
- [x] 각 정책이 tiny, small, full 시나리오를 종료할 수 있다.
- [x] 같은 정책과 seed의 결과가 재현된다.
- [x] 정책별 총 return과 평가 지표를 비교할 수 있다.

---

### 단계 5. Gymnasium 환경

#### 작업

- [x] `reset()` 구현
- [x] `step(action)` 구현
- [x] 고정 observation space
- [x] `Discrete(129)` action space
- [x] action mask 인터페이스
- [x] 최대 2,000개 strip padding
- [x] 최대 128개 후보 padding
- [x] 시간, 자세, 우선순위 및 개수 정규화
- [x] seed 처리
- [x] episode info와 reward breakdown

#### 필수 검증

- [x] Gymnasium 환경 검사 통과
- [x] 관측값 shape와 dtype 고정
- [x] 관측값에 NaN/Infinity 없음
- [x] 최소한 skip action은 항상 유효
- [x] padding 데이터는 정책 입력에서 구분 가능
- [x] 같은 seed와 action 열로 동일 trajectory 생성
- [x] 시뮬레이션 코어 직접 실행 결과와 Gym 환경 결과 일치

#### 완료 조건

- [x] 유효 action으로 여러 규모의 episode를 오류 없이 실행한다.
- [x] 환경 wrapper가 도메인 보상이나 제약을 중복 계산하지 않는다.

---

### 단계 6. Maskable PPO 학습

#### 작업

- [x] Maskable PPO 연결
- [x] 학습 설정 모델
- [x] checkpoint 저장
- [x] 학습 및 평가 seed 분리
- [x] 주기적 고정 시나리오 평가
- [x] 학습 metric 기록
- [x] 모델 저장과 다시 불러오기
- [x] 반복 seed 기반 Random valid 대비 성능 검증 CLI

#### 확장 순서

1. tiny 시나리오에서 동작 확인
2. Random valid보다 높은 성능 확인
3. small 시나리오로 확장
4. 주요 greedy 정책과 비교
5. full 시나리오로 확장
6. 필요할 때만 하이퍼파라미터 조정

#### 점검 사항

- 학습 return과 평가 return을 구분한다.
- 고정 시나리오 과적합이 현재 목표임을 결과에 명시한다.
- 보상 총합만 보지 않고 완료율과 우선순위 점수를 함께 본다.
- 정책이 부분 촬영만 분산하는지 확인한다.
- skip만 반복하거나 특정 후보만 고르는 정책 붕괴를 확인한다.

#### 완료 조건

- [x] 저장한 모델을 다시 불러와 동일한 방식으로 평가할 수 있다.
- [x] tiny 시나리오에서 Random valid보다 일관되게 우수하다.
- [x] 평가 결과에 reward breakdown과 도메인 지표가 포함된다.

`tools/stage6_benchmark.py`는 `synthetic-tiny-20260707`에서 Maskable PPO 5개 학습 seed와 Random valid 5개 평가 seed를 비교한다. 2026-07-08 엄격 검증 결과 `stage6_passed=true`였으며, PPO median return은 `5.326453530248241`, Random valid median return은 `5.325396139404043`이었다. 두 정책 모두 median completed strips는 `9`였고, skip 비율과 non-skip action slot 집중도 기준도 통과했다.

이 검증으로 단계 6은 완료로 본다. 다만 tiny 시나리오는 작아서 개선 폭이 매우 작으므로, small/full 확장과 greedy 정책 비교는 이후 단계의 성능 분석 또는 튜닝 작업으로 다룬다.

---

### 단계 7. 결과 및 재생 로그

#### 작업

- [x] episode 요약 형식
- [x] step별 state 요약
- [x] action 후보 및 선택 action
- [x] action mask와 마스킹 사유
- [x] reward breakdown
- [x] 촬영 전후 시간과 자세
- [x] 주문 및 strip 상태 변화
- [x] 최종 촬영 스케줄
- [x] 정책별 비교 결과

#### 로그 크기 관리

- 모든 학습 episode의 전체 step을 무조건 저장하지 않는다.
- 평가 episode, 최고 성능 episode 및 사용자가 지정한 episode를 우선 저장한다.
- 대용량 step 데이터는 파일로 저장하고 메타데이터만 데이터베이스에 둔다.

#### 완료 조건

- [x] 저장된 로그만으로 episode를 처음부터 끝까지 재생할 수 있다.
- [x] step별 reward의 합이 episode return과 일치한다.
- [x] 선택 당시의 마스킹 사유를 사후 확인할 수 있다.

`EpisodeReplay` 공통 계약을 추가해 기준 정책과 Maskable PPO 평가가 같은 재생 로그를 생성한다. PPO 학습 산출물에는 `metrics/replay.json`을 별도로 저장하고, 기준 정책 replay도 같은 저장/복원 helper를 사용한다.

`PolicyComparison` 공통 계약을 추가해 같은 시나리오의 여러 정책 replay와 성능 지표를 하나의 비교 artifact로 묶는다. 이로써 단계 7은 완료로 본다. 비교 artifact의 영구 저장 위치와 데이터베이스 메타데이터 연결은 단계 8 저장 계층에서 확정한다.

---

### 단계 7A. CP-SAT 최적화 기준해

이 단계는 Maskable PPO를 후처리하지 않고, 축소 시나리오에서 RL 정책과 기준 정책의 성능을 더 정밀하게 비교하기 위한 최적화 solver baseline을 추가한다. 첫 구현 대상은 `tiny` 시나리오이며, 모델 정합성과 실행 시간을 확인한 뒤 `small`로 확장한다.

#### 작업

- [x] OR-Tools CP-SAT 의존성 검토 및 추가
- [x] opportunity 선택 0/1 변수 모델링
- [x] 같은 strip 중복 선택 금지 제약
- [x] 시간 겹침 및 최소 촬영 간격 충돌 제약
- [x] simulator와 동일한 자세 전환시간 기반 충돌 제약
- [x] 주문 기간, access window, episode 범위 및 자세 제한 위반 후보 제외
- [x] reward와 유사한 선형 목적함수 정의
- [x] solver 선택 결과를 시간순 action 열로 변환
- [x] 선택 결과를 기존 simulator 평가기로 재검증하고 `EpisodeReplay` 생성
- [x] `OptimizationBaselineResult` 저장 및 복원 helper
- [x] `PolicyComparison`에 `cp_sat_baseline` 항목 포함
- [x] 제한 시간, solver status, best objective, bound 및 gap 기록

#### 필수 검증

- [x] tiny 시나리오에서 CP-SAT 결과가 모든 simulator 제약을 만족한다.
- [x] CP-SAT이 선택한 opportunity ID 목록을 replay로 다시 실행할 수 있다.
- [x] solver 목적함수와 simulator return 차이를 reward breakdown으로 설명할 수 있다.
- [x] infeasible 또는 time limit 상태가 명확한 artifact 상태로 저장된다.
- [x] Random valid, greedy, Maskable PPO 및 CP-SAT baseline을 같은 `PolicyComparison`으로 비교할 수 있다.

#### 완료 조건

- [x] tiny 시나리오에서 CP-SAT baseline 결과와 optimality gap을 산출할 수 있다.
- [x] CP-SAT 결과가 기존 평가·재생 로그 구조와 호환된다.
- [x] solver 모델이 simulator 제약과 어긋날 경우 테스트가 실패한다.

`rl_core/optimization.py`는 `solve_cp_sat_baseline()`으로 tiny 시나리오의 CP-SAT 기준해를 만든다. 현재 구현은 `strip_base_reward + angle_bonus` 합을 정수 계수로 스케일링해 최대화하고, 선택 결과는 `cp_sat_baseline` 정책 이름의 `EpisodeReplay`로 simulator에서 재검증한다. `INFEASIBLE` 및 `UNKNOWN`(time limit 등 해를 찾기 전 종료) 상태도 선택 목록과 solver 점수를 비운 artifact로 저장·복원하는 테스트를 추가했다. 또한 Random valid, 세 greedy 정책, 짧은 학습 smoke run의 Maskable PPO 및 CP-SAT baseline을 하나의 `PolicyComparison`으로 저장·복원하는 통합 테스트를 통과했다. small 시나리오(seed 20260707, solver seed 17, 60초 제한)는 0.05초에 `OPTIMAL`, gap 0으로 replay 검증까지 완료했다. 이로써 단계 7A는 완료로 본다.

---

### 단계 8. 저장 계층

#### 작업

- [x] SQLite schema
- [x] 시나리오 메타데이터 저장
- [x] 학습 및 평가 run 저장
- [x] 환경/보상 설정 snapshot
- [x] 모델 및 artifact 경로 저장
- [x] 로컬 artifact 디렉터리 관리
- [x] 원자적 파일 저장 또는 임시 파일 교체
- [x] 실패한 run 상태 복구

#### 완료 조건

- 실행 재시작 후에도 기존 시나리오와 결과를 조회할 수 있다.
- 각 모델이 어떤 시나리오와 설정으로 생성됐는지 추적할 수 있다.
- 삭제 또는 손상된 artifact를 명확한 오류로 표시한다.

`rl_core/storage.py`의 `StorageRepository`는 `data/scheduler.sqlite3`와 data root 아래 파일을 함께 관리한다. SQLite에는 scenario/run 메타데이터와 artifact의 상대 경로·SHA-256·크기만 저장하고, Scenario·설정·replay·비교 결과 같은 대용량 JSON과 모델 바이너리는 파일로 유지한다. JSON은 같은 디렉터리의 임시 파일을 fsync한 뒤 원자 교체하며, 이미 생성된 모델 파일은 `register_existing_artifact()`로 색인할 수 있다. 누락 artifact는 `ArtifactNotFoundError`와 `list_missing_artifacts()`로 복구 대상으로 확인한다.

실행 상태 전이는 `queued → running → completed/failed` 및 `running → stop_requested → stopped/failed`만 허용하며 terminal 상태는 다시 실행 상태로 바꾸지 않는다. worker supervisor는 기존 worker가 없음을 확인한 자신의 시작 시점에만 `recover_interrupted_runs()`를 호출한다. 이 호출은 `running`을 `failed`로, `stop_requested`를 `stopped`로 정리하되 checkpoint와 부분 artifact는 삭제하지 않는다. Backend 재시작만으로 이 복구를 호출하지 않아 살아 있는 별도 worker를 실패 처리하지 않도록 한다. `train_maskable_ppo()`는 선택적으로 `StorageRepository`를 받아 시작·완료·예외 상태를 SQLite에 기록한다. 이로써 단계 8은 완료로 본다.

---

### 단계 9. FastAPI Backend

#### 구현 순서

1. [x] health 및 버전 정보
2. [x] 시나리오 목록과 상세 조회
3. [x] 주문, strip 및 opportunity 조회
4. [x] 시나리오 유효성 검사
5. [x] 기준 정책 실행
6. [x] 결과와 타임라인 조회
7. [x] 학습 run 생성
8. [x] 학습 중지 요청
9. [x] 학습 상태와 로그 조회
10. [x] episode 및 step 조회

`backend.app.create_app()`은 테스트에서 임시 `StorageRepository`를 주입할 수 있는 app factory다. 현재 구현된 조회 API는 `GET /api/health`, `GET /api/version`, `GET /api/scenarios`, `GET /api/scenarios/{scenario_id}`와 하위 목록인 `GET /api/scenarios/{scenario_id}/orders`, `/strips`, `/opportunities`다. 목록은 대용량 Scenario JSON을 읽지 않고 SQLite 메타데이터만 반환하며, 하위 목록은 Scenario를 검증해 읽은 뒤 공통 `offset`/`limit` pagination(기본 0/100, limit 최대 500)을 적용한다. orders는 priority, strips는 order ID, opportunities는 order/strip/pass/kind 필터를 지원한다. `GET /api/scenarios/{scenario_id}/validation`은 artifact 존재 여부, SQLite SHA-256 일치 및 Pydantic 구조를 다시 검사해 `valid`와 `issues[]`를 반환한다. 구조 오류와 checksum 불일치는 200 응답의 검증 결과로 표시하고, 존재하지 않는 시나오는 `404 scenario_not_found`, 삭제된 artifact는 `409 scenario_artifact_missing`의 구조화된 오류 본문으로 반환한다.

`POST /api/evaluation-runs`는 `random_valid`, `earliest_deadline_first`, `priority_greedy`, `priority_efficiency_greedy`를 동기 실행한다. API는 `EvaluationRun`을 먼저 `running`으로 저장하고, 성공하면 `completed` 상태·summary/replay artifact를 `data/evaluations/<run-id>/`에 저장한다. 정책 실행 예외는 `failed` 상태와 내부 오류 메시지를 저장하고 API에는 `500 baseline_evaluation_failed`만 반환한다. `GET /api/evaluation-runs/{run_id}`는 worker 확장 전에도 polling 가능한 run 상태를 반환한다. `GET /api/results/{run_id}`는 저장된 summary를, `GET /api/results/{run_id}/timeline`은 저장된 replay의 schedule만 시간순 pagination으로 반환한다. `GET /api/results/{run_id}/episodes`는 현재 단일 replay를 예약된 `evaluation` episode ID의 요약으로, `GET /api/results/{run_id}/episodes/evaluation/steps`는 원본 `ReplayStep`을 pagination으로 반환한다. 따라서 후보, action mask 사유, 선택 action 및 reward breakdown을 재계산 없이 재생 화면에 제공한다. 결과 조회는 artifact 색인의 소유자·종류·SHA-256과 Pydantic 구조, run metadata 일치를 검증하며, 미완료 run·artifact 누락·손상은 각각 구조화된 409 응답으로 구분한다. 알 수 없는 episode ID는 `404 episode_not_found`이다.

`POST /api/training-runs`는 `202 Accepted`로 queued `TrainingRun`을 반환하고, `backend.workers.TrainingWorkerSupervisor`가 별도 non-daemon spawn process에서 저장된 scenario·config snapshot을 다시 읽어 `train_maskable_ppo()`를 실행한다. 요청의 `artifact_root` 값은 무시하고 서버의 `data/runs` 경로로 강제한다. 초기 구현은 로컬 단일 worker만 허용하며 실행 중인 worker가 있으면 새 run을 `failed`로 기록하고 `409 training_worker_busy`를 반환한다. process 시작 자체가 실패하면 `500 training_worker_start_failed`와 failed run을 남긴다. worker 내부 예외도 queued/running run의 `failed` 상태와 오류 메시지로 보존한다. 학습 중지, 상태·로그 조회와 CP-SAT baseline은 아직 구현하지 않았다.

`POST /api/training-runs/{run_id}/stop`은 queued run을 즉시 `stopped`로, running run을 `stop_requested`로 바꾸며 같은 요청을 다시 보내도 상태를 유지한다. PPO callback은 각 training step의 안전한 경계에서 SQLite의 `stop_requested`를 확인하고 즉시 checkpoint를 저장한 뒤 `model.learn()`을 끝낸다. trainer는 `running → stop_requested → stopped` 전이를 보장하고, 중지된 run에는 final model·최종 평가·replay를 만들지 않는다. worker가 시작 전 `stopped` 또는 `stop_requested`를 발견하면 trainer를 실행하지 않는다. 학습 상태·로그 조회와 CP-SAT baseline은 아직 구현하지 않았다.

`GET /api/training-runs/{run_id}`는 SQLite `TrainingRun` 상태를, `GET /api/training-runs/{run_id}/metrics`는 `runs/<run-id>/metrics/training-metrics.jsonl`의 고정 시나리오 평가 요약을 pagination으로 반환한다. metrics JSONL은 학습 곡선용 summary만 기록하며 episode replay를 중복 저장하지 않는다. 파일이 아직 없는 것은 정상 빈 목록이고, 실행 중 마지막 줄에 줄바꿈이 없으면 append 도중인 행으로 보고 다음 polling까지 보류한다. 그 외 JSON·UTF-8·DTO 구조 오류는 `409 training_metrics_invalid`로 구분한다. CP-SAT baseline은 아직 구현하지 않았다.

#### 완료 조건

- RL core가 Backend와 독립적으로 계속 테스트된다.
- API 입력 오류가 구조화된 오류 응답으로 반환된다.
- 학습 worker 장애가 FastAPI 서버를 종료시키지 않는다.
- 연결이 끊겨도 실행 중인 학습이 계속된다.

---

### 단계 10. React Frontend 기본 화면

#### 구현 순서

1. [x] 공통 레이아웃과 라우팅
2. [x] API client와 오류 처리
3. [x] 대시보드
4. [x] 시나리오 목록
5. [x] 시나리오 상세
6. [x] 주문과 strip 목록
7. [x] 촬영 기회 검사
8. [x] 읽기 전용 환경 및 보상 파라미터 표시
9. [x] 결과 지표 화면

#### 완료 조건

- 준비된 시나리오의 모든 주요 데이터를 웹에서 조회할 수 있다.
- 환경 및 보상 파라미터와 시나리오 구조 검증 결과를 읽기 전용으로 확인할 수 있다.
- 로딩, 빈 데이터 및 오류 상태가 구분되어 표시된다.

현재 단계 10은 검증된 Backend 조회 API를 사용하는 읽기 전용 탐색 화면으로 한정한다. 시나리오 생성·복제·삭제와 주문 속성, 환경·보상 파라미터 수정은 후속 범위다. 이 기능을 추가할 때에는 기존 시나리오와 실행 이력의 재현성을 보존하도록 새 scenario version 또는 명시적 복제본을 만들고, 변경 요청 검증·저장·artifact 재생성·실행 snapshot 정책을 함께 확정한다. 따라서 읽기 전용 화면의 라우팅, 선택 상태, API client 및 표 컴포넌트는 이후 mutation UI를 붙일 수 있게 구성하되 현재는 변경 요청을 보내지 않는다.

첫 구현 단위에서는 `react-router-dom`으로 `/scenarios`와 `/scenarios/:scenarioId` 경로를 만들고, 공통 `getJson()` client가 Backend의 `{ error: { code, message } }` 응답을 `ApiError`로 변환한다. Vite 개발 서버는 `/api`를 FastAPI 기본 주소 `http://127.0.0.1:8000`으로 proxy하며, 배포 환경에서는 `VITE_API_BASE_URL`로 기본 경로를 바꿀 수 있다. 시나리오 목록은 `GET /api/scenarios`를 사용하고 로딩·빈 목록·오류·재시도 상태를 제공한다. 상세 경로는 다음 단위의 탭 구현을 위한 진입점만 마련했다.

상세 화면은 `GET /api/scenarios/{scenario_id}`로 상단 개요와 읽기 전용 위성·환경·보상 설정을 로드한다. URL query의 `tab`, `orderId`, `stripId`, `passId`, `priority`, `kind`, `offset`은 탭과 선택·필터 상태를 보존한다. 주문, strip, 촬영 기회, 검증은 각각 전용 API를 독립적으로 요청하므로 한 탭의 오류가 다른 조회 결과를 가리지 않는다. 주문에서 Strip, Strip에서 촬영 기회로 이동할 때 관련 filter를 자동 적용하며, 모든 목록은 Backend pagination을 그대로 사용한다.

대시보드가 실행 ID를 사전에 알 필요 없도록 `GET /api/training-runs`와 `GET /api/evaluation-runs` 목록 API를 추가했다. 두 API는 `scenario_id`, `status`, `offset`, `limit` 필터와 최신 `updated_at` 순서를 사용하고 SQLite run metadata만 반환한다. 대시보드는 최근 시나리오, 학습 run, 평가 run을 표시하며 artifact summary/replay를 목록에서 열지 않는다. 결과 지표 상세 화면은 완료된 평가 run을 선택한 뒤 기존 `GET /api/results/{run_id}`를 호출하는 다음 작업으로 남긴다.

`/results`는 완료된 평가 run 목록을, `/results/{run_id}`는 검증된 summary의 total return, priority score, captures, 완료 order/strip, 평균 off-nadir 및 reward breakdown을 읽기 전용으로 표시한다. 결과 API의 미완료·실패·artifact 누락·무결성 오류는 각각의 구조화된 error code에 맞는 안내로 표시한다. 타임라인과 episode 재생 UI는 후속 단계에서 연결한다.

---

### 단계 11. 지도와 타임라인

#### 작업

- [x] 지도 라이브러리 선정 (Leaflet)
- [x] 주문 geometry 표시
- [x] strip 표시
- [x] 완료/부분 완료/미촬영 상태 표시
- [x] orbit/pass 및 촬영 결과 표시
- [x] 24시간 촬영 타임라인
- [x] 우선순위 색상
- [x] 자세와 reward 상세 panel (episode step 상세 연동)
- [x] 지도와 타임라인 선택 연동

#### 완료 조건

- 타임라인의 촬영을 선택하면 지도에서 해당 strip이 강조된다.
- 지도에서 strip을 선택하면 관련 촬영 결과를 조회할 수 있다.
- 대량 strip에서도 기본적인 탐색이 가능하다.

2026-07-22 첫 구현에서는 `leaflet`과 `@types/leaflet`을 추가했다. 시나리오 지도는 주문 윤곽을 기본 레이어로 유지하고, 선택한 pass의 ground track·footprint·접근 가능한 strip 또는 선택 strip만 상세 SVG 레이어로 그린다. 따라서 full 시나리오에서 모든 strip과 footprint를 항상 렌더링하지 않는다. 결과 상세는 검증된 `GET /api/results/{run_id}/timeline` schedule을 24시간 목록으로 표시하고 URL query `captureId`로 선택 상태를 보존한다. schedule에 있는 strip은 완료, 없는 strip은 미촬영으로 표시하고 주문은 전체 strip 완료 수에 따라 완료·부분 완료·미촬영으로 집계한다. 선택 capture의 `pass_id`·`strip_id`를 지도에 전달하며, 지도 strip 선택은 해당 strip의 첫 capture로 다시 이동한다. `GET /api/results/{run_id}/episodes/evaluation/steps/{step_index}`은 선택 capture의 자세 전후 상태, reward breakdown 및 후보 mask 근거를 재계산 없이 상세 패널에 제공한다.

---

### 단계 12. 학습 제어와 실시간 상태

#### 작업

- [x] 학습 설정 화면
- [x] 학습 시작
- [x] 안전한 중지 요청
- [x] 실행 상태 표시
- [ ] 진행률 및 로그 갱신 (metrics polling만 구현, 텍스트 로그는 미제공)
- [x] 학습 곡선
- [x] checkpoint와 최종 모델 표시
- [x] 연결 재시도 및 상태 복구

초기에는 REST polling으로 시작할 수 있고 필요하면 WebSocket으로 교체한다.

2026-07-22 1차 구현은 `/training` 설정 화면과 `/training/{run_id}` 상태 화면을 추가했다. 화면은 `POST /api/training-runs`로 snapshot을 저장해 worker를 시작하고, active 상태에서 3초마다 run detail과 metrics를 polling한다. 새로고침 뒤에는 URL run ID로 같은 상태를 다시 읽으며, 시작 화면은 단일 local worker가 active이면 새 시작을 비활성화한다. `GET /api/training-runs/{run_id}/detail`은 검증된 config snapshot, checkpoint 파일명 목록, final model/final evaluation 존재 여부를 반환한다. 초기 곡선은 저장된 evaluation return을 표시한다. 텍스트 worker 로그와 상세 PPO loss/value 지표는 현재 artifact 계약에 없으므로 후속 범위다.

#### 완료 조건

- 웹에서 학습을 시작하고 중지할 수 있다.
- 웹을 새로 고쳐도 실행 상태가 복구된다.
- 학습 중에도 다른 조회 API와 GUI가 응답한다.

---

### 단계 13. Episode 재생 및 정책 비교

#### 작업

- [x] 처음/이전/재생/정지/다음/마지막 제어
- [x] 재생 속도 조절
- [x] 특정 step 이동
- [x] 현재 state와 action 후보 표시
- [x] action mask와 사유 표시
- [x] 선택 action 및 reward breakdown 표시
- [x] 지도와 타임라인 동기화
- [x] RL 및 기준 정책 지표 비교
- [x] 서로 다른 완료 EvaluationRun 비교
- [x] PPO 최종 평가와 CP-SAT 기준해를 worker 기반 EvaluationRun으로 색인

2026-07-23 1차 구현은 `/results/{run_id}/replay`에서 단일 `evaluation` episode를 재생한다. episode 요약의 step 수를 범위로 사용하고 `GET /api/results/{run_id}/episodes/evaluation/steps/{step_index}`으로 선택 step만 읽는다. 재생은 저장 artifact를 재계산하지 않으며, 선택 action·state 전후·reward breakdown·후보 및 action mask 사유를 표시한다. 촬영 action이면 대응하는 pass/strip을 지도에 강조한다.

같은 날 추가한 `/comparisons` 화면은 완료된 `EvaluationRun`을 scenario·seed별로 묶고 사용자가 선택한 run 집합으로 `POST /api/policy-comparisons`를 호출한다. 생성된 immutable `PolicyComparison`은 total return, priority score, 완료 strip/order, captures를 표로 보여 주며, 각 행은 artifact에 보존한 `evaluation_run_id`로 정확한 결과·replay 화면으로 이동한다.

PPO 최종 평가와 CP-SAT 기준해는 모두 공통 summary·replay artifact와 completed `EvaluationRun`으로 색인된다. PPO 최종 평가는 학습 run 완료 시 `source_training_run_id`를 채운 `EvaluationRun`으로 저장되고(`rl_core/training.py`), CP-SAT은 `POST /api/cp-sat-evaluation-runs`가 PPO worker와 하나의 로컬 실행 슬롯(`TrainingWorkerSupervisor._lock`/`_process`)을 공유하는 `run_cp_sat_worker`로 실행된다(`backend/workers.py`). `POST /api/training-runs`와 동일하게 worker busy/start 실패를 `409 execution_worker_busy`/`500 execution_worker_start_failed`로 구분한다(`backend/app.py`). 2026-07-26 backend 통합 테스트로 PPO 최종 평가와 CP-SAT 결과가 `GET /api/evaluation-runs`에 함께 나열되고 `POST /api/policy-comparisons`로 같은 비교 artifact에 묶일 수 있음을 확인했고, 실제 uvicorn/Vite dev server를 띄워 동일 흐름을 curl로 재검증했다(`tests/test_backend.py::test_policy_comparison_combines_ppo_final_evaluation_and_cp_sat_baseline`, `tests/test_workers.py::test_supervisor_shares_single_slot_between_ppo_and_cp_sat_workers`).

#### 완료 조건

- 저장된 평가 episode를 step 단위로 끝까지 재생할 수 있다.
- 특정 정책의 선택이 가능했는지와 선택하지 않은 이유를 확인할 수 있다.
- 동일 시나리오의 RL과 기준 정책 결과를 한 화면에서 비교할 수 있다.

---

### 단계 14. 통합 검증과 문서 정리

#### 통합 검증

- [ ] 동일 seed의 시나리오 재현
- [ ] 동일 정책 평가 결과 재현
- [x] 모든 촬영의 시간 및 자세 제약 준수
- [ ] return과 reward breakdown 일치
- [x] 데이터 모델, API 및 GUI 단위 일치
- [x] 모델과 설정의 추적 가능성
- [x] 학습 worker 실패 처리
- [x] 잘못된 시나리오의 학습 차단
- [ ] full 시나리오의 성능과 메모리 확인
- [ ] RL과 모든 기준 정책 비교

2026-07-26 1차 배치는 아래 4개 항목을 완료했다.

- **데이터 모델·API·GUI 단위 일치**: `rl_core/models.py`의 각도(`*_deg`)·경과시간(`*_sec`) 필드가 `backend/app.py` DTO와 Frontend까지 변환 없이 그대로 전달됨을 코드 검토로 확인했다. 유일하게 문서화되지 않았던 `created_at`/`updated_at`(ISO 8601 UTC 문자열) 계약을 `docs/data-format.md`에 명시하고, `tests/test_backend.py`에 파싱 가능성 회귀 테스트를 추가했다.
- **모델과 설정의 추적 가능성**: `EvaluationRun.source_training_run_id` → `GET /api/training-runs/{run_id}/detail`의 config snapshot·checkpoint 목록·`final_model_available`까지 API로 끊김 없이 연결됨을 `tests/test_backend.py::test_evaluation_result_traces_back_to_training_run_and_model_artifacts`로 검증했다. 실제 모델 파일은 별도 다운로드 API 없이 `TrainingRun.artifact_directory`와 저장 규칙 조합으로 로컬에서 찾는다는 점을 `docs/web-application-design.md`에 명시했다.
- **촬영 시간·자세 제약 준수**: `rl_core/simulator.py`가 매 action마다 강제하는 window·roll/tilt·slew·최소 간격 제약을, 저장된 `EpisodeReplay.schedule`만으로 시뮬레이터를 다시 호출하지 않고 독립적으로 재확인하는 `tests/test_integration.py::test_replay_schedule_respects_capture_window_and_minimum_interval`을 추가했다.
- **잘못된 시나리오의 학습 차단**: `_load_scenario_or_api_error`가 SHA-256 불일치를 검사하지 않아 손상된(구조는 유효한) scenario로 학습·평가가 조용히 시작될 수 있던 gap을 발견해 수정했다. `backend/app.py`에 `_load_valid_scenario_or_api_error`를 추가해 `POST /api/training-runs`, `POST /api/evaluation-runs`, `POST /api/cp-sat-evaluation-runs` 세 곳 모두 `repository.validate_scenario()`까지 통과해야 실행을 시작하도록 교체했고(읽기 전용 조회 API는 기존 동작 유지), 세 endpoint 각각의 checksum 불일치 차단 테스트를 추가했다.

나머지 6개 항목(동일 seed·정책 재현, return/reward breakdown 일치, full 규모 성능·메모리, RL·전체 기준 정책 비교)과 문서 7개 항목은 결과 의존적이거나 더 큰 작업 단위라 후속 세션에서 진행한다.

#### 문서

- [ ] 설치 및 개발 환경
- [ ] 로컬 실행 방법
- [ ] 테스트 방법
- [ ] 시나리오 데이터 형식
- [ ] API 계약
- [ ] 학습 및 평가 실행 방법
- [ ] 알려진 제한사항

#### 최종 완료 조건

- 새 환경에서 문서만으로 프로젝트를 실행할 수 있다.
- 웹에서 시나리오 조회, 학습, 평가, 결과 비교 및 episode 재생이 가능하다.
- RL 정책이 최소한 Random valid와 정량적으로 비교된다.
- 미완성 기능과 향후 범위가 명시되어 있다.

## 5. 단계 진입 체크 규칙

각 단계 시작 전 다음을 확인한다.

```text
[ ] 이전 단계의 완료 조건을 모두 충족했는가?
[ ] 실패하거나 비활성화된 테스트가 없는가?
[ ] 새로 발견한 설계 변경을 문서에 반영했는가?
[ ] 현재 단계의 입력 데이터가 고정되었는가?
[ ] 현재 단계에서 만들지 않을 기능이 명확한가?
```

하나라도 충족하지 못하면 다음 단계로 넘어가지 않고 원인을 해결한다.

## 6. 구현 중 변경 관리

구현 중 다음 상황이 발생하면 설계 검토 대상으로 취급한다.

- 보상식 변경
- action 또는 observation 구조 변경
- opportunity 이산화 방식 변경
- 자세 및 시간 제약 변경
- 최대 strip 또는 후보 크기 변경
- 에피소드 종료 조건 변경
- Frontend가 도메인 판정을 직접 수행해야 하는 상황
- 저장된 결과만으로 episode를 재생할 수 없는 상황

변경 시 순서는 다음과 같다.

```text
문제와 근거 기록
-> 관련 설계 문서 수정
-> 테스트 수정 또는 추가
-> 코드 변경
-> 기준 정책과 RL 결과 재검증
```

## 7. 현재 권장 첫 구현 범위

첫 구현 묶음은 단계 0부터 단계 3까지만 포함한다.

```text
개발 기반
-> 데이터 계약
-> tiny 가상 시나리오
-> 결정론적 시뮬레이션 코어
-> 단위 테스트
```

이 묶음이 안정된 뒤 기준 정책과 Gymnasium 환경으로 진행한다. 첫 구현에서 RL, FastAPI 및 React를 동시에 연결하지 않는다.
