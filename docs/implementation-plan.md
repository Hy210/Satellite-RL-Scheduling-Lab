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

### 단계 8. 저장 계층

#### 작업

- [ ] SQLite schema
- [ ] 시나리오 메타데이터 저장
- [ ] 학습 및 평가 run 저장
- [ ] 환경/보상 설정 snapshot
- [ ] 모델 및 artifact 경로 저장
- [ ] 로컬 artifact 디렉터리 관리
- [ ] 원자적 파일 저장 또는 임시 파일 교체
- [ ] 실패한 run 상태 복구

#### 완료 조건

- 실행 재시작 후에도 기존 시나리오와 결과를 조회할 수 있다.
- 각 모델이 어떤 시나리오와 설정으로 생성됐는지 추적할 수 있다.
- 삭제 또는 손상된 artifact를 명확한 오류로 표시한다.

---

### 단계 9. FastAPI Backend

#### 구현 순서

1. [ ] health 및 버전 정보
2. [ ] 시나리오 목록과 상세 조회
3. [ ] 주문, strip 및 opportunity 조회
4. [ ] 시나리오 유효성 검사
5. [ ] 기준 정책 실행
6. [ ] 결과와 타임라인 조회
7. [ ] 학습 run 생성
8. [ ] 학습 중지 요청
9. [ ] 학습 상태와 로그 조회
10. [ ] episode 및 step 조회

#### 완료 조건

- RL core가 Backend와 독립적으로 계속 테스트된다.
- API 입력 오류가 구조화된 오류 응답으로 반환된다.
- 학습 worker 장애가 FastAPI 서버를 종료시키지 않는다.
- 연결이 끊겨도 실행 중인 학습이 계속된다.

---

### 단계 10. React Frontend 기본 화면

#### 구현 순서

1. [ ] 공통 레이아웃과 라우팅
2. [ ] API client와 오류 처리
3. [ ] 대시보드
4. [ ] 시나리오 목록
5. [ ] 시나리오 상세
6. [ ] 주문과 strip 목록
7. [ ] 촬영 기회 검사
8. [ ] 환경 및 보상 파라미터 수정
9. [ ] 결과 지표 화면

#### 완료 조건

- 준비된 시나리오의 모든 주요 데이터를 웹에서 조회할 수 있다.
- 허용된 주문 속성과 파라미터를 수정하고 검증 결과를 확인할 수 있다.
- 로딩, 빈 데이터 및 오류 상태가 구분되어 표시된다.

---

### 단계 11. 지도와 타임라인

#### 작업

- [ ] 지도 라이브러리 선정
- [ ] 주문 geometry 표시
- [ ] strip 표시
- [ ] 완료/부분 완료/미촬영 상태 표시
- [ ] orbit/pass 및 촬영 결과 표시
- [ ] 24시간 촬영 타임라인
- [ ] 우선순위 색상
- [ ] 자세와 reward 상세 tooltip
- [ ] 지도와 타임라인 선택 연동

#### 완료 조건

- 타임라인의 촬영을 선택하면 지도에서 해당 strip이 강조된다.
- 지도에서 strip을 선택하면 관련 촬영 결과를 조회할 수 있다.
- 대량 strip에서도 기본적인 탐색이 가능하다.

---

### 단계 12. 학습 제어와 실시간 상태

#### 작업

- [ ] 학습 설정 화면
- [ ] 학습 시작
- [ ] 안전한 중지 요청
- [ ] 실행 상태 표시
- [ ] 진행률 및 로그 갱신
- [ ] 학습 곡선
- [ ] checkpoint와 최종 모델 표시
- [ ] 연결 재시도 및 상태 복구

초기에는 REST polling으로 시작할 수 있고 필요하면 WebSocket으로 교체한다.

#### 완료 조건

- 웹에서 학습을 시작하고 중지할 수 있다.
- 웹을 새로 고쳐도 실행 상태가 복구된다.
- 학습 중에도 다른 조회 API와 GUI가 응답한다.

---

### 단계 13. Episode 재생 및 정책 비교

#### 작업

- [ ] 처음/이전/재생/정지/다음/마지막 제어
- [ ] 재생 속도 조절
- [ ] 특정 step 이동
- [ ] 현재 state와 action 후보 표시
- [ ] action mask와 사유 표시
- [ ] 선택 action 및 reward breakdown 표시
- [ ] 지도와 타임라인 동기화
- [ ] RL 및 기준 정책 지표 비교
- [ ] 서로 다른 run 비교

#### 완료 조건

- 저장된 평가 episode를 step 단위로 끝까지 재생할 수 있다.
- 특정 정책의 선택이 가능했는지와 선택하지 않은 이유를 확인할 수 있다.
- 동일 시나리오의 RL과 기준 정책 결과를 한 화면에서 비교할 수 있다.

---

### 단계 14. 통합 검증과 문서 정리

#### 통합 검증

- [ ] 동일 seed의 시나리오 재현
- [ ] 동일 정책 평가 결과 재현
- [ ] 모든 촬영의 시간 및 자세 제약 준수
- [ ] return과 reward breakdown 일치
- [ ] 데이터 모델, API 및 GUI 단위 일치
- [ ] 모델과 설정의 추적 가능성
- [ ] 학습 worker 실패 처리
- [ ] 잘못된 시나리오의 학습 차단
- [ ] full 시나리오의 성능과 메모리 확인
- [ ] RL과 모든 기준 정책 비교

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
