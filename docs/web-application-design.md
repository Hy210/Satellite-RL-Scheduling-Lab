# 위성 촬영 스케줄링 웹 애플리케이션 설계

## 1. 문서 목적

이 문서는 강화학습 기반 위성 촬영 스케줄링 프로토타입의 웹 GUI 설계를 기록한다.

웹 애플리케이션의 목적은 다음과 같다.

- 준비된 시나리오와 주문을 조회하고 검증한다. 생성·수정은 후속 단계에서 추가한다.
- 학습을 시작, 중지하고 진행 상태를 확인한다.
- RL 정책과 기준 정책의 결과를 비교한다.
- 하루 동안의 스케줄을 지도와 타임라인으로 확인한다.
- 에피소드와 step을 재생하며 정책의 판단을 분석한다.
- 보상과 환경 파라미터 변경 결과를 반복적으로 피드백한다.

RL 환경의 상세 정의는 [rl-scheduling-design.md](rl-scheduling-design.md)를 기준으로 한다.
단계별 구현 순서와 검증 기준은 [implementation-plan.md](implementation-plan.md)를 따른다.
Backend와 공유할 현재 데이터 계약은 [data-format.md](data-format.md)를 따른다.

## 2. 설계 원칙

- RL 환경과 웹 프레임워크를 분리한다.
- 학습 코어는 웹 없이도 테스트하고 실행할 수 있어야 한다.
- 장시간 학습은 웹 서버와 별도 프로세스에서 실행한다.
- 첫 버전은 로컬 PC에서 한 명이 사용하는 도구로 제한한다.
- 인증, 외부 공개 배포 및 다중 사용자는 초기 범위에서 제외한다.
- GUI에서 변경 가능한 값과 고정된 도메인 규칙을 구분한다.
- 모든 학습 실행은 사용한 시나리오와 파라미터를 함께 저장해 재현할 수 있게 한다.

## 3. 권장 기술 구성

| 영역 | 기술 | 역할 |
|---|---|---|
| Frontend | React + TypeScript | 화면, 편집기, 결과 재생 |
| Backend | FastAPI + Python | REST API, 실시간 상태, 데이터 검증 |
| RL Core | 독립 Python 패키지 | 환경, 정책 학습, 평가, 기준 정책 |
| Chart | Plotly | 학습 곡선, 타임라인, 통계 차트 |
| Map | 초기 구현 시 선정 | 주문, strip, 궤도 및 촬영 결과 표시 |
| Database | SQLite | 시나리오 메타데이터와 실행 기록 |
| File Storage | 로컬 디렉터리 | 모델, 로그, 스케줄, 대용량 기회 데이터 |

지도 라이브러리는 실제 geometry 형식과 필요한 편집 기능을 확인한 뒤 구현 단계에서 확정한다.

## 4. 전체 구조

```text
React Web UI
    |
    | REST API / WebSocket
    v
FastAPI Backend
    |-- Scenario Service
    |-- Training Run Service
    |-- Evaluation Service
    |-- Replay Service
    |-- Result Query Service
    |
    +----------> SQLite
    +----------> Local Artifact Storage
    |
    v
Training Worker Process
    |
    v
RL Core
    |-- Environment
    |-- Reward / Masking
    |-- RL Algorithm
    |-- Baseline Policies
    +-- Metrics / Episode Logger
```

### 4.1 Frontend

Frontend는 사용자 입력과 시각화만 담당한다. 보상 계산, 촬영 가능 판정 및 action masking 같은 도메인 규칙을 중복 구현하지 않는다.

### 4.2 Backend

Backend는 다음 책임을 가진다.

- 입력 데이터 검증
- 시나리오 저장 및 조회
- 학습 프로세스 생성과 제어
- 진행 상태와 로그 전달
- 결과 데이터 조회 및 변환
- 모델과 실행 산출물 관리

### 4.3 Training worker

학습은 FastAPI 요청 처리 프로세스와 분리한다. 웹 서버는 worker를 시작한 뒤 실행 ID를 반환하고, GUI는 해당 ID로 진행 상태를 조회한다.

초기에는 로컬 단일 worker만 지원한다. 분산 작업 큐는 도입하지 않는다.

### 4.4 RL core

RL core는 FastAPI나 React를 참조하지 않는 독립 모듈로 유지한다. CLI, 단위 테스트 및 향후 다른 UI에서도 같은 환경을 사용할 수 있어야 한다.

## 5. 주요 사용자 흐름

### 5.1 시나리오 준비와 탐색

```text
준비된 시나리오 불러오기
-> 주문 영역 확인
-> 사전 생성된 strip 확인
-> 가상 또는 실제 ground track/footprint 확인
-> 사전 계산된 촬영 기회 확인
-> 환경/보상 파라미터 설정
-> 유효성 검사
-> 시나리오 저장
```

첫 버전에서 주문 polygon의 strip 분할과 촬영 기회 계산은 웹 애플리케이션 또는 RL core가 담당하지 않는다. 별도의 전처리기나 가상 시나리오 생성기가 만든 결과를 입력받는다.

실제 궤도 데이터가 제공되기 전까지는 seed 기반 가상 생성기가 ground track, footprint, access window 및 opportunity를 함께 만든다. 웹은 이 데이터를 지도에서 검수하는 역할을 하며, footprint와 strip 교차 판정을 중복 구현하지 않는다.

### 5.2 학습

```text
시나리오 선택
-> 학습 설정 입력
-> 학습 시작
-> 진행 상태와 지표 확인
-> 완료 또는 중지
-> 모델과 결과 저장
```

### 5.3 평가와 비교

```text
학습 모델 선택
-> 동일 시나리오에서 평가
-> 기준 정책 실행
-> 지표와 스케줄 비교
-> 에피소드 재생
```

### 5.4 피드백과 재실행

```text
실패 또는 비효율 구간 확인
-> 보상/환경 설정 복제
-> 값 수정
-> 새 학습 실행
-> 이전 실행과 비교
```

기존 실행의 설정을 직접 변경하지 않고 복제 후 수정한다. 실행 이력의 재현성을 보존하기 위함이다.

## 6. 화면 구성

### 6.1 대시보드

첫 화면에서 다음 정보를 제공한다.

- 최근 시나리오
- 진행 중인 학습
- 최근 완료된 학습
- 최근 평가 결과
- 총 return과 완료 주문 수 요약
- RL과 기준 정책 비교 요약

### 6.2 시나리오 목록

- 읽기 전용 시나리오 목록과 상세 진입
- 시나리오 이름과 설명
- 시뮬레이션 기간
- 주문, strip, 촬영 기회 수
- 생성 및 수정 시각
- 유효성 검사 상태

단계 10에서는 `GET` 조회 API만 연결한다. 생성·복제·삭제 버튼과 변경 입력은 표시하지 않으며, 후속 mutation API와 versioning 정책이 확정된 뒤 추가한다.

### 6.3 시나리오 탐색기

탐색기는 요약, 지도, 주문·strip·촬영 기회 목록, 구조 검증 패널로 구성한다.

#### 지도 영역 (단계 11 1차 구현)

- 주문 geometry 표시
- 주문 우선순위별 색상 구분
- 선택한 pass의 접근 가능 회전 strip polygon 표시 및 선택 strip 강조
- 선택한 pass의 ground track 표시
- 선택한 pass의 footprint 또는 swath 표시
- URL query `passId`, `stripId`로 선택 상태 보존

지도는 Leaflet을 사용한다. 주문 윤곽은 기본 레이어로 표시하지만, full 시나리오의 SVG 노드를 제한하기 위해 모든 strip과 footprint를 동시에 그리지 않는다. pass 또는 strip을 선택한 경우에만 관련 상세 레이어를 추가한다.

초기 우선순위 색상은 다음과 같이 사용한다.

```text
red        = 빨간색
blue       = 파란색
background = 회색
```

#### 주문 목록 (단계 10 읽기 전용)

- 주문 ID와 이름
- 우선순위
- 촬영 요구 시작/종료 시각
- strip 수
- 허용 roll/tilt 범위
- geometry 유효성 상태

#### 설정 패널 (단계 10 읽기 전용)

- 하루 시뮬레이션 범위
- orbit/pass 데이터
- strip 촬영시간
- 최소 촬영 간격
- 자세 범위와 각속도
- 안정화 시간
- 보상과 패널티 계수
- 행동 후보 padding 크기

현재는 준비된 주문 데이터와 환경·보상 설정을 읽기 전용으로 확인한다. 주문 우선순위·촬영 요구 기간·허용 자세, 환경·보상 파라미터 변경과 지도 polygon 편집은 mutation API가 없으므로 보류한다. 후속 구현은 기존 scenario를 직접 덮어쓰지 않고 복제 또는 version 생성 뒤 검증, strip/opportunity 재생성, 실행 snapshot 연결을 하나의 변경 단위로 처리해야 한다.

### 6.4 촬영 기회 검사 화면

- pass별 촬영 기회 수
- pass별 ground track과 footprint 샘플
- footprint와 strip의 교차로 생성된 access window
- 선택한 주문/strip의 모든 기회
- 초반, 최소각, 후반 후보 구분
- 후보 시각과 요구 roll/tilt
- off-nadir
- 주문 요구 기간 포함 여부
- 사전 마스킹 또는 데이터 오류 사유

이 화면은 궤도 인터페이스와 RL 환경 사이의 입력 데이터를 검증하는 용도로 사용한다. 실제 궤도 데이터가 없는 동안에는 가상 ground track/footprint 생성기가 만든 opportunity의 공간적 근거를 확인한다.

### 6.5 학습 설정 및 제어

- 시나리오 선택
- 알고리즘 선택
- random seed
- 총 학습 step 또는 episode
- 주요 하이퍼파라미터
- 체크포인트 저장 주기
- 평가 실행 주기
- 학습 시작과 중지

진행 중에는 다음 정보를 표시한다.

- 실행 상태
- 경과 시간
- 완료 step과 episode
- 최근 평균 return
- 최고 평가 점수
- 완료 주문 수
- 우선순위별 완료율
- 오류 및 경고 로그

학습 중지 요청은 현재 안전한 저장 지점에서 모델과 실행 상태를 저장한 뒤 종료하는 것을 원칙으로 한다.

1차 학습 제어 화면은 `/training`과 `/training/{run_id}`로 구성한다. 설정 화면은 저장된 시나리오와 Maskable PPO의 snapshot 필드만 입력받고, active run이 하나라도 있으면 새 시작을 막는다. 상세 화면은 URL의 run ID를 상태 복구 키로 사용하며, `queued`·`running`·`stop_requested` 상태에서 3초 REST polling으로 run detail과 metrics를 갱신한다. 중지 버튼은 `stop_requested`를 최종 중지로 오해하지 않고 worker callback이 checkpoint를 저장한 뒤 `stopped`가 될 때까지 상태를 표시한다. 화면은 checkpoint 목록 수와 final model/final evaluation 존재 여부를 표시하지만 artifact 파일을 직접 다운로드하거나 열지 않는다.

### 6.6 학습 곡선

- episode return
- 기본 우선순위 점수
- 미완료 패널티
- 평균 off-nadir
- 전체 strip 촬영률
- 완전 완료 주문 수
- 우선순위별 완료율
- policy/value 관련 학습 지표

그래프는 smoothing 적용 여부를 선택할 수 있어야 하며 원본 값도 확인할 수 있어야 한다.

### 6.7 결과 비교

동일 시나리오에서 다음 정책을 비교한다.

- 학습된 RL 정책
- Random valid
- Earliest deadline first
- Priority greedy
- Priority-efficiency greedy
- 소규모 시나리오의 최적화 결과

표와 차트에서 다음 지표를 비교한다.

- 총 return
- 기본 우선순위 점수
- 완료 주문 수
- red/blue/background 완료율
- 전체 strip 촬영률
- 평균 off-nadir
- 총 자세 전환시간
- 미완료 패널티
- 유효 기회 사용률

1차 `/comparisons` 화면은 완료된 `EvaluationRun`만 대상으로 한다. 화면은 scenario와 seed가 같은 run을 한 그룹으로 보여 주고, 사용자가 선택한 집합을 immutable `PolicyComparison` artifact로 만든다. 비교 표는 total return, priority score, 완료 strip/order, captures를 표시하며, 각 행이 보존한 `evaluation_run_id`를 이용해 원본 결과 상세와 episode 재생으로 이동한다. PPO 최종 평가와 CP-SAT 결과가 정식 EvaluationRun으로 저장되면 같은 선택 흐름에 추가한다.

### 6.8 스케줄 타임라인

24시간 동안의 촬영 결과를 시간축으로 표시한다.

- orbit/pass 구간
- 실제 촬영 구간
- 주문과 strip
- 우선순위 색상
- 촬영 전후 자세
- 자세 전환시간
- 최소 촬영 간격
- 선택되지 않은 주요 후보

촬영 항목을 선택하면 주문, strip, opportunity, 보상 및 마스킹 정보를 상세 패널에 표시한다.

구현된 결과 지도는 schedule에 포함된 strip을 완료, 포함되지 않은 strip을 미촬영으로 표시한다. 주문은 소유 strip의 완료 수가 0이면 미촬영, 전체면 완료, 그 사이는 부분 완료다. 우선순위는 polygon 테두리 색으로 유지하고 실행 상태는 채움 색으로 구분한다. 선택 capture는 `step_index`로 원본 replay step을 직접 조회해 촬영 전후 roll/tilt, 보상 구성, 누적 return, 무효 후보 수와 action mask 사유를 상세 패널에 표시한다.

### 6.9 지도 결과

- 주문 영역과 strip
- 촬영 완료, 부분 완료, 미촬영 상태
- 선택한 pass의 ground track
- 선택한 시각 또는 구간의 footprint
- 해당 pass에서 촬영한 strip
- 선택된 촬영 기회의 자세와 시각

지도와 타임라인 선택 상태를 연동한다. 타임라인의 촬영을 선택하면 지도에서 해당 strip을 강조하고, 지도에서 strip을 선택하면 관련 촬영 시각으로 이동한다.

### 6.10 에피소드 재생 및 step 분석

다음 제어를 제공한다.

```text
처음 / 이전 step / 재생 / 정지 / 다음 step / 마지막
```

각 step에서 표시할 내용은 다음과 같다.

- 현재 시각과 위성 자세
- 현재 action 후보 목록
- 후보별 action mask 상태
- 마스킹 사유
- 선택한 action 또는 skip
- 선택 전후 state 요약
- 받은 촬영 보상
- 각도 보너스
- 미완료 패널티
- 누적 return
- 주문과 strip 완료 상태 변화

초기 구현의 재생 데이터는 RL core의 `EpisodeReplay` JSON 계약을 기준으로 조회한다. Backend는 보상, 후보, action mask 사유를 재계산하지 않고 저장된 replay를 전달하거나 필요한 화면 형식으로만 변환한다.

재생 속도와 특정 step 직접 이동 기능을 제공한다.

1차 재생 화면은 결과 상세의 `Episode 재생 열기`에서 진입한다. 처음·이전·재생/정지·다음·마지막 제어, 0.5×~4× 속도 및 step range 입력을 제공한다. 각 step은 direct step API에서 읽으며, 선택 action이 촬영이면 Scenario의 pass와 strip 지도 레이어를 강조한다. skip action은 지도 선택을 바꾸지 않는다.

## 7. 실시간 통신

일반 조회와 명령에는 REST API를 사용한다.

- 시나리오 조회와 검증 (초기 범위)
- 학습 시작 및 중지
- 실행 목록과 상세 조회
- 평가 실행
- 결과 및 재생 데이터 조회

학습 진행률과 로그처럼 지속적으로 갱신되는 정보에는 WebSocket을 사용한다.

연결이 끊겨도 학습은 계속되어야 한다. 다시 연결하면 데이터베이스와 실행 상태를 기준으로 최신 상태를 복구한다.

초기 구현에서는 WebSocket이 복잡할 경우 짧은 주기의 REST polling으로 시작할 수 있으나, 외부 API 계약은 실시간 전달 방식을 교체할 수 있게 분리한다.

## 8. 데이터 저장

### 8.1 SQLite

SQLite에는 비교적 작은 구조화 데이터를 저장한다.

- 시나리오 메타데이터
- 주문과 strip 메타데이터
- 환경 및 보상 설정
- 학습 실행 정보
- 평가 실행 정보
- 주요 평가 지표
- 산출물 파일 위치

### 8.2 파일 저장소

대용량 또는 바이너리 데이터는 로컬 파일로 저장한다.

```text
data/
|-- scenarios/
|   +-- <scenario-id>/
|       |-- geometry/
|       +-- opportunities/
|-- runs/
|   +-- <run-id>/
|       |-- config.json
|       |-- checkpoints/
|       |-- metrics/
|       |-- episodes/
|       +-- logs/
+-- models/
```

실제 디렉터리 구조와 파일 형식은 구현 단계에서 확정한다. 데이터베이스에는 대용량 episode step 로그를 직접 저장하지 않고 파일 위치와 요약만 기록한다.

2026-07-19 단계 8의 첫 구현에서는 `data/scheduler.sqlite3`를 기본 DB로 확정했다. `rl_core.storage.StorageRepository`가 scenario, training/evaluation run 및 artifact 색인을 관리하며, DB 경로는 data root 기준 상대 경로만 저장한다. JSON artifact는 임시 파일을 fsync한 뒤 원자 교체하고, 모델처럼 기존에 만들어진 바이너리 파일은 해시와 크기를 계산해 색인한다. 실행 worker의 중단 상태 복구는 Backend와 worker를 도입하는 단계에서 완성한다.

2026-07-20에는 `recover_interrupted_runs()`를 추가했다. 단일 worker supervisor는 기존 worker가 없음을 확인한 뒤 시작할 때만 이를 호출해 남아 있는 `running` 실행을 `failed`로, `stop_requested` 실행을 `stopped`로 기록한다. Backend 프로세스만 재시작한 경우에는 호출하지 않으므로 별도 worker가 계속 실행 중일 때 상태를 잘못 바꾸지 않는다. 자동 재개는 초기 범위에서 제외하고, 사용자는 보존된 checkpoint를 새 run으로 명시적으로 재시작한다.

## 9. 실행 상태 모델

학습과 평가 실행은 다음 상태를 가진다.

```text
queued
running
stop_requested
completed
stopped
failed
```

실행에는 다음 식별 정보를 저장한다.

- run ID
- scenario ID와 버전
- 알고리즘 및 파라미터
- 환경 및 보상 파라미터 snapshot
- random seed
- 시작 및 종료 시각
- 현재 상태
- 오류 정보
- 모델과 결과 파일 위치

## 10. API 영역

세부 URL과 payload는 구현 단계에서 별도 API 계약으로 확정한다. 초기 API 영역은 다음과 같다.

```text
/api/scenarios
/api/scenarios/{scenario_id}/orders
/api/scenarios/{scenario_id}/opportunities
/api/training-runs
/api/training-runs/{run_id}
/api/training-runs/{run_id}/detail
/api/training-runs/{run_id}/stop
/api/evaluation-runs
/api/evaluation-runs/{run_id}
/api/results/{run_id}
/api/results/{run_id}/timeline
/api/results/{run_id}/episodes
/api/results/{run_id}/episodes/{episode_id}/steps
/api/policy-comparisons
/api/policy-comparisons/{comparison_id}
/ws/training-runs/{run_id}
```

Backend가 반환하는 상태, 보상, action mask 및 결과 값은 RL core의 용어와 단위를 그대로 사용한다.

2026-07-20 첫 Backend 구현에서는 `GET /api/health`, `GET /api/version`, `GET /api/scenarios`, `GET /api/scenarios/{scenario_id}`를 추가했다. 목록은 `scenario_id`, 이름, seed, 생성·수정 시각만 반환하고, 상세는 검증된 `Scenario` 원본을 반환한다. 오류는 `{ "error": { "code", "message" } }` 형식을 공통으로 사용하며, 없는 시나리오는 404, DB에 색인됐지만 artifact가 없는 시나리오는 409로 구분한다.

같은 날 order/strip/opportunity 전용 목록 API를 추가했다. 세 API는 `offset`, `limit`, `total`, `items[]`를 공통으로 사용한다. orders는 priority, strips는 order ID, opportunities는 order ID·strip ID·pass ID·kind 필터를 지원한다. orders와 strips에는 하위 항목 수를, opportunities에는 `off_nadir_deg`를 포함해 Frontend가 도메인 계산을 중복하지 않도록 한다.

`GET /api/scenarios/{scenario_id}/validation`은 저장된 Scenario artifact의 존재, SHA-256 무결성 및 Pydantic 구조를 다시 검사한다. 파일을 읽을 수 있으나 hash 또는 구조가 잘못된 경우에도 200 응답의 `valid: false`, `issues[]`로 상세 경로와 원인을 표시한다. 파일 자체가 없을 때만 409 오류로 처리한다. action mask로 판단할 실행 시점 제약은 이 API의 오류로 포함하지 않는다.

2026-07-20 단계 10의 첫 Frontend 범위는 읽기 전용으로 확정했다. 목록·상세·주문·strip·opportunity·검증 API를 라우팅과 선택 상태에 연결하지만, scenario/order/environment/reward mutation endpoint는 추가하지 않는다. 후속 수정 기능은 scenario version 또는 명시적 복제본을 만든 뒤 검증, 파생 artifact 재생성 및 training/evaluation run snapshot과의 관계를 확정한 후 도입한다.

같은 날 첫 Frontend 구현은 `react-router-dom` 기반 app shell과 `/scenarios`, `/scenarios/:scenarioId` 경로를 추가했다. 공통 API client는 성공 JSON과 Backend의 구조화된 오류를 분리하고, 목록 페이지는 `GET /api/scenarios`의 로딩·빈 목록·오류·재시도 상태를 표시한다. 개발 환경에서는 Vite `/api` proxy가 FastAPI `127.0.0.1:8000`으로 요청을 전달하며, 배포 시 `VITE_API_BASE_URL` 환경 변수로 API 기본 경로를 설정할 수 있다. 상세 경로는 아직 placeholder이며 다음 구현 단위에서 scenario 상세와 탭을 연결한다.

이후 상세 경로에는 개요, 주문, Strip, 촬영 기회, 검증 탭을 연결했다. 상단 개요는 전체 Scenario에서 위성·환경·보상 설정과 데이터 개수를 읽기 전용으로 보여 주고, 대용량 목록은 각 전용 API를 탭별로 요청한다. URL query는 `tab`, `orderId`, `stripId`, `passId`, `priority`, `kind`, `offset`을 보존한다. 주문에서 Strip, Strip에서 촬영 기회로 이동하면 선택 ID가 다음 목록 필터로 이어지며, API 오류·로딩·빈 목록은 탭마다 독립적으로 표시한다.

대시보드는 최근 scenario와 training/evaluation run metadata를 각 목록 API에서 최대 5건 읽어 표시한다. `GET /api/training-runs`와 `GET /api/evaluation-runs`는 artifact를 읽지 않는 `{ items, offset, limit, total }` 응답이며 `scenario_id`, `status` 필터를 지원한다. 따라서 하나의 손상된 평가 artifact가 대시보드 전체를 막지 않으며, 완료 run의 결과 지표는 사용자가 선택한 뒤 결과 상세 API에서 검증한다.

`/results`는 완료된 평가 run 목록을, `/results/{run_id}`는 `GET /api/results/{run_id}`로 검증된 결과 지표를 표시한다. summary에는 total return, priority score, captures, 완료 order/strip, 평균 off-nadir 및 reward breakdown을 표시하고, 아직 결과가 없는 run은 상세 링크를 제공하지 않는다. 1차 지도·타임라인은 `GET /api/results/{run_id}/timeline`의 capture schedule을 24시간 목록으로 표시하고 선택 capture의 pass·strip을 Leaflet 지도에 강조한다. 선택 상태는 `captureId` query에 보존하며, 지도 strip 선택도 해당 strip의 capture를 선택한다. 후보/action mask 사유와 자세 전후 상태는 episode 재생 화면에서 step API로 연결한다.

`POST /api/evaluation-runs`는 초기에는 빠르게 끝나는 네 기준 정책만 동기 실행한다. 요청은 scenario ID, policy name, seed를 받고, 응답은 완료된 `EvaluationRun`, 성능 요약과 replay 경로를 반환한다. replay와 summary는 `data/evaluations/<run-id>/`에 저장하며, 정책 실행 실패는 run 상태를 `failed`로 남긴다. Maskable PPO 학습이나 장시간 CP-SAT 실행은 이 endpoint에 넣지 않고 worker 제어 단계에서 비동기 실행으로 확장한다. `POST /api/training-runs`, `POST /api/cp-sat-evaluation-runs`와 마찬가지로 scenario artifact가 저장된 해시·구조와 다르면(손상 또는 외부 수정) `409 scenario_artifact_invalid`로 실행을 시작하지 않는다.

`GET /api/evaluation-runs/{run_id}`는 실행 상태 polling을 위한 metadata를 반환한다. `GET /api/results/{run_id}`는 저장된 summary와 run을, `GET /api/results/{run_id}/timeline`은 화면에 필요한 capture schedule만 `offset`/`limit` pagination으로 반환한다. `GET /api/results/{run_id}/episodes`는 현재 단일 replay를 `evaluation` episode ID의 요약으로, `GET /api/results/{run_id}/episodes/evaluation/steps`는 후보·action mask 사유·선택 action·reward breakdown을 포함한 원본 step을 pagination으로 반환한다. `GET /api/results/{run_id}/episodes/evaluation/steps/{step_index}`는 선택 capture가 참조하는 단일 step을 직접 반환한다. Backend는 결과를 재계산하지 않으며 artifact의 색인 소유자·종류·SHA-256, JSON 구조, scenario/policy/seed 일치를 검증한다. 아직 완료되지 않은 run은 `409 evaluation_result_not_ready`, 실패·중지 run은 `409 evaluation_run_not_completed`, artifact 누락·손상은 각각 `evaluation_artifact_missing`, `evaluation_artifact_invalid`으로 구분하고, 알 수 없는 episode ID는 `404 episode_not_found`, step index는 `404 episode_step_not_found`이다.

`POST /api/policy-comparisons`는 `{ scenario_id, seed, evaluation_run_ids }`를 받고 선택한 모든 run이 완료 상태이며 같은 scenario·seed인지 검증한 뒤 `201`과 비교 metadata·artifact를 반환한다. 비교 artifact는 선택 당시의 요약과 replay를 다시 검증해 저장하므로, 화면이 임의의 최신 run을 자동으로 합치지 않는다. `GET /api/policy-comparisons/{comparison_id}`도 색인 소유자·SHA-256·Pydantic 계약을 검증하며, 누락·손상 artifact는 `409 policy_comparison_invalid`으로 구분한다.

`POST /api/cp-sat-evaluation-runs`는 CP-SAT 평가를 직접 기다리지 않고 queued `EvaluationRun`을 `202`로 반환한다. CP-SAT worker는 PPO 학습 worker와 단일 로컬 실행 슬롯을 공유하며, 완료 시 동일한 결과·지도·replay·비교 화면에서 읽는 summary와 replay artifact를 저장한다. `POST /api/training-runs`와 동일하게 이미 다른 worker가 실행 중이면 `409 execution_worker_busy`, process 시작 실패면 failed run을 남기고 `500 execution_worker_start_failed`를 반환한다. 완료된 CP-SAT `EvaluationRun`은 PPO 최종 평가와 마찬가지로 `GET /api/evaluation-runs`에 노출되므로 `POST /api/policy-comparisons`가 둘을 구분 없이 같은 비교 artifact로 묶을 수 있다. scenario artifact 해시·구조 검증 실패는 `POST /api/evaluation-runs`와 동일하게 `409 scenario_artifact_invalid`로 차단한다.

`POST /api/training-runs`는 `{ scenario_id, config }`을 받고 `202 Accepted`와 queued `TrainingRun`을 반환한다. Backend는 요청의 저장 경로를 신뢰하지 않고 artifact root를 `data/runs`로 고정한 config snapshot을 먼저 저장한다. 그 다음 단일 `TrainingWorkerSupervisor`가 별도 non-daemon spawn process를 시작하며, worker는 SQLite와 artifact에서 run·scenario·config를 다시 읽어 `train_maskable_ppo()`를 실행한다. 따라서 HTTP 요청 연결이 끝나도 학습은 계속된다. 이미 worker가 실행 중이면 `409 training_worker_busy`, process 시작 실패면 failed run을 남기고 `500 training_worker_start_failed`를 반환한다. Backend 재시작만으로 `recover_interrupted_runs()`를 호출하지 않아 살아 있는 별도 worker를 실패 처리하지 않는다. scenario artifact가 저장된 해시·구조와 다르면(손상 또는 외부 수정) worker를 시작하기 전 `409 scenario_artifact_invalid`로 학습을 차단한다.

`POST /api/training-runs/{run_id}/stop`은 process 강제 종료가 아닌 cooperative cancellation 요청이다. queued run은 즉시 `stopped`, running run은 `stop_requested`가 되며 요청은 idempotent하다. worker의 PPO callback이 다음 training step 경계에서 상태를 관찰해 마지막 checkpoint를 저장하고 학습을 멈춘 뒤 `stopped`를 기록한다. 중지된 run은 불완전한 정책을 최종 결과로 보이지 않도록 final model·final evaluation·replay를 생성하지 않는다. 없는 run은 `404 training_run_not_found`, terminal run 중지 요청은 `409 training_run_not_stoppable`으로 반환한다.

`GET /api/training-runs/{run_id}`는 SQLite의 `TrainingRun`을 polling용으로 반환한다. `GET /api/training-runs/{run_id}/detail`은 동일 run의 검증된 config snapshot, checkpoint 파일명 목록과 final model/final evaluation 존재 여부를 반환해 새로고침 후 제어 화면을 복구한다. `EvaluationRun.source_training_run_id` → 이 endpoint의 config snapshot·checkpoint 목록·`final_model_available`까지가 API로 보장하는 추적 경로이며, 실제 모델 파일은 API로 내려받지 않고 `TrainingRun.artifact_directory`와 `model/final-model.zip` 저장 규칙을 조합해 로컬 파일시스템에서 찾는다(단일 사용자 로컬 프로토타입이므로 별도 다운로드 endpoint는 두지 않는다). `GET /api/training-runs/{run_id}/metrics`는 `{ run, items, offset, limit, total }` 형식으로 학습 곡선을 반환한다. Backend는 worker process의 메모리가 아닌 저장된 run과 `training-metrics.jsonl`만 조회한다. 첫 평가 전에는 빈 목록이 정상이며, 실행 중 마지막 JSONL 행이 아직 끝나지 않았으면 잠시 보류한다. 완료·중지·실패 run의 손상된 metrics 행은 `409 training_metrics_invalid`으로 표시한다.

## 11. 검증과 오류 표시

시나리오 저장 또는 학습 시작 전에 다음 항목을 검사한다.

- 주문 요구 시작 시각이 종료 시각보다 빠른가
- geometry와 strip 데이터가 유효한가
- 주문에 하나 이상의 strip이 있는가
- 촬영 기회가 주문 요구 기간 안에 있는가
- 요구 자세가 허용 범위 안에 있는가
- 촬영 후보가 1일 범위 안에 있는가
- 동일 ID가 중복되지 않는가
- 필수 환경 파라미터가 유효한가
- 최대 strip 및 행동 후보 크기를 위반하지 않는가

오류는 학습 실행 시점까지 숨기지 않고 편집 화면에서 해당 주문, strip 또는 필드와 연결해 표시한다.

## 12. 초기 개발 범위

### 포함

- 로컬 단일 사용자 웹 앱
- 기존 시나리오 조회와 속성 편집
- 주문, strip, 촬영 기회 시각화
- 가상 또는 실제 ground track/footprint 조회와 검수
- 학습 시작, 중지 및 상태 확인
- 학습 곡선
- 정책별 결과 비교
- 지도와 타임라인 결과
- 에피소드 step 재생
- SQLite와 로컬 파일 저장

### 제외

- 사용자 계정과 인증
- 인터넷 공개 배포
- 다중 사용자 동시 편집
- 여러 학습 worker의 분산 실행
- 클라우드 스토리지
- 모바일 전용 UI
- 지도 기반의 고급 GIS 편집
- 지도에서 주문 polygon 직접 생성
- 주문 polygon의 자동 strip 분할
- 웹 또는 RL core 내부의 정밀 궤도 전파와 촬영 기회 계산
- 운영 위성 시스템과 실시간 연동

## 13. 권장 구현 순서

1. RL core와 데이터 모델을 웹에서 독립적으로 구현한다.
2. 고정 시나리오를 로드하고 검증하는 backend API를 만든다.
3. 시나리오와 촬영 기회 조회 화면을 만든다.
4. 기준 정책 실행과 결과 저장을 연결한다.
5. 타임라인과 지도 결과 화면을 만든다.
6. 학습 worker와 실행 상태 API를 연결한다.
7. 학습 곡선과 실시간 상태를 추가한다.
8. 에피소드 step 로그와 재생 화면을 추가한다.
9. RL과 기준 정책 비교 및 피드백 흐름을 완성한다.

## 14. 향후 분리 가능한 문서

프로젝트가 커지면 다음 내용을 별도 문서로 분리한다.

- `api-contract.md`: 구체적인 endpoint와 request/response schema
- `data-format.md`: 시나리오, geometry, opportunity 및 episode 파일 형식
- `development-guide.md`: 로컬 실행, 테스트 및 빌드 방법

현 단계에서는 RL 설계와 웹 설계 두 문서를 기준 문서로 유지한다.
