# Project Handoff

## 현재 상태

- 마지막 갱신일: 2026-07-08
- 작업 디렉터리: `D:\HY\01. Developer\Project\NSICPS_RL_Scheduling`
- 프로젝트 상태: 구현 계획 단계 0~7 완료, 단계 2A 완료
- 현재 구현 단계: 단계 8 저장 계층 구현 전
- 진행 중 작업: 없음
- Blocker: 없음
- Git 상태: 작업 트리에 단계 7 replay/comparison artifact 구현 및 문서 갱신 변경 있음
- Git 원격: `https://github.com/Hy210/Satellite-RL-Scheduling-Lab.git`

## 현재 목표

강화학습 기반 단일 위성 촬영 스케줄링 프로토타입을 단계적으로 구현한다. 구현과 함께 사용자의 RL 학습을 지원하고 설계, 구현, 프로젝트 지식 및 작업 상태를 지속적으로 문서화한다.

## 완료된 작업

- RL 환경의 상태, 행동, 보상, 이벤트 진행 및 action masking 설계
- 단일 위성, 1일, 30 pass, 주문 100개 규모 확정
- seed 기반 고정 가상 시나리오 방향 확정
- 외부에서 분할된 strip과 사전 계산된 opportunity 입력 책임 확정
- 실제 궤도 데이터가 없을 때 opportunity의 공간적 근거를 만들기 위한 가상 ground track/footprint 생성기 필요성 설계 반영
- `GroundTrackPoint`, `FootprintSample`, `AccessWindow` 데이터 모델 추가
- seed 기반 가상 ground track, 회전 strip/footprint polygon, footprint-strip 교차 기반 access window 생성 구현
- opportunity가 `source_access_window_id`로 파생 access window를 추적하도록 구현
- pass, ground track, footprint, strip 및 opportunity를 확인하는 임시 HTML 뷰어 `tools/scenario_viewer.html` 추가
- `tools/scenario_viewer.html`의 지도 컨테이너 높이와 로드 후 `fitBounds()` 처리를 보정해 타일과 오버레이가 분리되어 보이는 렌더링 문제를 수정
- React, FastAPI 및 독립 RL core 기반 웹 구조 설계
- 단계 0~14 구현 계획과 단계별 완료 조건 작성
- 개인 RL 학습 노트와 자동 기록 규칙 작성
- 프로젝트 지식 베이스와 자동 갱신 규칙 작성
- 설계 및 구현 변경 시 관련 문서 자동 동기화 규칙 작성
- 세션 간 handoff 규칙 작성
- 초기 알고리즘 Maskable PPO의 선택 이유와 학습 개념 기록
- Python 3.12 가상환경, 패키지 설정, lint/type/test 도구 구성
- React 19, TypeScript 6 및 Vite 8 최소 Frontend 골격 구성
- Pydantic 기반 Scenario, Satellite, Pass, Order, Strip, Opportunity 및 run 데이터 계약 구현
- seed 기반 tiny, small 및 full 가상 시나리오 생성기 구현
- 결정론적 이벤트 시뮬레이터, 자세 전환, action mask, 보상 및 주문 만료 구현
- 데이터 형식과 구조 검증/action mask 책임 경계 문서화
- Maskable PPO의 확률 마스킹, rollout 및 정책 업데이트 흐름을 개인 학습 노트에 보강
- 네 가지 기준 정책과 공통 평가·결정 로그 인터페이스 구현
- tiny/small/full 정책 종료, 마스킹 안전성 및 seed 재현성 검증
- 촬영 시각을 10초 grid로 양자화해 경쟁 후보가 생기도록 가상 생성기 보완
- 기준 정책 비교에서 환경의 의미 있는 선택 필요성을 학습 노트와 지식 베이스에 기록
- 주요 클래스·함수와 RL/도메인 로직에 간결한 한글 docstring 및 이유 중심 주석 보강
- 이후 작성하는 코드에도 같은 한글 설명 원칙을 적용하도록 `AGENTS.md`에 규칙 추가
- Gymnasium 1.3, Stable-Baselines3 2.9 및 sb3-contrib 2.9 의존성 구성
- 고정 Dict observation, `Discrete(129)` action과 `action_masks()` Gym wrapper 구현
- strip 2,000개와 후보 128개 padding 및 presence 배열 구현
- Gym checker, core 일치, seed, full 관측, Maskable PPO 예측·rollout 검증
- Dict observation과 padding/action mask 개념을 학습 노트와 데이터 형식에 기록
- Git 저장소를 `main` 브랜치로 초기화하고 GitHub 원격 `origin` 등록
- 초기 프로젝트 commit을 생성하고 GitHub `origin/main`으로 push 완료
- Stable-Baselines3와 sb3-contrib의 관계를 개인 RL 학습 노트에 기록
- `MaskablePPOTrainingConfig` 학습 설정 모델 구현
- `rl_core/training.py`에 Maskable PPO 학습, checkpoint, 주기적 고정 시나리오 평가, metric 기록, 최종 모델 저장 및 reload 평가 구현
- 학습 산출물 경로를 `data/runs/<run-id>/` 구조로 확정
- 단계 6 smoke 테스트 추가
- `tools/stage6_benchmark.py` 반복 seed 성능 검증 CLI 추가
- `synthetic-tiny-20260707`에서 Maskable PPO가 Random valid 대비 단계 6 통과 기준을 만족함을 검증
- 단계 6 검증이 Maskable PPO와 Random valid 기준선 비교라는 점과 tiny 검증의 한계를 개인 RL 학습 노트에 기록
- `EpisodeReplay`, `ReplayStep`, `ReplayCandidate`, `ReplayRewardBreakdown`, `ReplayCapture` 공통 재생 로그 계약 추가
- 기준 정책 평가와 Maskable PPO 평가가 같은 replay 로그를 생성하도록 구현
- PPO 학습 산출물에 `metrics/replay.json` 저장 추가
- replay JSON 저장/복원 helper와 reward 합계, action mask 사유 검증 테스트 추가
- `PolicyComparison` 및 `PolicyComparisonEntry` 비교 artifact 계약 추가
- 여러 정책 replay와 성능 지표를 하나의 비교 JSON으로 저장/복원하는 helper와 테스트 추가
- `docs/rl-scheduling-design.md`, `docs/data-format.md`, `docs/web-application-design.md`, `docs/implementation-plan.md`, `docs/project-knowledge.md`에 가상 ground track, footprint, access window 및 지도 검수 흐름 반영

## 주요 파일

- `AGENTS.md`: 프로젝트 작업과 자동 문서화 규칙
- `docs/rl-scheduling-design.md`: RL 환경과 도메인 설계
- `docs/web-application-design.md`: 웹 애플리케이션 설계
- `docs/implementation-plan.md`: 단계별 구현 계획
- `docs/rl-study-notes.md`: 개인 RL 학습 기록
- `docs/project-knowledge.md`: 프로젝트 전반의 축적 지식
- `docs/data-format.md`: 구현된 데이터 계약과 검증 경계
- `HANDOFF.md`: 현재 작업 상태와 다음 작업
- `pyproject.toml`: Python 패키지, 테스트, lint 및 type 검사 설정
- `rl_core/models.py`: Pydantic 데이터 계약
- `rl_core/generator.py`: seed 기반 가상 시나리오 생성기
- `tools/scenario_viewer.html`: 시나리오 JSON을 불러와 pass별 ground track, footprint, strip 및 opportunity를 확인하는 임시 지도 뷰어
- `tools/stage6_benchmark.py`: 단계 6 Maskable PPO와 Random valid 반복 seed 성능 비교 CLI
- `rl_core/simulator.py`: 결정론적 이벤트 시뮬레이터
- `rl_core/policies.py`: 기준 정책과 공통 평가기
- `rl_core/gym_env.py`: Gymnasium 및 Maskable PPO 연결 wrapper
- `rl_core/replay.py`: 평가 episode 재생 로그 생성과 JSON 저장/복원
- `rl_core/training.py`: Maskable PPO 학습, 평가 및 artifact 저장
- `tests/`: 데이터, 생성기, 시뮬레이터 및 통합 테스트
- `tests/test_training.py`: 단계 6 학습 artifact와 reload 평가 smoke 테스트
- `frontend/`: 최소 React/TypeScript/Vite 실행 골격

## 수행한 검증

- Python 3.12.10 전용 `.venv`를 생성하고 editable package 설치를 확인했다.
- Pytest: 63개 테스트 통과
- Python coverage: 95%
- Ruff lint: 통과
- Ruff format check: 통과
- Mypy strict 검사: 통과
- Frontend production build: 통과
- full 정책 비교(seed 20260707): strip 572개, opportunity 3,447개, 정책별 54~57개 경쟁 step 확인
- 단계 6 변경 파일 기준 Ruff lint 통과
- 단계 6 변경 파일 기준 Mypy 검사 통과
- `tests/test_training.py`: 3개 테스트 통과
- 단계 2A 변경 후 Pytest: 66개 테스트 통과
- 단계 2A 변경 후 Ruff lint: 통과
- 단계 2A 변경 후 Mypy strict 검사: 통과
- 2026-07-08 작업 시작 전 `git status --short --branch`: `main...origin/main`, 작업 트리 변경 없음
- `.venv\Scripts\python.exe -m pytest tests/test_training.py tests/test_policies.py`: 25개 테스트 통과
- `.venv\Scripts\python.exe tools/stage6_benchmark.py`: `stage6_passed=true`
  - 산출물: `data/runs/stage6-benchmark-20260708-224725/summary.json`
  - Maskable PPO median return: `5.326453530248241`
  - Random valid median return: `5.325396139404043`
  - 두 정책 모두 median completed strips: `9`
- 대표 PPO run의 `final-evaluation.json`에 reward breakdown과 step별 `decisions` 포함 확인
- `.venv\Scripts\python.exe -m ruff check .`: 통과
- `.venv\Scripts\python.exe -m mypy`: 통과
- `git status --short --ignored`: `data/runs/`는 ignored로 표시되어 학습 산출물이 Git 추적 대상에서 제외됨을 확인
- `.venv\Scripts\python.exe -m pytest tests/test_policies.py tests/test_training.py`: 26개 테스트 통과
- `.venv\Scripts\python.exe -m pytest`: 67개 테스트 통과
- `.venv\Scripts\python.exe -m ruff check .`: 통과
- `.venv\Scripts\python.exe -m mypy`: 통과
- `.venv\Scripts\python.exe -m pytest tests/test_policies.py`: 24개 테스트 통과

## 알려진 문제와 미확정 사항

- 지도 라이브러리는 실제 geometry와 편집 요구를 확인한 뒤 선정한다.
- 실제 궤도 전파 인터페이스는 가상 ground track/footprint 데이터 계약을 실제 데이터 형식에 맞춰 교체하는 방향으로 검토한다.
- 가상 ground track/footprint 생성기는 정밀 궤도 물리가 아니라 검수 가능한 근거 데이터용 단순 모델이다.
- `tools/scenario_viewer.html`은 추가했고, 첫 수동 확인에서 지도 컨테이너 렌더링 문제가 발견되어 CSS와 Leaflet size/bounds 처리를 수정했다. 이후 strip/footprint를 pass 진행 방향에 맞춘 polygon으로 변경했고, 브라우저에서 지도 렌더링과 기울기 정합성을 확인했다.
- 후보 128개 제한은 full 시나리오 생성 후 분포와 잘림 영향을 검증해야 한다.
- 10초 opportunity 양자화는 가상 시나리오용 가정이므로 실제 궤도 데이터 연결 시 재검토해야 한다.
- 2,000개 strip을 평탄화하는 기본 MultiInputPolicy는 full 규모에서 계산량이 클 수 있어 small/full 확장 시 성능을 다시 측정해야 한다.
- 단계 6 엄격 검증에서 tiny 시나리오는 통과했지만 개선 폭은 작고 완료 strip/order 수는 Random valid와 같았다. small/full 시나리오와 greedy 정책 비교에서 성능 의미를 다시 확인해야 한다.

## 다음 세션의 첫 작업

[구현 계획의 단계 8](docs/implementation-plan.md#단계-8-저장-계층)로 이동한다.

SQLite schema와 로컬 artifact 디렉터리 관리 방식을 구현한다. 시작점은 시나리오 메타데이터, 학습 run, 평가 run, `EpisodeReplay` 및 `PolicyComparison` 파일 경로를 추적하는 최소 저장 계층을 설계하고 테스트하는 것이다.

## 관련 문서

- [RL 스케줄링 설계](docs/rl-scheduling-design.md)
- [웹 애플리케이션 설계](docs/web-application-design.md)
- [단계별 구현 계획](docs/implementation-plan.md)
- [개인 RL 학습 노트](docs/rl-study-notes.md)
- [프로젝트 지식 베이스](docs/project-knowledge.md)
- [데이터 형식](docs/data-format.md)
