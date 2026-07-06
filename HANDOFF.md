# Project Handoff

## 현재 상태

- 마지막 갱신일: 2026-07-07
- 작업 디렉터리: `D:\HY\01. Developer\Project\NSICPS_RL_Scheduling`
- 프로젝트 상태: 구현 계획 단계 0~5 완료
- 현재 구현 단계: 단계 6 Maskable PPO 학습 시작 전
- 진행 중 작업: 없음
- Blocker: 없음
- Git 상태: 로컬 `main`이 `origin/main`을 추적하며 첫 push 완료
- Git 원격: `https://github.com/Hy210/Satellite-RL-Scheduling-Lab.git`

## 현재 목표

강화학습 기반 단일 위성 촬영 스케줄링 프로토타입을 단계적으로 구현한다. 구현과 함께 사용자의 RL 학습을 지원하고 설계, 구현, 프로젝트 지식 및 작업 상태를 지속적으로 문서화한다.

## 완료된 작업

- RL 환경의 상태, 행동, 보상, 이벤트 진행 및 action masking 설계
- 단일 위성, 1일, 30 pass, 주문 100개 규모 확정
- seed 기반 고정 가상 시나리오 방향 확정
- 외부에서 분할된 strip과 사전 계산된 opportunity 입력 책임 확정
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
- `rl_core/simulator.py`: 결정론적 이벤트 시뮬레이터
- `rl_core/policies.py`: 기준 정책과 공통 평가기
- `rl_core/gym_env.py`: Gymnasium 및 Maskable PPO 연결 wrapper
- `tests/`: 데이터, 생성기, 시뮬레이터 및 통합 테스트
- `frontend/`: 최소 React/TypeScript/Vite 실행 골격

## 수행한 검증

- Python 3.12.10 전용 `.venv`를 생성하고 editable package 설치를 확인했다.
- Pytest: 60개 테스트 통과
- Python coverage: 95%
- Ruff lint: 통과
- Ruff format check: 통과
- Mypy strict 검사: 통과
- Frontend production build: 통과
- full 정책 비교(seed 20260707): strip 572개, opportunity 3,447개, 정책별 54~57개 경쟁 step 확인

## 알려진 문제와 미확정 사항

- 지도 라이브러리는 실제 geometry와 편집 요구를 확인한 뒤 선정한다.
- 실제 궤도 전파 인터페이스는 초기 가상 생성기 이후 검토한다.
- 후보 128개 제한은 full 시나리오 생성 후 분포와 잘림 영향을 검증해야 한다.
- 10초 opportunity 양자화는 가상 시나리오용 가정이므로 실제 궤도 데이터 연결 시 재검토해야 한다.
- 2,000개 strip을 평탄화하는 기본 MultiInputPolicy는 계산량이 클 수 있어 실제 학습 성능을 측정해야 한다.

## 다음 세션의 첫 작업

[구현 계획의 단계 6](docs/implementation-plan.md#단계-6-maskable-ppo-학습)를 시작한다.

tiny 시나리오에서 Maskable PPO 학습 설정, checkpoint, 평가 callback과 metric 기록을 구현한다. 먼저 짧은 학습으로 Random valid보다 나은 방향으로 학습 가능한지 확인하고, 관측 크기에 따른 속도와 메모리를 측정한다.

## 관련 문서

- [RL 스케줄링 설계](docs/rl-scheduling-design.md)
- [웹 애플리케이션 설계](docs/web-application-design.md)
- [단계별 구현 계획](docs/implementation-plan.md)
- [개인 RL 학습 노트](docs/rl-study-notes.md)
- [프로젝트 지식 베이스](docs/project-knowledge.md)
- [데이터 형식](docs/data-format.md)
