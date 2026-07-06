# NSICPS RL Scheduling

강화학습 기반 단일 위성 촬영 스케줄링 학습용 프로토타입이다.

## 개발 환경

- Python 3.12
- Node.js 22
- React 19 + TypeScript + Vite

## Python 설정 및 검증

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy rl_core
```

## Frontend 검증

```powershell
Set-Location frontend
npm install
npm run build
```

구현 순서와 완료 조건은 [단계별 구현 계획](docs/implementation-plan.md)을 따른다.

