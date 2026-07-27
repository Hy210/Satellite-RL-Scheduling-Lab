"""Maskable PPO 학습 방향성과 모든 기준 정책(4개 휴리스틱 + CP-SAT)을 비교한다.

원래 단계 14 마지막 통합 검증(full 규모)을 위해 작성했으나, `--scenario-size`로
규모를 바꿔 재사용할 수 있게 일반화했다 — 예: full 규모 학습이 수렴하는지 보기 전에
같은 설계로 small 규모에서 훨씬 긴 학습이 깨끗하게 수렴하는지 먼저 저비용으로 확인하는 용도.
`--ppo-learning-seeds`로 learning seed 개수·값도 조절할 수 있다 — seed 2개로는 학습
안정성(seed에 따라 수렴 양상이 크게 다른지)을 판단하기에 통계적으로 부족하다고 판단해
기본값을 8개로 늘렸다.

사전에 정한 합격선은 없다 — CP-SAT이 OPTIMAL을 못 찾거나 PPO가 기준선을 못 넘어도
"실패"가 아니라 "관찰된 결과"로 기록하는 것이 목적이다. full 규모로 실행할 때는 처리량
벤치마크(`tools/stage14_scale_benchmark.py`)에서 측정한 것과 같은 하이퍼파라미터를 그대로
써야 그 처리량(steps/sec) 기반 시간 예산 추정이 유효하다.

full 규모 다중 seed 실행은 몇 시간이 걸릴 수 있고, 원인 불명 재부팅으로 두 번(2026-07-26,
2026-07-28) 중단된 전례가 있다. `--resume-from`에 중단된 실행의 benchmark_root(예:
`data/runs/stage14-full-direction-check-<timestamp>`)를 주면, 이미 `metrics/final-evaluation.json`이
저장된 seed는 재학습 없이 그대로 재사용하고 아직 안 끝난/시작 안 한 seed만 이어서 학습한다.
baseline 4종과 CP-SAT은 결정론적이고 재계산 비용이 낮아 재사용 없이 항상 새로 계산한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCENARIO_SEED = 20260707
DEFAULT_SCENARIO_SIZE = "full"
RANDOM_VALID_EVAL_SEEDS = (101, 102, 103)
DETERMINISTIC_BASELINE_EVAL_SEED = 101
CP_SAT_SEED = 17
CP_SAT_DEFAULT_TIME_LIMIT_SEC = 900.0
PPO_DEFAULT_LEARNING_SEEDS = (11, 23, 37, 41, 53, 67, 79, 89)
PPO_EVALUATION_SEED = 17
PPO_DEFAULT_TOTAL_TIMESTEPS = 30_000
N_STEPS = 256
BATCH_SIZE = 64
N_EPOCHS = 5
EVALUATION_INTERVAL = 2_000
CHECKPOINT_INTERVAL = 10_000


def main() -> None:
    """CLI 인자를 읽고 검증을 실행한 뒤 사람이 읽을 요약을 출력한다."""

    args = _parse_args()
    benchmark_root = (
        args.artifact_root / f"stage14-{args.scenario_size}-direction-check-{_timestamp()}"
    )
    summary = run_check(
        benchmark_root=benchmark_root,
        scenario_size=args.scenario_size,
        cp_sat_time_limit_sec=args.cp_sat_time_limit_sec,
        ppo_total_timesteps=args.ppo_total_timesteps,
        checkpoint_interval=args.checkpoint_interval,
        ppo_learning_seeds=args.ppo_learning_seeds,
        resume_from=args.resume_from,
    )
    summary_path = benchmark_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _print_summary(summary_path, summary)


def run_check(
    *,
    benchmark_root: Path,
    scenario_size: str,
    cp_sat_time_limit_sec: float,
    ppo_total_timesteps: int,
    checkpoint_interval: int,
    ppo_learning_seeds: tuple[int, ...] = PPO_DEFAULT_LEARNING_SEEDS,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """baseline 4종 → CP-SAT → Maskable PPO 순서로 지정 규모 시나리오를 평가·학습한다."""

    from rl_core.generator import generate_scenario
    from rl_core.models import MaskablePPOTrainingConfig
    from rl_core.optimization import solve_cp_sat_baseline
    from rl_core.policies import (
        EarliestDeadlineFirstPolicy,
        PriorityEfficiencyGreedyPolicy,
        PriorityGreedyPolicy,
        RandomValidPolicy,
        evaluate_policy,
    )
    from rl_core.training import train_maskable_ppo

    benchmark_root.mkdir(parents=True, exist_ok=True)
    scenario = generate_scenario(seed=SCENARIO_SEED, size=scenario_size)

    random_valid_runs = [
        _common_payload(asdict(evaluate_policy(RandomValidPolicy(), scenario, seed=seed)))
        for seed in RANDOM_VALID_EVAL_SEEDS
    ]
    baseline: dict[str, Any] = {
        "random_valid": {
            "runs": random_valid_runs,
            "median_total_return": median(item["total_return"] for item in random_valid_runs),
        }
    }
    for name, policy in (
        ("earliest_deadline_first", EarliestDeadlineFirstPolicy()),
        ("priority_greedy", PriorityGreedyPolicy()),
        ("priority_efficiency_greedy", PriorityEfficiencyGreedyPolicy()),
    ):
        result = evaluate_policy(policy, scenario, seed=DETERMINISTIC_BASELINE_EVAL_SEED)
        baseline[name] = _common_payload(asdict(result))

    cp_sat_result = solve_cp_sat_baseline(
        scenario, time_limit_sec=cp_sat_time_limit_sec, seed=CP_SAT_SEED
    )
    cp_sat_payload = {
        "status": cp_sat_result.status,
        "objective_value": cp_sat_result.objective_value,
        "best_objective_bound": cp_sat_result.best_objective_bound,
        "optimality_gap": cp_sat_result.optimality_gap,
        "time_limit_sec": cp_sat_time_limit_sec,
        "replay_total_return": cp_sat_result.replay.total_return,
        "completed_strips": cp_sat_result.replay.completed_strips,
        "completed_orders": cp_sat_result.replay.completed_orders,
    }

    ppo_results: list[dict[str, Any]] = []
    for learning_seed in ppo_learning_seeds:
        run_id = f"full-direction-check-seed-{learning_seed}"
        reused_run_dir = _find_resumable_run(resume_from, run_id)
        if reused_run_dir is not None:
            final_evaluation = json.loads(
                (reused_run_dir / "metrics" / "final-evaluation.json").read_text(encoding="utf-8")
            )
            payload = _common_payload(final_evaluation)
            payload["learning_curve"] = [
                {
                    "timesteps": row["timesteps"],
                    "total_return": row["evaluation"]["total_return"],
                }
                for row in _read_metric_rows(reused_run_dir / "metrics" / "training-metrics.jsonl")
            ]
            payload["reused_from"] = str(reused_run_dir)
        else:
            config = MaskablePPOTrainingConfig(
                total_timesteps=ppo_total_timesteps,
                learning_seed=learning_seed,
                evaluation_seed=PPO_EVALUATION_SEED,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                checkpoint_interval=checkpoint_interval,
                evaluation_interval=EVALUATION_INTERVAL,
                artifact_root=benchmark_root / "ppo-runs",
            )
            artifacts = train_maskable_ppo(scenario, config, run_id=run_id)
            payload = _common_payload(asdict(artifacts.final_evaluation))
            payload["learning_curve"] = [
                {"timesteps": row["timesteps"], "total_return": row["evaluation"]["total_return"]}
                for row in _read_metric_rows(artifacts.metrics_path)
            ]
        payload["learning_seed"] = learning_seed
        ppo_results.append(payload)

    ppo_median_return = median(item["total_return"] for item in ppo_results)
    comparison = {
        "maskable_ppo_median": ppo_median_return,
        "cp_sat": cp_sat_payload["replay_total_return"],
        "priority_efficiency_greedy": baseline["priority_efficiency_greedy"]["total_return"],
        "priority_greedy": baseline["priority_greedy"]["total_return"],
        "earliest_deadline_first": baseline["earliest_deadline_first"]["total_return"],
        "random_valid_median": baseline["random_valid"]["median_total_return"],
    }

    return {
        "check": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scenario_seed": SCENARIO_SEED,
            "scenario_size": scenario_size,
            "scenario_id": scenario.scenario_id,
            "artifact_root": str(benchmark_root),
        },
        "ppo_config": {
            "total_timesteps": ppo_total_timesteps,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "evaluation_interval": EVALUATION_INTERVAL,
            "checkpoint_interval": checkpoint_interval,
        },
        "baseline": baseline,
        "cp_sat": cp_sat_payload,
        "maskable_ppo": {
            "learning_seeds": list(ppo_learning_seeds),
            "runs": ppo_results,
            "median_total_return": ppo_median_return,
        },
        "comparison": comparison,
    }


def _find_resumable_run(resume_from: Path | None, run_id: str) -> Path | None:
    """`resume_from` 아래에 해당 run_id의 완료된 학습 산출물이 있으면 그 디렉터리를 반환한다.

    `final-evaluation.json`이 있어야 완료로 인정한다 — checkpoint만 있고 중단된
    run(예: 재부팅 도중 20,000/30,000 timesteps에서 끊긴 경우)은 재사용하지 않고
    처음부터 다시 학습한다.
    """

    if resume_from is None:
        return None
    run_dir = resume_from / "ppo-runs" / run_id
    if (run_dir / "metrics" / "final-evaluation.json").is_file():
        return run_dir
    return None


def _common_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": raw["policy_name"],
        "total_return": raw["total_return"],
        "priority_score": raw["priority_score"],
        "angle_bonus": raw["angle_bonus"],
        "missed_penalty": raw["missed_penalty"],
        "completed_strips": raw["completed_strips"],
        "completed_orders": raw["completed_orders"],
        "captures": raw["captures"],
    }


def _read_metric_rows(metrics_path: Path) -> list[dict[str, Any]]:
    """total_timesteps < evaluation_interval면 평가가 한 번도 안 돌아 파일이 없을 수 있다."""

    if not metrics_path.is_file():
        return []
    return [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Maskable PPO against every baseline policy "
            "(4 heuristics + CP-SAT) on the full-scale scenario."
        ),
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--scenario-size",
        choices=("tiny", "small", "full"),
        default=DEFAULT_SCENARIO_SIZE,
        help="Scenario scale to generate and evaluate/train against.",
    )
    parser.add_argument(
        "--cp-sat-time-limit-sec",
        type=float,
        default=CP_SAT_DEFAULT_TIME_LIMIT_SEC,
        help="Hard wall-clock cutoff for the CP-SAT solve.",
    )
    parser.add_argument(
        "--ppo-total-timesteps",
        type=int,
        default=PPO_DEFAULT_TOTAL_TIMESTEPS,
        help="PPO total timesteps per learning seed.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help="Timesteps between saved checkpoints (increase for long runs to limit disk I/O).",
    )
    parser.add_argument(
        "--ppo-learning-seeds",
        type=_parse_seed_list,
        default=PPO_DEFAULT_LEARNING_SEEDS,
        help=(
            "Comma-separated PPO learning seeds, e.g. '11,23,37' "
            f"(default: {','.join(str(s) for s in PPO_DEFAULT_LEARNING_SEEDS)})."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Prior benchmark_root (e.g. data/runs/stage14-full-direction-check-<timestamp>) "
            "whose already-completed PPO seed runs should be reused instead of retrained."
        ),
    )
    return parser.parse_args()


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _print_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    print(f"summary: {summary_path}")
    print(f"scenario_id: {summary['check']['scenario_id']}")
    cp_sat = summary["cp_sat"]
    print(f"cp_sat status: {cp_sat['status']} (gap={cp_sat['optimality_gap']})")
    print("comparison (total_return):")
    for name, value in summary["comparison"].items():
        rendered = f"{value:.3f}" if value is not None else "None"
        print(f"- {name}: {rendered}")
    for run in summary["maskable_ppo"]["runs"]:
        curve = run["learning_curve"]
        first = curve[0]["total_return"] if curve else None
        last = curve[-1]["total_return"] if curve else None
        print(
            f"ppo seed {run['learning_seed']}: first_eval={first}, last_eval={last}, "
            f"final={run['total_return']:.3f}"
        )


if __name__ == "__main__":
    main()
