"""단계 14 마지막 통합 검증: full 규모에서 Maskable PPO 학습 방향성과 모든 기준
정책(4개 휴리스틱 + CP-SAT)을 비교한다.

사전에 정한 합격선은 없다 — CP-SAT이 OPTIMAL을 못 찾거나 PPO가 기준선을 못 넘어도
"실패"가 아니라 "관찰된 결과"로 기록하는 것이 목적이다. 처리량 벤치마크
(`tools/stage14_scale_benchmark.py`)에서 측정한 것과 같은 하이퍼파라미터를 그대로 써야
그 처리량(steps/sec) 기반 시간 예산 추정이 유효하다.
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
SCENARIO_SIZE = "full"
RANDOM_VALID_EVAL_SEEDS = (101, 102, 103)
DETERMINISTIC_BASELINE_EVAL_SEED = 101
CP_SAT_SEED = 17
CP_SAT_DEFAULT_TIME_LIMIT_SEC = 900.0
PPO_LEARNING_SEEDS = (11, 23)
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
    benchmark_root = args.artifact_root / f"stage14-full-direction-check-{_timestamp()}"
    summary = run_check(
        benchmark_root=benchmark_root,
        cp_sat_time_limit_sec=args.cp_sat_time_limit_sec,
        ppo_total_timesteps=args.ppo_total_timesteps,
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
    cp_sat_time_limit_sec: float,
    ppo_total_timesteps: int,
) -> dict[str, Any]:
    """baseline 4종 → CP-SAT → Maskable PPO 순서로 full 시나리오를 평가·학습한다."""

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
    scenario = generate_scenario(seed=SCENARIO_SEED, size=SCENARIO_SIZE)

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
    for learning_seed in PPO_LEARNING_SEEDS:
        config = MaskablePPOTrainingConfig(
            total_timesteps=ppo_total_timesteps,
            learning_seed=learning_seed,
            evaluation_seed=PPO_EVALUATION_SEED,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            evaluation_interval=EVALUATION_INTERVAL,
            artifact_root=benchmark_root / "ppo-runs",
        )
        artifacts = train_maskable_ppo(
            scenario, config, run_id=f"full-direction-check-seed-{learning_seed}"
        )
        payload = _common_payload(asdict(artifacts.final_evaluation))
        payload["learning_seed"] = learning_seed
        payload["learning_curve"] = [
            {"timesteps": row["timesteps"], "total_return": row["evaluation"]["total_return"]}
            for row in _read_metric_rows(artifacts.metrics_path)
        ]
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
            "scenario_size": SCENARIO_SIZE,
            "scenario_id": scenario.scenario_id,
            "artifact_root": str(benchmark_root),
        },
        "ppo_config": {
            "total_timesteps": ppo_total_timesteps,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "evaluation_interval": EVALUATION_INTERVAL,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        "baseline": baseline,
        "cp_sat": cp_sat_payload,
        "maskable_ppo": {
            "learning_seeds": list(PPO_LEARNING_SEEDS),
            "runs": ppo_results,
            "median_total_return": ppo_median_return,
        },
        "comparison": comparison,
    }


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
    return parser.parse_args()


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
