"""단계 6 Maskable PPO 성능 검증을 반복 seed 기준으로 실행한다."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCENARIO_SEED = 20260707
SCENARIO_SIZE = "tiny"
PPO_LEARNING_SEEDS = (11, 23, 37, 41, 53)
RANDOM_VALID_SEEDS = (101, 102, 103, 104, 105)
DEFAULT_TOTAL_TIMESTEPS = 50_000
DEFAULT_N_STEPS = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_EPOCHS = 5
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_EVALUATION_INTERVAL = 2_500
DEFAULT_CHECKPOINT_INTERVAL = 10_000
SKIP_COLLAPSE_THRESHOLD = 0.80
ACTION_CONCENTRATION_THRESHOLD = 0.80


def main() -> None:
    """CLI 인자를 읽고 benchmark를 실행한 뒤 사람이 읽을 요약을 출력한다."""

    args = _parse_args()
    benchmark_root = args.artifact_root / f"stage6-benchmark-{_timestamp()}"
    summary = run_benchmark(
        benchmark_root=benchmark_root,
        total_timesteps=args.total_timesteps,
    )
    summary_path = benchmark_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _print_summary(summary_path, summary)


def run_benchmark(*, benchmark_root: Path, total_timesteps: int) -> dict[str, Any]:
    """tiny 시나리오에서 Random valid와 Maskable PPO 반복 실험을 수행한다."""

    from rl_core.generator import generate_scenario
    from rl_core.models import MaskablePPOTrainingConfig
    from rl_core.policies import RandomValidPolicy, evaluate_policy
    from rl_core.training import train_maskable_ppo

    scenario = generate_scenario(seed=SCENARIO_SEED, size=SCENARIO_SIZE)
    benchmark_root.mkdir(parents=True, exist_ok=True)

    random_results = [
        _baseline_payload(evaluate_policy(RandomValidPolicy(), scenario, seed=seed))
        for seed in RANDOM_VALID_SEEDS
    ]
    random_total_return_median = median(item["total_return"] for item in random_results)
    random_completed_strips_median = median(item["completed_strips"] for item in random_results)

    ppo_results: list[dict[str, Any]] = []
    for learning_seed in PPO_LEARNING_SEEDS:
        run_id = f"maskable-ppo-seed-{learning_seed}"
        config = MaskablePPOTrainingConfig(
            total_timesteps=total_timesteps,
            learning_seed=learning_seed,
            evaluation_seed=RANDOM_VALID_SEEDS[0],
            n_steps=DEFAULT_N_STEPS,
            batch_size=DEFAULT_BATCH_SIZE,
            n_epochs=DEFAULT_N_EPOCHS,
            learning_rate=DEFAULT_LEARNING_RATE,
            gamma=DEFAULT_GAMMA,
            checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
            evaluation_interval=DEFAULT_EVALUATION_INTERVAL,
            deterministic_eval=True,
            artifact_root=benchmark_root / "ppo-runs",
        )
        artifacts = train_maskable_ppo(scenario, config, run_id=run_id)
        payload = _trained_payload(artifacts.final_evaluation)
        payload["learning_seed"] = learning_seed
        payload["run_directory"] = str(artifacts.run_directory)
        payload["final_model_path"] = str(artifacts.final_model_path)
        payload["final_evaluation_path"] = str(
            artifacts.run_directory / "metrics" / "final-evaluation.json"
        )
        ppo_results.append(payload)

    criteria = _criteria(
        ppo_results=ppo_results,
        random_total_return_median=random_total_return_median,
        random_completed_strips_median=random_completed_strips_median,
    )
    return {
        "benchmark": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scenario_seed": SCENARIO_SEED,
            "scenario_size": SCENARIO_SIZE,
            "scenario_id": scenario.scenario_id,
            "artifact_root": str(benchmark_root),
        },
        "training_config": {
            "total_timesteps": total_timesteps,
            "n_steps": DEFAULT_N_STEPS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "n_epochs": DEFAULT_N_EPOCHS,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "gamma": DEFAULT_GAMMA,
            "evaluation_interval": DEFAULT_EVALUATION_INTERVAL,
            "checkpoint_interval": DEFAULT_CHECKPOINT_INTERVAL,
            "deterministic_eval": True,
        },
        "random_valid": {
            "seeds": list(RANDOM_VALID_SEEDS),
            "runs": random_results,
            "summary": _aggregate(random_results),
        },
        "maskable_ppo": {
            "learning_seeds": list(PPO_LEARNING_SEEDS),
            "runs": ppo_results,
            "summary": _aggregate(ppo_results),
        },
        "criteria": criteria,
        "stage6_passed": all(criteria.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the strict stage 6 Maskable PPO benchmark.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/runs"),
        help="Benchmark artifact root. A timestamped subdirectory is created below this path.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="PPO total timesteps per learning seed.",
    )
    return parser.parse_args()


def _baseline_payload(evaluation: Any) -> dict[str, Any]:
    payload = _common_payload(asdict(evaluation))
    payload["seed"] = evaluation.seed
    payload["skip_ratio"] = _skip_ratio([decision.action for decision in evaluation.decisions])
    payload["non_skip_action_concentration"] = _non_skip_action_concentration(
        [decision.action for decision in evaluation.decisions]
    )
    return payload


def _trained_payload(evaluation: Any) -> dict[str, Any]:
    payload = _common_payload(asdict(evaluation))
    payload["evaluation_seed"] = evaluation.seed
    payload["skip_ratio"] = _skip_ratio([decision.action for decision in evaluation.decisions])
    payload["non_skip_action_concentration"] = _non_skip_action_concentration(
        [decision.action for decision in evaluation.decisions]
    )
    return payload


def _common_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": raw["policy_name"],
        "scenario_id": raw["scenario_id"],
        "steps": raw["steps"],
        "captures": raw["captures"],
        "total_return": raw["total_return"],
        "priority_score": raw["priority_score"],
        "angle_bonus": raw["angle_bonus"],
        "missed_penalty": raw["missed_penalty"],
        "completed_strips": raw["completed_strips"],
        "completed_orders": raw["completed_orders"],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_total_return": mean(item["total_return"] for item in results),
        "median_total_return": median(item["total_return"] for item in results),
        "mean_completed_strips": mean(item["completed_strips"] for item in results),
        "median_completed_strips": median(item["completed_strips"] for item in results),
        "mean_completed_orders": mean(item["completed_orders"] for item in results),
        "median_completed_orders": median(item["completed_orders"] for item in results),
        "mean_captures": mean(item["captures"] for item in results),
        "median_captures": median(item["captures"] for item in results),
        "mean_skip_ratio": mean(item["skip_ratio"] for item in results),
        "median_skip_ratio": median(item["skip_ratio"] for item in results),
        "mean_non_skip_action_concentration": mean(
            item["non_skip_action_concentration"] for item in results
        ),
        "median_non_skip_action_concentration": median(
            item["non_skip_action_concentration"] for item in results
        ),
    }


def _criteria(
    *,
    ppo_results: list[dict[str, Any]],
    random_total_return_median: float,
    random_completed_strips_median: float,
) -> dict[str, bool]:
    ppo_total_returns = [item["total_return"] for item in ppo_results]
    ppo_completed_strips = [item["completed_strips"] for item in ppo_results]
    return {
        "ppo_4_of_5_runs_exceed_random_valid_median_return": (
            sum(value > random_total_return_median for value in ppo_total_returns) >= 4
        ),
        "ppo_median_return_exceeds_random_valid_median_return": (
            median(ppo_total_returns) > random_total_return_median
        ),
        "ppo_median_completed_strips_not_below_random_valid_median": (
            median(ppo_completed_strips) >= random_completed_strips_median
        ),
        "no_ppo_run_has_skip_ratio_at_or_above_0_80": all(
            item["skip_ratio"] < SKIP_COLLAPSE_THRESHOLD for item in ppo_results
        ),
        "no_ppo_run_with_captures_has_non_skip_action_concentration_at_or_above_0_80": all(
            item["captures"] == 0
            or item["non_skip_action_concentration"] < ACTION_CONCENTRATION_THRESHOLD
            for item in ppo_results
        ),
    }


def _skip_ratio(actions: list[int]) -> float:
    if not actions:
        return 0.0
    return sum(action == 0 for action in actions) / len(actions)


def _non_skip_action_concentration(actions: list[int]) -> float:
    non_skip_actions = [action for action in actions if action != 0]
    if not non_skip_actions:
        return 0.0
    counts = Counter(non_skip_actions)
    return max(counts.values()) / len(non_skip_actions)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _print_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    random_summary = summary["random_valid"]["summary"]
    ppo_summary = summary["maskable_ppo"]["summary"]
    print(f"summary: {summary_path}")
    print(f"stage6_passed: {summary['stage6_passed']}")
    print(
        "random_valid median return: "
        f"{random_summary['median_total_return']:.3f}, "
        f"median strips: {random_summary['median_completed_strips']:.1f}"
    )
    print(
        "maskable_ppo median return: "
        f"{ppo_summary['median_total_return']:.3f}, "
        f"median strips: {ppo_summary['median_completed_strips']:.1f}"
    )
    for name, passed in summary["criteria"].items():
        print(f"- {name}: {passed}")


if __name__ == "__main__":
    main()
