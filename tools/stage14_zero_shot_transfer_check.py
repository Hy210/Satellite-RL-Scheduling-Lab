"""이미 학습된 Maskable PPO 모델을 재학습 없이 새로운 시나리오들에 평가한다.

`tools/stage14_full_scale_direction_check.py`의 8-seed 실행은 전부 같은 시나리오
(seed 20260707)를 학습한 것이었다 — 학습 seed만 다양화했지 문제(시나리오) 자체는
한 번도 안 바뀌었다. 그래서 "다른 시나리오에서도 잘할 것인가"에는 답을 주지 못한다.

이 스크립트는 그 질문에 직접 답한다: 학습을 새로 하지 않고, 이미 학습된 모델을
`--scenario-seeds`로 지정한 학습에 쓰이지 않은(unseen) 시나리오 여러 개에 그대로
평가(zero-shot)한다. 시나리오 하나만 확인하면 "운 좋게 비슷했다/나빴다"를 구분할 수
없으므로, 기본으로 여러 개를 확인한다. 평가만 하므로(학습 없음) 전부 몇 분 안에 끝난다.

각 시나리오에서 baseline 4종도 함께 평가해, "이 새 시나리오 자체의 난이도" 대비
PPO zero-shot 성적을 상대적으로 비교할 수 있게 한다(시나리오마다 만점 자체가 다르므로
PPO의 raw return만으로는 비교가 안 된다).
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

TRAINING_SCENARIO_SEED = 20260707
DEFAULT_SCENARIO_SEEDS = (1001, 1002, 1003, 1004, 1005)
DEFAULT_SCENARIO_SIZE = "full"
RANDOM_VALID_EVAL_SEEDS = (101, 102, 103)
DETERMINISTIC_BASELINE_EVAL_SEED = 101
PPO_EVALUATION_SEED = 17


def main() -> None:
    """CLI 인자를 읽고 zero-shot 전이 확인을 실행한 뒤 사람이 읽을 요약을 출력한다."""

    args = _parse_args()
    if TRAINING_SCENARIO_SEED in args.scenario_seeds:
        raise ValueError(
            f"scenario seed {TRAINING_SCENARIO_SEED}은 학습에 쓰인 시나리오라 "
            "unseen 확인용으로 쓸 수 없다"
        )

    benchmark_root = args.artifact_root / f"stage14-zero-shot-transfer-{_timestamp()}"
    summary = run_check(
        benchmark_root=benchmark_root,
        model_path=args.model_path,
        scenario_size=args.scenario_size,
        scenario_seeds=args.scenario_seeds,
    )
    summary_path = benchmark_root / "summary.json"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _print_summary(summary_path, summary)


def run_check(
    *,
    benchmark_root: Path,
    model_path: Path,
    scenario_size: str,
    scenario_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """학습된 모델을 각 unseen 시나리오에서 baseline 4종과 함께 평가한다."""

    from rl_core.generator import generate_scenario
    from rl_core.policies import (
        EarliestDeadlineFirstPolicy,
        PriorityEfficiencyGreedyPolicy,
        PriorityGreedyPolicy,
        RandomValidPolicy,
        evaluate_policy,
    )
    from rl_core.training import evaluate_trained_policy, load_maskable_ppo_model

    per_scenario: list[dict[str, Any]] = []
    for scenario_seed in scenario_seeds:
        scenario = generate_scenario(seed=scenario_seed, size=scenario_size)

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

        model = load_maskable_ppo_model(model_path, scenario)
        zero_shot_evaluation = evaluate_trained_policy(model, scenario, seed=PPO_EVALUATION_SEED)
        ppo_payload = _common_payload(asdict(zero_shot_evaluation))

        heuristic_floor = min(
            baseline["earliest_deadline_first"]["total_return"],
            baseline["priority_greedy"]["total_return"],
            baseline["priority_efficiency_greedy"]["total_return"],
        )
        per_scenario.append(
            {
                "scenario_seed": scenario_seed,
                "scenario_id": scenario.scenario_id,
                "baseline": baseline,
                "ppo_zero_shot": ppo_payload,
                "beats_all_heuristics": ppo_payload["total_return"] > heuristic_floor,
                "beats_random_valid": ppo_payload["total_return"]
                > baseline["random_valid"]["median_total_return"],
            }
        )

    beat_heuristic_count = sum(1 for item in per_scenario if item["beats_all_heuristics"])
    beat_random_count = sum(1 for item in per_scenario if item["beats_random_valid"])

    return {
        "check": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "training_scenario_seed": TRAINING_SCENARIO_SEED,
            "model_path": str(model_path),
            "scenario_size": scenario_size,
            "artifact_root": str(benchmark_root),
        },
        "per_scenario": per_scenario,
        "summary": {
            "scenario_count": len(per_scenario),
            "beats_all_heuristics_count": beat_heuristic_count,
            "beats_random_valid_count": beat_random_count,
        },
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an already-trained Maskable PPO model zero-shot "
            "(no retraining) against scenarios it never trained on."
        ),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--scenario-size",
        choices=("tiny", "small", "full"),
        default=DEFAULT_SCENARIO_SIZE,
    )
    parser.add_argument(
        "--scenario-seeds",
        type=_parse_seed_list,
        default=DEFAULT_SCENARIO_SEEDS,
        help=(
            "Comma-separated unseen scenario seeds, e.g. '1001,1002' "
            f"(default: {','.join(str(s) for s in DEFAULT_SCENARIO_SEEDS)})."
        ),
    )
    return parser.parse_args()


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _print_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    print(f"summary: {summary_path}")
    print(f"model: {summary['check']['model_path']}")
    print(
        f"beats all heuristics on {summary['summary']['beats_all_heuristics_count']}/"
        f"{summary['summary']['scenario_count']} unseen scenarios"
    )
    print(
        f"beats random_valid on {summary['summary']['beats_random_valid_count']}/"
        f"{summary['summary']['scenario_count']} unseen scenarios"
    )
    for item in summary["per_scenario"]:
        ppo = item["ppo_zero_shot"]
        baseline = item["baseline"]
        heuristic_floor = min(
            baseline["earliest_deadline_first"]["total_return"],
            baseline["priority_greedy"]["total_return"],
            baseline["priority_efficiency_greedy"]["total_return"],
        )
        print(
            f"- seed {item['scenario_seed']}: ppo_zero_shot={ppo['total_return']:.3f} "
            f"heuristic_floor={heuristic_floor:.3f} "
            f"random_valid={baseline['random_valid']['median_total_return']:.3f} "
            f"beats_heuristics={item['beats_all_heuristics']}"
        )


if __name__ == "__main__":
    main()
