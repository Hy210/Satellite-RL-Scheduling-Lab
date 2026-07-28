"""여러 시나리오를 번갈아 겪으며(domain randomization) Maskable PPO를 학습하고,
held-out 시나리오에서 실제로 일반화가 개선됐는지 확인한다.

`tools/stage14_zero_shot_transfer_check.py`로 확인한 결과, 단일 시나리오(seed
20260707)로 8-seed 학습한 모델은 학습 시나리오에서는 8개 중 7개가 세 휴리스틱을
이겼지만, 재학습 없이 unseen 시나리오 5개(seed 1001~1005)에 그대로 평가하자 세
휴리스틱을 전부 이긴 건 1/5뿐이었다(`docs/project-knowledge.md` 6.16절) — 학습
시나리오에서 보인 우위의 상당 부분이 그 시나리오 하나에 대한 과적합이었다는 뜻이다.

이 스크립트는 학습 자체를 `--training-scenario-seeds`(기본 20개, 2001~2020) 여러
시나리오에 걸쳐 진행해 정책이 일반적인 전략을 배우도록 강제하고, 학습이 끝나면 같은
held-out 5개(1001~1005, 기존 zero-shot 확인과 동일 세트)에 대해 재학습 없이 평가해
기존 단일 시나리오 모델의 1/5 결과와 직접 비교한다.

learning rate 감쇠 등 다른 학습 안정화 기법은 이번 범위에서 의도적으로 제외했다 —
"여러 시나리오에 걸친 학습 자체가 되는지"를 먼저 확인하는 게 그 디테일을 다듬는 것보다
우선한다고 판단했다(사용자 논의, 2026-07-28). 학습 시나리오 자체에 대한 최고 성능은
단일 시나리오 학습(64.23, 6.14절)보다 낮아질 수 있다 — 여러 문제를 동시에 배우는 게
하나만 파는 것보다 어렵기 때문이며, 이는 알려진 trade-off다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

TRAINING_SCENARIO_SEED_START = 2001
DEFAULT_TRAINING_SCENARIO_COUNT = 20
# zero-shot 확인(stage14_zero_shot_transfer_check.py)과 동일 세트를 써야 결과가
# 직접 비교된다 — 절대 training pool과 겹치면 안 된다(main()에서 검사).
HELD_OUT_SCENARIO_SEEDS = (1001, 1002, 1003, 1004, 1005)
DEFAULT_SCENARIO_SIZE = "full"
LEARNING_SEED = 11
EVALUATION_SEED = 17
N_STEPS = 256
BATCH_SIZE = 64
N_EPOCHS = 5
DEFAULT_CHECKPOINT_INTERVAL = 10_000
DEFAULT_EVALUATION_INTERVAL = 2_000


def main() -> None:
    """CLI 인자를 읽고 domain randomization 학습·확인을 실행한다."""

    args = _parse_args()
    overlap = set(args.training_scenario_seeds) & set(HELD_OUT_SCENARIO_SEEDS)
    if overlap:
        raise ValueError(
            f"training scenario seeds must not overlap held-out seeds: {sorted(overlap)}"
        )

    benchmark_root = args.artifact_root / f"stage14-domain-randomization-{_timestamp()}"
    summary = run_check(
        benchmark_root=benchmark_root,
        scenario_size=args.scenario_size,
        training_scenario_seeds=args.training_scenario_seeds,
        total_timesteps=args.total_timesteps,
        checkpoint_interval=args.checkpoint_interval,
        evaluation_interval=args.evaluation_interval,
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
    training_scenario_seeds: tuple[int, ...],
    total_timesteps: int,
    checkpoint_interval: int,
    evaluation_interval: int,
) -> dict[str, Any]:
    """training pool에 걸쳐 학습한 뒤, 같은 held-out 5개로 zero-shot 성적을 확인한다."""

    from stage14_zero_shot_transfer_check import run_check as run_zero_shot_check

    from rl_core.generator import generate_scenario
    from rl_core.models import MaskablePPOTrainingConfig
    from rl_core.training import train_maskable_ppo_with_scenario_pool

    training_scenarios = [
        generate_scenario(seed=seed, size=scenario_size) for seed in training_scenario_seeds
    ]
    # 학습 중 곡선 추적은 held-out 세트 중 하나만 쓴다 — eval_freq마다 5개를 전부
    # 돌리면 학습 비용이 커지고, 최종 zero-shot 비교는 아래에서 5개를 전부 따로 확인한다.
    held_out_training_curve_scenario = generate_scenario(
        seed=HELD_OUT_SCENARIO_SEEDS[0], size=scenario_size
    )

    config = MaskablePPOTrainingConfig(
        total_timesteps=total_timesteps,
        learning_seed=LEARNING_SEED,
        evaluation_seed=EVALUATION_SEED,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        checkpoint_interval=checkpoint_interval,
        evaluation_interval=evaluation_interval,
        artifact_root=benchmark_root / "ppo-run",
    )
    artifacts = train_maskable_ppo_with_scenario_pool(
        training_scenarios,
        held_out_training_curve_scenario,
        config,
        run_id="domain-randomization",
    )

    zero_shot_summary = run_zero_shot_check(
        benchmark_root=benchmark_root / "zero-shot-check",
        model_path=artifacts.final_model_path,
        scenario_size=scenario_size,
        scenario_seeds=HELD_OUT_SCENARIO_SEEDS,
    )

    training_curve = [
        {"timesteps": row["timesteps"], "total_return": row["evaluation"]["total_return"]}
        for row in _read_metric_rows(artifacts.metrics_path)
    ]

    return {
        "check": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scenario_size": scenario_size,
            "training_scenario_seeds": list(training_scenario_seeds),
            "held_out_scenario_seeds": list(HELD_OUT_SCENARIO_SEEDS),
            "total_timesteps": total_timesteps,
            "artifact_root": str(benchmark_root),
            "final_model_path": str(artifacts.final_model_path),
        },
        "training_curve_on_held_out": training_curve,
        "zero_shot_comparison": zero_shot_summary,
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
            "Train Maskable PPO across a pool of scenarios (domain randomization) and "
            "check zero-shot generalization on the same held-out scenarios used by "
            "stage14_zero_shot_transfer_check.py."
        ),
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--scenario-size",
        choices=("tiny", "small", "full"),
        default=DEFAULT_SCENARIO_SIZE,
    )
    parser.add_argument(
        "--training-scenario-seeds",
        type=_parse_seed_list,
        default=tuple(
            range(
                TRAINING_SCENARIO_SEED_START,
                TRAINING_SCENARIO_SEED_START + DEFAULT_TRAINING_SCENARIO_COUNT,
            )
        ),
        help="Comma-separated training scenario seeds (the randomization pool).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        required=True,
        help="No sensible default — pick based on scenario-size/pilot results.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--evaluation-interval", type=int, default=DEFAULT_EVALUATION_INTERVAL)
    return parser.parse_args()


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _print_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    print(f"summary: {summary_path}")
    curve = summary["training_curve_on_held_out"]
    if curve:
        print(
            f"held-out training curve: first={curve[0]['total_return']:.3f} "
            f"last={curve[-1]['total_return']:.3f} (n={len(curve)} points)"
        )
    zero_shot = summary["zero_shot_comparison"]["summary"]
    print(
        f"zero-shot: beats all heuristics on {zero_shot['beats_all_heuristics_count']}/"
        f"{zero_shot['scenario_count']} held-out scenarios "
        "(single-scenario baseline was 1/5, see docs/project-knowledge.md 6.16)"
    )
    print(
        f"zero-shot: beats random_valid on {zero_shot['beats_random_valid_count']}/"
        f"{zero_shot['scenario_count']} held-out scenarios"
    )


if __name__ == "__main__":
    main()
