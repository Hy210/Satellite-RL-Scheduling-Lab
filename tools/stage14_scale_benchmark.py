"""단계 14: tiny/small/full 시나리오의 Maskable PPO 학습 처리량·메모리를 측정한다.

각 규모마다 먼저 rollout 1회(calibration)로 처리량을 실측한 뒤, 남은 시간 예산에 맞춰
본측정(scaled) timesteps를 계산해 실행한다. 이전 실행의 메모리가 다음 측정을 오염시키지
않도록 각 측정은 별도 spawn 자식 프로세스에서 실행하고, 부모가 peak RSS를 polling한다.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCENARIO_SEED = 20260707
SCENARIO_SIZES: tuple[str, ...] = ("tiny", "small", "full")
LEARNING_SEED = 11
EVALUATION_SEED = 17
N_STEPS = 256
BATCH_SIZE = 64
N_EPOCHS = 5
MEMORY_POLL_INTERVAL_SEC = 0.5
DEFAULT_TOTAL_BUDGET_SEC = 25 * 60.0
DEFAULT_HARD_CEILING_SEC = 35 * 60.0


def main() -> None:
    """CLI 인자를 읽고 benchmark를 실행한 뒤 사람이 읽을 요약을 출력한다."""

    args = _parse_args()
    benchmark_root = args.artifact_root / f"stage14-scale-benchmark-{_timestamp()}"
    summary = run_benchmark(
        benchmark_root=benchmark_root,
        total_budget_sec=args.total_budget_sec,
        hard_ceiling_sec=args.hard_ceiling_sec,
    )
    summary_path = benchmark_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _print_summary(summary_path, summary)


def run_benchmark(
    *,
    benchmark_root: Path,
    total_budget_sec: float,
    hard_ceiling_sec: float,
) -> dict[str, Any]:
    """tiny/small/full 순서로 calibration과 적응형 본측정을 실행한다."""

    benchmark_root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    remaining_sizes = list(SCENARIO_SIZES)

    for size in SCENARIO_SIZES:
        remaining_sizes.pop(0)
        if time.monotonic() - start >= hard_ceiling_sec:
            results[size] = {"skipped": True, "reason": "hard_time_ceiling_reached"}
            continue

        run_root = benchmark_root / size
        calibration = _run_isolated(
            size=size, total_timesteps=N_STEPS, run_root=run_root / "calibration"
        )

        remaining_budget = max(total_budget_sec - (time.monotonic() - start), 0.0)
        budget_for_size = remaining_budget / max(len(remaining_sizes) + 1, 1)
        scaled_timesteps = _scaled_timesteps(
            steps_per_sec=calibration["steps_per_sec"], budget_sec=budget_for_size
        )

        scaled: dict[str, Any] | None = None
        if scaled_timesteps > N_STEPS and time.monotonic() - start < hard_ceiling_sec:
            scaled = _run_isolated(
                size=size, total_timesteps=scaled_timesteps, run_root=run_root / "scaled"
            )

        results[size] = {"calibration": calibration, "scaled": scaled}

    return {
        "benchmark": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scenario_seed": SCENARIO_SEED,
            "total_budget_sec": total_budget_sec,
            "hard_ceiling_sec": hard_ceiling_sec,
            "elapsed_sec": time.monotonic() - start,
            "artifact_root": str(benchmark_root),
        },
        "training_config": {
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
        },
        "sizes": results,
        "scaling_vs_tiny": _scaling_ratios(results),
    }


def _run_isolated(*, size: str, total_timesteps: int, run_root: Path) -> dict[str, Any]:
    """별도 spawn 자식 프로세스에서 학습을 실행하고 peak RSS를 polling으로 기록한다."""

    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / "result.json"
    context = multiprocessing.get_context("spawn")
    process: SpawnProcess = context.Process(
        target=_train_worker,
        args=(size, total_timesteps, str(run_root), str(result_path)),
        daemon=False,
    )
    process.start()

    peak_rss_bytes = 0
    try:
        probe = psutil.Process(process.pid)
        while process.is_alive():
            try:
                peak_rss_bytes = max(peak_rss_bytes, probe.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(MEMORY_POLL_INTERVAL_SEC)
    finally:
        process.join()

    if process.exitcode != 0:
        raise RuntimeError(
            f"benchmark worker failed for size={size} (exit code {process.exitcode})"
        )

    payload: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    payload["peak_rss_mb"] = peak_rss_bytes / (1024 * 1024)
    payload["steps_per_sec"] = payload["total_timesteps"] / payload["training_sec"]
    return payload


def _train_worker(size: str, total_timesteps: int, run_root: str, result_path: str) -> None:
    """자식 프로세스 진입점: device를 CPU로 고정하고 지정 규모를 짧게 학습한다.

    `training_sec`은 `train_maskable_ppo()` 호출 전체(학습 + 마지막 1회 필수 평가 +
    artifact 저장)를 포함한다 — 실제 운영에서 학습 1회가 지불하는 총 비용과 같다.
    """

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    Path(run_root).mkdir(parents=True, exist_ok=True)

    setup_start = time.monotonic()
    from rl_core.generator import generate_scenario
    from rl_core.models import MaskablePPOTrainingConfig
    from rl_core.training import train_maskable_ppo

    scenario = generate_scenario(seed=SCENARIO_SEED, size=size)
    config = MaskablePPOTrainingConfig(
        total_timesteps=total_timesteps,
        learning_seed=LEARNING_SEED,
        evaluation_seed=EVALUATION_SEED,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        checkpoint_interval=max(total_timesteps, N_STEPS),
        evaluation_interval=max(total_timesteps, N_STEPS),
        artifact_root=Path(run_root),
    )
    setup_sec = time.monotonic() - setup_start

    training_start = time.monotonic()
    train_maskable_ppo(scenario, config, run_id=f"{size}-benchmark")
    training_sec = time.monotonic() - training_start

    Path(result_path).write_text(
        json.dumps(
            {
                "size": size,
                "scenario_id": scenario.scenario_id,
                "total_timesteps": total_timesteps,
                "setup_sec": setup_sec,
                "training_sec": training_sec,
            }
        ),
        encoding="utf-8",
    )


def _scaled_timesteps(*, steps_per_sec: float, budget_sec: float) -> int:
    """calibration 처리량과 남은 예산으로 본측정 timesteps를 n_steps 배수로 정한다."""

    if steps_per_sec <= 0 or budget_sec <= 0:
        return N_STEPS
    rollouts = max(1, round((steps_per_sec * budget_sec) / N_STEPS))
    return rollouts * N_STEPS


def _scaling_ratios(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """tiny calibration 대비 처리량·메모리 배율을 계산한다."""

    tiny = results.get("tiny")
    tiny_calibration = tiny.get("calibration") if tiny else None
    if not tiny_calibration:
        return {}

    ratios: dict[str, Any] = {}
    for size, entry in results.items():
        if size == "tiny":
            continue
        calibration = entry.get("calibration")
        if not calibration:
            continue
        ratios[size] = {
            "throughput_ratio_vs_tiny": (
                tiny_calibration["steps_per_sec"] / calibration["steps_per_sec"]
                if calibration["steps_per_sec"] > 0
                else None
            ),
            "peak_memory_ratio_vs_tiny": (
                calibration["peak_rss_mb"] / tiny_calibration["peak_rss_mb"]
                if tiny_calibration["peak_rss_mb"] > 0
                else None
            ),
        }
    return ratios


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Maskable PPO throughput and memory across scenario scales.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/runs"),
        help="Benchmark artifact root. A timestamped subdirectory is created below this path.",
    )
    parser.add_argument(
        "--total-budget-sec",
        type=float,
        default=DEFAULT_TOTAL_BUDGET_SEC,
        help="Soft total time budget shared across all scenario sizes.",
    )
    parser.add_argument(
        "--hard-ceiling-sec",
        type=float,
        default=DEFAULT_HARD_CEILING_SEC,
        help="Hard safety stop; remaining sizes are skipped once reached.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _print_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    print(f"summary: {summary_path}")
    for size, entry in summary["sizes"].items():
        if entry.get("skipped"):
            print(f"- {size}: skipped ({entry['reason']})")
            continue
        calibration = entry["calibration"]
        print(
            f"- {size} calibration ({calibration['total_timesteps']} steps): "
            f"{calibration['steps_per_sec']:.1f} steps/sec, "
            f"peak {calibration['peak_rss_mb']:.1f} MB"
        )
        scaled = entry.get("scaled")
        if scaled:
            print(
                f"  {size} scaled ({scaled['total_timesteps']} steps): "
                f"{scaled['steps_per_sec']:.1f} steps/sec, "
                f"peak {scaled['peak_rss_mb']:.1f} MB"
            )
    print(f"scaling_vs_tiny: {json.dumps(summary['scaling_vs_tiny'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
