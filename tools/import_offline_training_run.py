"""외부(예: tools/stage14_*.py) 스크립트로 이미 완료한 Maskable PPO 학습 산출물을
`StorageRepository`/SQLite에 등록해 웹 GUI(`/results`, `/comparisons`)에서 볼 수 있게 한다.

학습을 다시 돌리지 않는다 — `metrics/replay.json`·`metrics/final-evaluation.json`은
`train_maskable_ppo()`가 `storage` 인자 유무와 무관하게 항상 저장하므로, 이미 계산된
값을 정식 저장 메서드로 옮겨 등록하기만 한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    args = _parse_args()
    training_run_id, evaluation_run_id = import_offline_training_run(
        run_directory=args.run_directory,
        data_root=args.data_root,
    )
    print(f"training_run_id: {training_run_id}")
    print(f"evaluation_run_id: {evaluation_run_id}")


def import_offline_training_run(*, run_directory: Path, data_root: Path) -> tuple[str, str]:
    """오프라인 PPO 학습 산출물 한 건을 저장소에 등록하고 (학습, 평가) run id를 반환한다."""

    from rl_core.models import (
        EvaluationRun,
        EvaluationSummary,
        MaskablePPOTrainingConfig,
        RunStatus,
        TrainingRun,
    )
    from rl_core.replay import load_episode_replay
    from rl_core.storage import StorageRepository

    repository = StorageRepository(data_root)
    run_id = run_directory.name

    run_json = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    config = MaskablePPOTrainingConfig.model_validate(
        json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    ).model_copy(update={"artifact_root": data_root / "runs"})

    destination = data_root / "runs" / run_id
    for subdirectory in ("checkpoints", "model", "metrics"):
        source = run_directory / subdirectory
        if source.is_dir():
            shutil.copytree(source, destination / subdirectory, dirs_exist_ok=True)

    training_run = TrainingRun(
        run_id=run_id,
        scenario_id=run_json["scenario_id"],
        algorithm=run_json["algorithm"],
        seed=run_json["seed"],
        total_timesteps=run_json["total_timesteps"],
        status=RunStatus.COMPLETED,
        artifact_directory=f"runs/{run_id}",
    )
    repository.save_training_config(training_run, config)

    final_evaluation = json.loads(
        (run_directory / "metrics" / "final-evaluation.json").read_text(encoding="utf-8")
    )
    replay = load_episode_replay(run_directory / "metrics" / "replay.json")

    evaluation_run_id = f"evaluation-ppo-{run_id}"
    running_evaluation = EvaluationRun(
        run_id=evaluation_run_id,
        scenario_id=training_run.scenario_id,
        policy_name=final_evaluation["policy_name"],
        seed=final_evaluation["seed"],
        status=RunStatus.RUNNING,
        source_training_run_id=run_id,
    )
    repository.save_evaluation_run(running_evaluation)
    summary = EvaluationSummary(
        policy_name=final_evaluation["policy_name"],
        scenario_id=training_run.scenario_id,
        seed=final_evaluation["seed"],
        steps=final_evaluation["steps"],
        captures=final_evaluation["captures"],
        total_return=final_evaluation["total_return"],
        priority_score=final_evaluation["priority_score"],
        angle_bonus=final_evaluation["angle_bonus"],
        missed_penalty=final_evaluation["missed_penalty"],
        completed_strips=final_evaluation["completed_strips"],
        completed_orders=final_evaluation["completed_orders"],
        average_off_nadir_deg=final_evaluation["average_off_nadir_deg"],
        replay_path=(Path("evaluations") / evaluation_run_id / "replay.json").as_posix(),
    )
    repository.save_completed_evaluation(running_evaluation, summary, replay)

    return run_id, evaluation_run_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import an offline Maskable PPO training run into the StorageRepository.",
    )
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="Path to the completed run's artifact directory (has run.json, config.json, ...).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="StorageRepository data root the backend server points at.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
