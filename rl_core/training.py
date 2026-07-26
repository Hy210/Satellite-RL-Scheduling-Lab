"""Maskable PPO 학습, checkpoint 저장 및 고정 시나리오 평가를 담당한다."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

from rl_core.gym_env import SatelliteSchedulingEnv
from rl_core.models import (
    EpisodeReplay,
    EvaluationRun,
    EvaluationSummary,
    MaskablePPOTrainingConfig,
    ReplayCapture,
    ReplayStep,
    RunStatus,
    Scenario,
    TrainingRun,
)
from rl_core.replay import (
    episode_replay,
    replay_candidates,
    replay_capture,
    replay_step,
    save_episode_replay,
)
from rl_core.simulator import RewardBreakdown
from rl_core.storage import StorageRepository


@dataclass(frozen=True, slots=True)
class TrainedPolicyDecision:
    """학습 정책의 평가 episode를 재현하기 위한 step 단위 선택 요약이다."""

    step_index: int
    time_sec: float
    action: int
    opportunity_id: str | None
    reward: float
    cumulative_return: float


@dataclass(frozen=True, slots=True)
class TrainedPolicyEvaluation:
    """RL 정책 평가 결과를 기준 정책 지표와 같은 의미로 모은 요약이다."""

    policy_name: str
    scenario_id: str
    seed: int
    steps: int
    captures: int
    total_return: float
    priority_score: float
    angle_bonus: float
    missed_penalty: float
    completed_strips: int
    completed_orders: int
    average_off_nadir_deg: float
    decisions: tuple[TrainedPolicyDecision, ...]
    replay: EpisodeReplay


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """한 학습 run이 남긴 파일 경로와 최종 평가 결과다."""

    run: TrainingRun
    run_directory: Path
    final_model_path: Path
    metrics_path: Path
    replay_path: Path
    checkpoints: tuple[Path, ...]
    final_evaluation: TrainedPolicyEvaluation


class TrainingStoppedError(RuntimeError):
    """사용자 중지 요청으로 checkpoint를 보존한 채 학습이 끝났음을 알린다."""


class FixedScenarioEvalCallback(BaseCallback):
    """학습 중 같은 평가 시나리오를 주기적으로 실행하고 metric을 JSONL로 남긴다."""

    def __init__(
        self,
        *,
        scenario: Scenario,
        seed: int,
        eval_freq: int,
        checkpoint_freq: int,
        checkpoint_dir: Path,
        metrics_path: Path,
        deterministic: bool,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.scenario = scenario
        self.seed = seed
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.metrics_path = metrics_path
        self.deterministic = deterministic
        self.should_stop = should_stop
        self.checkpoints: list[Path] = []
        self.stop_requested = False

    def _on_step(self) -> bool:
        # num_timesteps는 실제 학습이 소비한 transition 수다. 평가와 checkpoint를
        # 이 값에 맞춰 남기면 학습 곡선과 모델 파일을 같은 축에서 비교할 수 있다.
        if self.should_stop is not None and self.should_stop():
            self.stop_requested = True
            self._save_checkpoint()
            return False
        if self.num_timesteps % self.eval_freq == 0:
            evaluation = evaluate_trained_policy(
                cast(MaskablePPO, self.model),
                self.scenario,
                seed=self.seed,
                deterministic=self.deterministic,
            )
            _append_jsonl(
                self.metrics_path,
                {
                    "timesteps": self.num_timesteps,
                    "evaluation": _evaluation_payload(evaluation, include_decisions=False),
                },
            )
        if self.num_timesteps % self.checkpoint_freq == 0:
            self._save_checkpoint()
        return True

    def _save_checkpoint(self) -> None:
        """중지 시점에도 마지막 model 상태를 보존하도록 checkpoint를 한 번만 저장한다."""

        path = self.checkpoint_dir / f"checkpoint-{self.num_timesteps}.zip"
        if path in self.checkpoints:
            return
        self.model.save(path)
        self.checkpoints.append(path)


def train_maskable_ppo(
    scenario: Scenario,
    config: MaskablePPOTrainingConfig,
    *,
    run_id: str,
    storage: StorageRepository | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainingArtifacts:
    """단일 시나리오에서 Maskable PPO를 학습하고 모델·metric artifact를 저장한다."""

    run_directory = config.artifact_root / run_id
    checkpoint_dir = run_directory / "checkpoints"
    metrics_dir = run_directory / "metrics"
    model_dir = run_directory / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "training-metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    env = SatelliteSchedulingEnv(scenario)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        seed=config.learning_seed,
        verbose=0,
    )
    callback = FixedScenarioEvalCallback(
        scenario=scenario,
        seed=config.evaluation_seed,
        eval_freq=config.evaluation_interval,
        checkpoint_freq=config.checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        metrics_path=metrics_path,
        deterministic=config.deterministic_eval,
        should_stop=should_stop,
    )

    run = TrainingRun(
        run_id=run_id,
        scenario_id=scenario.scenario_id,
        algorithm="maskable_ppo",
        seed=config.learning_seed,
        total_timesteps=config.total_timesteps,
        status=RunStatus.RUNNING,
        artifact_directory=str(run_directory),
    )
    if storage is not None:
        storage.save_scenario(scenario)
        storage.save_training_run(run)
    _write_json(run_directory / "config.json", config.model_dump(mode="json"))
    _write_json(run_directory / "run.json", run.model_dump(mode="json"))

    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callback)

        if callback.stop_requested:
            stop_requested_run = run.model_copy(update={"status": RunStatus.STOP_REQUESTED})
            stopped_run = stop_requested_run.model_copy(update={"status": RunStatus.STOPPED})
            if storage is not None:
                # API가 먼저 상태를 바꾸지 못한 직접 실행 경로도 허용 전이를 지킨다.
                storage.save_training_run(stop_requested_run)
                storage.save_training_run(stopped_run)
            _write_json(run_directory / "run.json", stopped_run.model_dump(mode="json"))
            raise TrainingStoppedError(f"training stopped at timestep {model.num_timesteps}")

        final_model_path = model_dir / "final-model.zip"
        model.save(final_model_path)
        final_evaluation = evaluate_trained_policy(
            model,
            scenario,
            seed=config.evaluation_seed,
            deterministic=config.deterministic_eval,
        )
        _write_json(
            metrics_dir / "final-evaluation.json",
            _evaluation_payload(final_evaluation, include_decisions=True),
        )
        replay_path = metrics_dir / "replay.json"
        save_episode_replay(replay_path, final_evaluation.replay)

        if storage is not None:
            evaluation_run = EvaluationRun(
                run_id=f"evaluation-ppo-{run_id}",
                scenario_id=scenario.scenario_id,
                policy_name=final_evaluation.policy_name,
                seed=final_evaluation.seed,
                status=RunStatus.RUNNING,
                source_training_run_id=run_id,
            )
            storage.save_evaluation_run(evaluation_run)
            summary = EvaluationSummary(
                policy_name=final_evaluation.policy_name,
                scenario_id=scenario.scenario_id,
                seed=final_evaluation.seed,
                steps=final_evaluation.steps,
                captures=final_evaluation.captures,
                total_return=final_evaluation.total_return,
                priority_score=final_evaluation.priority_score,
                angle_bonus=final_evaluation.angle_bonus,
                missed_penalty=final_evaluation.missed_penalty,
                completed_strips=final_evaluation.completed_strips,
                completed_orders=final_evaluation.completed_orders,
                average_off_nadir_deg=final_evaluation.average_off_nadir_deg,
                replay_path=(
                    Path("evaluations") / evaluation_run.run_id / "replay.json"
                ).as_posix(),
            )
            storage.save_completed_evaluation(evaluation_run, summary, final_evaluation.replay)

        completed_run = run.model_copy(update={"status": RunStatus.COMPLETED})
        if storage is not None:
            storage.save_training_run(completed_run)
        _write_json(run_directory / "run.json", completed_run.model_dump(mode="json"))

        return TrainingArtifacts(
            run=completed_run,
            run_directory=run_directory,
            final_model_path=final_model_path,
            metrics_path=metrics_path,
            replay_path=replay_path,
            checkpoints=tuple(callback.checkpoints),
            final_evaluation=final_evaluation,
        )
    except TrainingStoppedError:
        raise
    except Exception as error:
        failed_run = run.model_copy(
            update={"status": RunStatus.FAILED, "error_message": str(error)}
        )
        if storage is not None:
            storage.save_training_run(failed_run)
        _write_json(run_directory / "run.json", failed_run.model_dump(mode="json"))
        raise


def load_maskable_ppo_model(path: Path, scenario: Scenario) -> MaskablePPO:
    """저장된 모델을 같은 환경 계약으로 다시 불러와 평가할 수 있게 한다."""

    return MaskablePPO.load(path, env=SatelliteSchedulingEnv(scenario))


def evaluate_trained_policy(
    model: MaskablePPO,
    scenario: Scenario,
    *,
    seed: int,
    deterministic: bool = True,
    max_steps: int = 1_000_000,
) -> TrainedPolicyEvaluation:
    """학습된 Maskable PPO 정책을 action mask와 함께 한 episode 평가한다."""

    env = SatelliteSchedulingEnv(scenario)
    observation, info = env.reset(seed=seed)
    decisions: list[TrainedPolicyDecision] = []
    replay_steps: list[ReplayStep] = []
    schedule: list[ReplayCapture] = []
    captures = 0
    priority_score = 0.0
    angle_bonus = 0.0
    missed_penalty = 0.0

    for step_index in range(max_steps):
        observation_before = env.simulator.observe()
        replay_candidates_before = replay_candidates(env.simulator, observation_before)
        masks = get_action_masks(env)
        raw_action, _ = model.predict(
            observation,
            action_masks=masks,
            deterministic=deterministic,
        )
        action = int(np.asarray(raw_action).item())
        if not masks[action]:
            raise ValueError(f"trained policy selected masked action {action}")

        time_sec = float(info.get("current_time_sec", env.simulator.current_time_sec))
        observation, reward, terminated, truncated, info = env.step(action)
        breakdown = cast(dict[str, float], info["reward_breakdown"])
        reward_breakdown = RewardBreakdown(
            strip_base=breakdown["strip_base"],
            angle_bonus=breakdown["angle_bonus"],
            missed_penalty=breakdown["missed_penalty"],
        )
        selected_opportunity_id = cast(str | None, info["selected_opportunity_id"])
        replay_steps.append(
            replay_step(
                step_index=step_index,
                simulator=env.simulator,
                observation_before=observation_before,
                candidates_before=replay_candidates_before,
                action=action,
                selected_opportunity_id=selected_opportunity_id,
                expired_order_ids=tuple(cast(tuple[str, ...], info["expired_order_ids"])),
                breakdown=reward_breakdown,
                observation_after=env.simulator.observe(),
            )
        )
        priority_score += breakdown["strip_base"]
        angle_bonus += breakdown["angle_bonus"]
        missed_penalty += breakdown["missed_penalty"]
        if selected_opportunity_id is not None:
            captures += 1
            schedule.append(
                replay_capture(
                    step_index=step_index,
                    simulator=env.simulator,
                    opportunity_id=selected_opportunity_id,
                    reward=float(reward),
                )
            )
        decisions.append(
            TrainedPolicyDecision(
                step_index=step_index,
                time_sec=time_sec,
                action=action,
                opportunity_id=selected_opportunity_id,
                reward=float(reward),
                cumulative_return=float(info["cumulative_return"]),
            )
        )
        if terminated or truncated:
            break
    else:
        raise RuntimeError(f"trained policy evaluation exceeded max_steps={max_steps}")

    replay = episode_replay(
        policy_name="maskable_ppo",
        scenario_id=scenario.scenario_id,
        seed=seed,
        steps=replay_steps,
        schedule=schedule,
    )
    opportunity_by_id = {item.opportunity_id: item for item in scenario.opportunities}
    average_off_nadir_deg = (
        sum(opportunity_by_id[item.opportunity_id].off_nadir_deg for item in schedule)
        / len(schedule)
        if schedule
        else 0.0
    )
    return TrainedPolicyEvaluation(
        policy_name="maskable_ppo",
        scenario_id=scenario.scenario_id,
        seed=seed,
        steps=len(decisions),
        captures=captures,
        total_return=float(info["cumulative_return"]),
        priority_score=priority_score,
        angle_bonus=angle_bonus,
        missed_penalty=missed_penalty,
        completed_strips=int(info["completed_strip_count"]),
        completed_orders=int(info["completed_order_count"]),
        average_off_nadir_deg=average_off_nadir_deg,
        decisions=tuple(decisions),
        replay=replay,
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _evaluation_payload(
    evaluation: TrainedPolicyEvaluation,
    *,
    include_decisions: bool,
) -> dict[str, Any]:
    payload = asdict(evaluation)
    payload["replay"] = evaluation.replay.model_dump(mode="json")
    if not include_decisions:
        payload.pop("decisions")
        # 학습 곡선은 요약 지표만 필요하므로 큰 episode replay를 매 행에 중복하지 않는다.
        payload.pop("replay")
    else:
        payload["decisions"] = list(_decision_payloads(evaluation.decisions))
    return payload


def _decision_payloads(decisions: Iterable[TrainedPolicyDecision]) -> Iterable[dict[str, Any]]:
    for decision in decisions:
        yield asdict(decision)
