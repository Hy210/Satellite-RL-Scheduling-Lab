"""Maskable PPO 학습, checkpoint 저장 및 고정 시나리오 평가를 담당한다."""

from __future__ import annotations

import json
from collections.abc import Iterable
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
    ) -> None:
        super().__init__(verbose=0)
        self.scenario = scenario
        self.seed = seed
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.metrics_path = metrics_path
        self.deterministic = deterministic
        self.checkpoints: list[Path] = []

    def _on_step(self) -> bool:
        # num_timesteps는 실제 학습이 소비한 transition 수다. 평가와 checkpoint를
        # 이 값에 맞춰 남기면 학습 곡선과 모델 파일을 같은 축에서 비교할 수 있다.
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
            path = self.checkpoint_dir / f"checkpoint-{self.num_timesteps}.zip"
            self.model.save(path)
            self.checkpoints.append(path)
        return True


def train_maskable_ppo(
    scenario: Scenario,
    config: MaskablePPOTrainingConfig,
    *,
    run_id: str,
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
    _write_json(run_directory / "config.json", config.model_dump(mode="json"))
    _write_json(run_directory / "run.json", run.model_dump(mode="json"))

    model.learn(total_timesteps=config.total_timesteps, callback=callback)

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

    completed_run = run.model_copy(update={"status": RunStatus.COMPLETED})
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
        decisions=tuple(decisions),
        replay=replay,
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        file.write("\n")


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
    else:
        payload["decisions"] = list(_decision_payloads(evaluation.decisions))
    return payload


def _decision_payloads(decisions: Iterable[TrainedPolicyDecision]) -> Iterable[dict[str, Any]]:
    for decision in decisions:
        yield asdict(decision)
