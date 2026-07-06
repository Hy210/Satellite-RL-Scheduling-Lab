from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

from rl_core.generator import generate_scenario
from rl_core.gym_env import (
    CANDIDATE_FEATURES,
    GLOBAL_FEATURES,
    STRIP_FEATURES,
    SatelliteSchedulingEnv,
)
from rl_core.simulator import SatelliteSchedulingSimulator


@pytest.fixture
def tiny_env() -> SatelliteSchedulingEnv:
    return SatelliteSchedulingEnv(generate_scenario(seed=20260707, size="tiny"))


def test_gymnasium_environment_contract(tiny_env: SatelliteSchedulingEnv) -> None:
    # checker는 action mask를 모르기 때문에 무작위 invalid action도 보내며,
    # 환경은 이를 안전한 skip으로 처리해야 한다.
    check_env(tiny_env, skip_render_check=True)


def test_observation_shapes_dtypes_and_ranges(tiny_env: SatelliteSchedulingEnv) -> None:
    observation, _ = tiny_env.reset(seed=7)
    config = tiny_env.scenario.environment

    assert observation["global"].shape == (len(GLOBAL_FEATURES),)
    assert observation["strips"].shape == (config.max_strips, len(STRIP_FEATURES))
    assert observation["strip_presence"].shape == (config.max_strips,)
    assert observation["candidates"].shape == (
        config.max_candidates,
        len(CANDIDATE_FEATURES),
    )
    assert observation["candidate_presence"].shape == (config.max_candidates,)
    assert all(value.dtype == np.float32 for value in observation.values())
    assert all(np.isfinite(value).all() for value in observation.values())
    assert tiny_env.observation_space.contains(observation)


def test_padding_presence_distinguishes_real_rows(tiny_env: SatelliteSchedulingEnv) -> None:
    observation, _ = tiny_env.reset()
    strip_count = len(tiny_env.scenario.strips)

    assert observation["strip_presence"][:strip_count].sum() == strip_count
    assert observation["strip_presence"][strip_count:].sum() == 0
    assert observation["strips"][strip_count:].sum() == 0
    assert observation["candidate_presence"].sum() == 0


def test_action_mask_has_fixed_shape_and_skip_is_always_valid(
    tiny_env: SatelliteSchedulingEnv,
) -> None:
    tiny_env.reset()

    masks = tiny_env.action_masks()

    assert masks.shape == (tiny_env.scenario.environment.max_candidates + 1,)
    assert masks.dtype == np.bool_
    assert masks[0]
    assert not masks[1:].any()


def test_candidate_features_and_masks_share_same_slots(
    tiny_env: SatelliteSchedulingEnv,
) -> None:
    tiny_env.reset()
    observation, _, _, _, _ = tiny_env.step(0)
    masks = tiny_env.action_masks()
    candidate_count = int(observation["candidate_presence"].sum())

    assert candidate_count > 0
    assert observation["candidate_presence"][:candidate_count].sum() == candidate_count
    assert observation["candidate_presence"][candidate_count:].sum() == 0
    for index in range(candidate_count):
        assert bool(observation["candidates"][index, -1]) == bool(masks[index + 1])


def test_masked_external_action_is_safely_converted_to_skip(
    tiny_env: SatelliteSchedulingEnv,
) -> None:
    tiny_env.reset()

    _, reward, _, _, info = tiny_env.step(1)

    assert reward == 0.0
    assert info["action_was_masked"] is True
    assert info["requested_action"] == 1
    assert info["executed_action"] == 0


def test_gym_wrapper_matches_core_for_same_valid_actions() -> None:
    scenario = generate_scenario(seed=31, size="tiny")
    env = SatelliteSchedulingEnv(scenario)
    core = SatelliteSchedulingSimulator(scenario)
    env.reset(seed=9)
    core.reset()

    for _ in range(10_000):
        if core.terminated:
            break
        core_mask = np.asarray(core.observe().action_mask, dtype=np.bool_)
        assert np.array_equal(env.action_masks(), core_mask)
        action = next((index for index, valid in enumerate(core_mask) if index and valid), 0)

        _, env_reward, env_terminated, env_truncated, info = env.step(action)
        core_result = core.step(action)

        assert env_reward == pytest.approx(core_result.reward)
        assert env_terminated == core_result.terminated
        assert env_truncated is False
        assert info["cumulative_return"] == pytest.approx(core.cumulative_return)
    else:
        pytest.fail("environment did not terminate")

    assert core.terminated


def test_reset_with_same_seed_is_reproducible(tiny_env: SatelliteSchedulingEnv) -> None:
    first, first_info = tiny_env.reset(seed=123)
    tiny_env.step(0)
    second, second_info = tiny_env.reset(seed=123)

    assert first_info == second_info
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_full_scenario_observation_stays_inside_fixed_space() -> None:
    env = SatelliteSchedulingEnv(generate_scenario(seed=71, size="full"))
    observation, _ = env.reset(seed=1)

    for _ in range(100):
        assert env.observation_space.contains(observation)
        masks = env.action_masks()
        action = next((index for index, valid in enumerate(masks) if index and valid), 0)
        observation, _, terminated, truncated, _ = env.step(action)
        assert not truncated
        if terminated:
            break


def test_sb3_contrib_detects_mask_and_predicts_valid_action(
    tiny_env: SatelliteSchedulingEnv,
) -> None:
    observation, _ = tiny_env.reset(seed=5)
    assert is_masking_supported(tiny_env)

    model = MaskablePPO(
        "MultiInputPolicy",
        tiny_env,
        n_steps=8,
        batch_size=4,
        seed=5,
        verbose=0,
    )
    masks = get_action_masks(tiny_env)
    action, _ = model.predict(observation, action_masks=masks, deterministic=True)

    assert masks[int(action)]


def test_maskable_ppo_can_collect_short_rollout(tiny_env: SatelliteSchedulingEnv) -> None:
    model = MaskablePPO(
        "MultiInputPolicy",
        tiny_env,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        seed=7,
        verbose=0,
    )

    model.learn(total_timesteps=8)

    assert model.num_timesteps == 8
