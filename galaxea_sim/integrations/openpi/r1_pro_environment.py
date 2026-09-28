"""把 R1 Pro Gym/SAPIEN 任务包装成 OpenPI Runtime 环境。"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from openpi_client.runtime import environment as openpi_environment
from typing_extensions import override

from galaxea_sim import ASSETS_DIR

from .r1_pro_action_adapter import GalaxeaR1ProActionAdapter
from .r1_pro_observation_adapter import GalaxeaR1ProObservationAdapter
from .r1_pro_safety import R1ProSafetyReport, make_r1_pro_safety_filter


class GalaxeaR1ProSimEnvironment(openpi_environment.Environment):
    """提供三相机、16维状态和16维关节位置动作的 R1 Pro 仿真环境。"""

    def __init__(
        self,
        env_name: str = "R1ProDualBottlesPickEasy-v0",
        *,
        seed: int = 0,
        control_freq: int = 15,
        headless: bool = True,
        ray_tracing: bool = False,
        max_episode_steps: int | None = None,
        max_joint_delta: float = 0.12,
        max_gripper_delta: float = 0.01,
        default_prompt: str | None = None,
        table_height: float | None = None,
    ) -> None:
        """创建 R1 Pro 任务及观测、动作和安全边界。

        类名和 Gym ID 都要求显式包含 ``R1Pro``，这是为了让错误的标准 R1
        策略在动作送入仿真器之前失败，而不是依靠 NumPy 广播产生隐蔽行为。
        """
        if not env_name.startswith("R1Pro"):
            raise ValueError("R1 Pro OpenPI adapter requires an R1Pro* Gym environment")
        self._rng = np.random.default_rng(seed)
        self._headless = headless
        self._gym = gym.make(
            env_name,
            control_freq=control_freq,
            headless=headless,
            obs_mode="image",
            ray_tracing=ray_tracing,
            include_depth=False,
            camera_resolution_scale=4,
            table_height_override=table_height,
        )
        self._env = self._gym.unwrapped
        if self._env.action_space.shape != (16,):
            raise ValueError(
                "R1 Pro joint-position adapter requires a 16-D action space, "
                f"got {self._env.action_space}"
            )
        if getattr(self._env, "controller_type", None) != "bimanual_joint_position":
            raise ValueError("R1 Pro OpenPI adapter requires bimanual_joint_position")
        if getattr(self._env.robot, "name", None) != "r1_pro":
            raise ValueError("Selected environment did not create an R1 Pro robot")

        self._observation_adapter = GalaxeaR1ProObservationAdapter(
            default_prompt=default_prompt or env_name
        )
        self._action_adapter = GalaxeaR1ProActionAdapter()
        self._safety = make_r1_pro_safety_filter(
            Path(ASSETS_DIR) / self._env.robot.urdf_path,
            max_joint_delta=max_joint_delta,
            max_gripper_delta=max_gripper_delta,
        )
        registered_limit = getattr(self._gym.spec, "max_episode_steps", None)
        if max_episode_steps is None:
            if registered_limit is None:
                raise ValueError(f"{env_name} has no registered episode limit")
            max_episode_steps = int(registered_limit)
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self._max_episode_steps = max_episode_steps
        self._last_raw_observation: dict | None = None
        self._last_observation: dict | None = None
        self._last_info: dict = {}
        self._done = True
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._safety_reports: list[R1ProSafetyReport] = []

    @override
    def reset(self) -> None:
        observation, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        if not self._headless:
            self._gym.render()
        self._last_raw_observation = observation
        self._last_observation = self._observation_adapter(observation)
        self._last_info = {}
        self._done = False
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._safety_reports = []

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_observation is None:
            raise RuntimeError("Observation is not set. Call reset() first.")
        return self._last_observation

    @override
    def apply_action(self, action: dict) -> None:
        """执行一次经过16维校验、URDF限位和单步限速的动作。"""
        if self._done:
            raise RuntimeError("Cannot apply an action after the episode is complete")
        if self._last_raw_observation is None:
            raise RuntimeError("Observation is not set. Call reset() first.")
        candidate = self._action_adapter(action)
        current = self._observation_adapter.state_from_observation(self._last_raw_observation)
        safe_action, report = self._safety.filter(candidate, current)
        self._safety_reports.append(report)
        observation, reward, terminated, truncated, info = self._gym.step(safe_action)
        if not self._headless:
            self._gym.render()
        self._last_raw_observation = observation
        self._last_observation = self._observation_adapter(observation)
        self._episode_steps += 1
        self._episode_reward += float(reward)
        self._last_info = {**(info or {}), "safe_action": safe_action.copy(), "safety": report.as_dict()}
        self._done = bool(terminated or truncated or self._episode_steps >= self._max_episode_steps)

    @property
    def last_info(self) -> dict:
        return self._last_info

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    @property
    def episode_reward(self) -> float:
        return self._episode_reward

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def safety_reports(self) -> tuple[R1ProSafetyReport, ...]:
        return tuple(self._safety_reports)

    def close(self) -> None:
        self._gym.close()


__all__ = ["GalaxeaR1ProSimEnvironment"]
