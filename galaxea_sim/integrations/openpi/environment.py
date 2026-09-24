"""把 GalaxeaManipSim 包装成 OpenPI Runtime 可调用的环境。

OpenPI Runtime 不认识 Gym 的 ``reset/step`` 细节，它只要求环境实现：
``reset``、``get_observation``、``apply_action`` 和
``is_episode_complete``。本适配器负责在这两种接口之间转换，并在动作
真正进入 ``gym.step`` 前增加最后一道安全检查。
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from openpi_client.runtime import environment as openpi_environment
from typing_extensions import override

from galaxea_sim import ASSETS_DIR

from .action_adapter import GalaxeaActionAdapter
from .observation_adapter import GalaxeaObservationAdapter
from .safety import SafetyReport, make_r1_safety_filter


class GalaxeaSimEnvironment(openpi_environment.Environment):
    """向 OpenPI 暴露一个标准 R1 Gym 仿真环境。

    Expose a Galaxea Gym environment through the OpenPI runtime contract.

    The policy server returns raw simulator actions after its Galaxea output
    transform.  This wrapper adds a final fail-closed safety layer before
    calling ``gym.step``.
    """

    def __init__(
        self,
        env_name: str = "R1DualBottlesPickEasy-v0",
        *,
        seed: int = 0,
        control_freq: int = 15,
        headless: bool = True,
        ray_tracing: bool = False,
        max_episode_steps: int | None = None,
        max_joint_delta: float = 0.12,
        max_gripper_delta: float = 0.01,
        default_prompt: str | None = None,
    ) -> None:
        """创建任务、观测/动作适配器和基于 URDF 的安全过滤器。

        ``env_name`` 是任务选择的真实入口，例如
        ``R1ToolAdjust-v0``。这里刻意限制为标准 R1 + 14 维关节位置控制，
        以避免把 R1 Pro、R1 Lite 或末端位姿控制器误接到当前 OpenPI 契约。
        """
        if not env_name.startswith("R1") or "Pro" in env_name or "Lite" in env_name:
            raise ValueError("The first OpenPI adapter only supports the standard R1 environment")

        self._rng = np.random.default_rng(seed)
        self._headless = headless
        self._gym = gym.make(
            env_name,
            control_freq=control_freq,
            headless=headless,
            obs_mode="image",
            ray_tracing=ray_tracing,
            include_depth=False,
            # Match the simulator's data-collection default.  The
            # observation adapter performs the final 224px resize/pad; a
            # lower renderer scale would change image detail at inference.
            camera_resolution_scale=4,
        )
        self._env = self._gym.unwrapped
        if self._env.action_space.shape != (14,):
            raise ValueError(f"R1 joint-position adapter requires a 14-D action space, got {self._env.action_space}")
        if getattr(self._env, "controller_type", None) != "bimanual_joint_position":
            raise ValueError("R1 OpenPI adapter requires controller_type='bimanual_joint_position'")

        self._observation_adapter = GalaxeaObservationAdapter(default_prompt=default_prompt or env_name)
        self._action_adapter = GalaxeaActionAdapter()
        urdf_path = Path(ASSETS_DIR) / self._env.robot.urdf_path
        self._safety = make_r1_safety_filter(
            urdf_path,
            max_joint_delta=max_joint_delta,
            max_gripper_delta=max_gripper_delta,
        )
        registered_limit = getattr(self._gym.spec, "max_episode_steps", None)
        if max_episode_steps is None:
            if registered_limit is None:
                raise ValueError(f"{env_name} has no registered episode limit; pass max_episode_steps")
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
        self._safety_reports: list[SafetyReport] = []

    @override
    def reset(self) -> None:
        """重置 Gym 任务，并缓存转换后的第一帧观测。"""
        gym_observation, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        if not self._headless:
            self._gym.render()
        self._last_raw_observation = gym_observation
        self._last_observation = self._observation_adapter(gym_observation)
        self._last_info = {}
        self._done = False
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._safety_reports = []

    @override
    def is_episode_complete(self) -> bool:
        """返回任务成功、截断或达到最大步数后的完成状态。"""
        return self._done

    @override
    def get_observation(self) -> dict:
        """返回当前已经适配成 OpenPI 格式的观测。"""
        if self._last_observation is None:
            raise RuntimeError("Observation is not set. Call reset() first.")
        return self._last_observation

    @override
    def apply_action(self, action: dict) -> None:
        """校验、限幅并执行一个单步动作。

        这里是安全边界：服务端输出即使形状正确，也必须经过 finite 检查、
        URDF 关节限位和每步 slew-rate 限制，之后才允许调用底层 Gym 环境。
        执行结束后会立即更新缓存的 raw observation 和 policy observation，
        所以下一轮 Runtime 读取到的是 ``state_{t+1}``。
        """
        if self._done:
            raise RuntimeError("Cannot apply an action after the episode is complete")
        if self._last_raw_observation is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        candidate = self._action_adapter(action)
        current_state = self._observation_adapter.state_from_observation(self._last_raw_observation)
        safe_action, safety_report = self._safety.filter(candidate, current_state)
        self._safety_reports.append(safety_report)

        gym_observation, reward, terminated, truncated, info = self._gym.step(safe_action)
        if not self._headless:
            self._gym.render()
        self._last_raw_observation = gym_observation
        self._last_observation = self._observation_adapter(gym_observation)
        self._episode_steps += 1
        self._episode_reward += float(reward)
        self._last_info = {
            **(info or {}),
            "safe_action": safe_action.copy(),
            "safety": safety_report.as_dict(),
        }
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
    def safety_reports(self) -> tuple[SafetyReport, ...]:
        return tuple(self._safety_reports)

    def close(self) -> None:
        """关闭底层 Gym/SAPIEN 环境。"""
        self._gym.close()


__all__ = ["GalaxeaSimEnvironment"]
