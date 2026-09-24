"""校验 OpenPI 服务端与 Galaxea 客户端之间的动作块契约。

OpenPI 的通用 action-chunk 机制可以适配不同长度、不同维度的动作；
Galaxea R1 闭环则需要更严格的边界检查，确保加载的 checkpoint 真正
输出 15x14 的 R1 动作，而不是把其它任务的 50 步动作静默送进仿真器。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from openpi_client import base_policy

from .contract import ACTION_DIM, EXECUTE_ACTION_HORIZON, POLICY_ACTION_HORIZON


def validate_galaxea_policy_metadata(metadata: Mapping[str, object]) -> None:
    """验证 WebSocket 建连时服务端公布的机器人和控制协议。"""

    expected = {
        "robot": "r1",
        "controller": "bimanual_joint_position",
        "action_dim": ACTION_DIM,
        "action_horizon": POLICY_ACTION_HORIZON,
        "execute_horizon": EXECUTE_ACTION_HORIZON,
    }
    missing = [key for key in expected if key not in metadata]
    if missing:
        raise ValueError(
            "Policy server metadata is missing the Galaxea contract fields "
            f"{missing}; start the server with a Galaxea R1 config"
        )
    mismatches = {
        key: (metadata[key], value)
        for key, value in expected.items()
        if metadata[key] != value
    }
    if mismatches:
        raise ValueError(f"Policy server metadata does not match the R1 contract: {mismatches}")


class ActionChunkContractPolicy(base_policy.BasePolicy):
    """只有返回精确 R1 动作块的策略才允许进入 Broker。

    Fail closed unless a policy returns the exact R1 action chunk.

    ``ActionChunkBroker`` is intentionally generic and can consume a chunk
    longer than the requested execution horizon.  The Galaxea adapter has a
    stricter project contract: π0.5 must return ``(15, 14)`` raw R1 actions.
    Keeping this check at the client/server boundary prevents an accidentally
    loaded 50-step checkpoint from being silently evaluated as a 15-step one.
    """

    def __init__(
        self,
        policy: base_policy.BasePolicy,
        *,
        action_horizon: int = POLICY_ACTION_HORIZON,
        action_dim: int = ACTION_DIM,
    ) -> None:
        """包装底层策略，并固定允许的动作块尺寸。"""
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        self._policy = policy
        self._action_horizon = action_horizon
        self._action_dim = action_dim

    def infer(self, obs: dict) -> dict:
        """推理一次并把列表形式动作规范化成 NumPy 数组。"""
        result = self._policy.infer(obs)
        if not isinstance(result, dict) or "actions" not in result:
            raise ValueError("policy response must be a dictionary containing 'actions'")

        actions = np.asarray(result["actions"], dtype=np.float32)
        expected_shape = (self._action_horizon, self._action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                "Galaxea policy must return an action chunk with shape "
                f"{expected_shape}, got {actions.shape}. "
                "Check that the server uses action_horizon=15."
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Galaxea policy returned NaN/Inf actions")
        # Keep the broker's wire contract explicit even if a test/dummy policy
        # returned a Python list. The broker requires an ndarray before slicing.
        normalized_result = dict(result)
        normalized_result["actions"] = actions
        return normalized_result

    def reset(self) -> None:
        """清空底层策略的 episode 状态。"""
        self._policy.reset()


__all__ = ["ActionChunkContractPolicy", "validate_galaxea_policy_metadata"]
