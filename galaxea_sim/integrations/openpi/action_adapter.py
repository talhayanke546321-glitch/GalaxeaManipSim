"""校验 OpenPI 服务返回的单步 R1 动作。

服务端返回的是一个动作块，``ActionChunkBroker`` 会先把它切成单步动作；
本适配器只接收切分后的 14 维向量。它不重复做夹爪单位转换，因为
``GalaxeaOutputs`` 已经在 OpenPI 服务端完成了模型单位到仿真单位的转换。
"""

from __future__ import annotations

import numpy as np

from .contract import ACTION_DIM


class GalaxeaActionAdapter:
    """从策略响应中提取一个可以交给 Gym ``step`` 的动作。

    Extract a single raw R1 simulator action from a policy response.

    ``GalaxeaOutputs`` on the OpenPI side has already converted the grippers
    back to simulator metres.  This class intentionally performs validation
    only; it does not duplicate that conversion.
    """

    def __call__(self, policy_result: dict | np.ndarray) -> np.ndarray:
        """读取 ``actions`` 字段并检查形状、类型和有限值。"""
        value = policy_result["actions"] if isinstance(policy_result, dict) else policy_result
        action = np.asarray(value, dtype=np.float32)
        if action.ndim != 1 or action.shape[0] != ACTION_DIM:
            raise ValueError(f"policy action must have shape ({ACTION_DIM},), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("policy action must contain only finite values")
        return action


__all__ = ["GalaxeaActionAdapter"]
