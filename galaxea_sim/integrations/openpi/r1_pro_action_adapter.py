"""校验 ActionChunkBroker 切出的单步 R1 Pro 仿真动作。"""

from __future__ import annotations

import numpy as np

from .r1_pro_contract import ACTION_DIM


class GalaxeaR1ProActionAdapter:
    """只允许有限的16维动作进入 R1 Pro 仿真控制器。"""

    def __call__(self, policy_result: dict | np.ndarray) -> np.ndarray:
        value = policy_result["actions"] if isinstance(policy_result, dict) else policy_result
        action = np.asarray(value, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"R1 Pro policy action must have shape ({ACTION_DIM},), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("R1 Pro policy action must contain only finite values")
        return action


__all__ = ["GalaxeaR1ProActionAdapter"]
