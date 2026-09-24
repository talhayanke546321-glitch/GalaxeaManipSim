"""R1 Pro WebSocket metadata和15x16动作块校验。"""

from __future__ import annotations

from collections.abc import Mapping

from .chunk_contract import ActionChunkContractPolicy
from .r1_pro_contract import ACTION_DIM, EXECUTE_ACTION_HORIZON, POLICY_ACTION_HORIZON, STATE_ORDER


def validate_r1_pro_policy_metadata(metadata: Mapping[str, object]) -> None:
    """拒绝标准 R1、其它控制器或关节顺序不一致的策略服务。"""
    expected = {
        "protocol_version": "galaxea-r1-pro-openpi-v1",
        "robot": "r1_pro",
        "controller": "bimanual_joint_position",
        "action_dim": ACTION_DIM,
        "action_horizon": POLICY_ACTION_HORIZON,
        "execute_horizon": EXECUTE_ACTION_HORIZON,
        "state_order": list(STATE_ORDER),
    }
    missing = [key for key in expected if key not in metadata]
    if missing:
        raise ValueError(f"Policy server metadata is missing R1 Pro fields: {missing}")
    mismatches = {key: (metadata[key], value) for key, value in expected.items() if metadata[key] != value}
    if mismatches:
        raise ValueError(f"Policy server metadata does not match the R1 Pro contract: {mismatches}")


class R1ProActionChunkContractPolicy(ActionChunkContractPolicy):
    """固定要求服务端每次返回15步、每步16维动作。"""

    def __init__(self, policy) -> None:
        super().__init__(policy, action_horizon=POLICY_ACTION_HORIZON, action_dim=ACTION_DIM)


__all__ = ["R1ProActionChunkContractPolicy", "validate_r1_pro_policy_metadata"]
