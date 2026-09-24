from pathlib import Path

import numpy as np
import pytest

from galaxea_sim.integrations.openpi.r1_pro_action_adapter import GalaxeaR1ProActionAdapter
from galaxea_sim.integrations.openpi.r1_pro_chunk_contract import (
    R1ProActionChunkContractPolicy,
    validate_r1_pro_policy_metadata,
)
from galaxea_sim.integrations.openpi.r1_pro_contract import STATE_ORDER, pack_r1_pro_state
from galaxea_sim.integrations.openpi.r1_pro_observation_adapter import GalaxeaR1ProObservationAdapter
from galaxea_sim.integrations.openpi.r1_pro_safety import make_r1_pro_safety_filter


ROOT = Path(__file__).resolve().parents[1]
R1_PRO_URDF = ROOT / "galaxea_sim" / "assets" / "r1_pro" / "robot.urdf"


def _observation() -> dict:
    image = np.zeros((18, 24, 3), dtype=np.uint8)
    return {
        "upper_body_observations": {
            "rgb_head": image,
            "rgb_left_hand": image,
            "rgb_right_hand": image,
            "left_arm_joint_position": np.arange(7, dtype=np.float32),
            "left_arm_gripper_position": np.asarray([0.05], dtype=np.float32),
            "right_arm_joint_position": np.arange(7, 14, dtype=np.float32),
            "right_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
        },
        "language_instruction": "stack the blocks",
    }


def test_pack_and_observation_use_16d_order() -> None:
    packed = pack_r1_pro_state(np.arange(7), [0.05], np.arange(7, 14), [0.0])
    assert packed.shape == (16,)
    np.testing.assert_allclose(packed[[7, 15]], [0.05, 0.0])
    result = GalaxeaR1ProObservationAdapter()(_observation())
    assert result["state"].shape == (16,)
    assert result["images"]["cam_high"].shape == (224, 224, 3)
    assert result["prompt"] == "stack the blocks"


def test_action_adapter_requires_single_16d_action() -> None:
    action = np.arange(16, dtype=np.float32)
    np.testing.assert_array_equal(GalaxeaR1ProActionAdapter()({"actions": action}), action)
    with pytest.raises(ValueError, match="16"):
        GalaxeaR1ProActionAdapter()(np.zeros(14, dtype=np.float32))


class _Policy:
    def infer(self, observation):
        return {"actions": np.zeros((15, 16), dtype=np.float32)}

    def reset(self):
        pass


def test_chunk_and_metadata_reject_standard_r1_contract() -> None:
    result = R1ProActionChunkContractPolicy(_Policy()).infer({})
    assert result["actions"].shape == (15, 16)
    metadata = {
        "protocol_version": "galaxea-r1-pro-openpi-v1",
        "robot": "r1_pro",
        "controller": "bimanual_joint_position",
        "action_dim": 16,
        "action_horizon": 15,
        "execute_horizon": 10,
        "state_order": list(STATE_ORDER),
    }
    validate_r1_pro_policy_metadata(metadata)
    with pytest.raises(ValueError, match="R1 Pro"):
        validate_r1_pro_policy_metadata(dict(metadata, robot="r1", action_dim=14))


def test_r1_pro_safety_uses_14_arm_limits_and_two_grippers() -> None:
    safety = make_r1_pro_safety_filter(
        R1_PRO_URDF,
        max_joint_delta=0.1,
        max_gripper_delta=0.01,
    )
    current = np.zeros(16, dtype=np.float32)
    safe, report = safety.filter(np.full(16, 100.0, dtype=np.float32), current)
    assert report.joint_limit_clipped
    assert report.rate_limited
    assert safe.shape == (16,)
    assert 0.0 <= safe[7] <= 0.05
    assert 0.0 <= safe[15] <= 0.05

    held, report = safety.filter(np.full(16, np.nan, dtype=np.float32), safe)
    assert report.rejected and report.nonfinite
    np.testing.assert_array_equal(held, safe)
