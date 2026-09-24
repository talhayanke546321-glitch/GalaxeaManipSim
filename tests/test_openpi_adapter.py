from pathlib import Path

import numpy as np

from galaxea_sim.integrations.openpi.action_adapter import GalaxeaActionAdapter
from galaxea_sim.integrations.openpi.chunk_contract import (
    ActionChunkContractPolicy,
    validate_galaxea_policy_metadata,
)
from galaxea_sim.integrations.openpi.contract import pack_r1_state
from galaxea_sim.integrations.openpi.observation_adapter import GalaxeaObservationAdapter
from galaxea_sim.integrations.openpi.safety import make_r1_safety_filter
from galaxea_sim.utils.image_utils import prepare_rgb_image


ROOT = Path(__file__).resolve().parents[1]
R1_URDF = ROOT / "galaxea_sim" / "assets" / "r1" / "robot.urdf"


def _observation() -> dict:
    image = np.zeros((18, 24, 3), dtype=np.uint8)
    return {
        "upper_body_observations": {
            "rgb_head": image,
            "rgb_left_hand": image,
            "rgb_right_hand": image,
            "left_arm_joint_position": np.arange(6, dtype=np.float32),
            "left_arm_gripper_position": np.asarray([0.05], dtype=np.float32),
            "right_arm_joint_position": np.arange(6, 12, dtype=np.float32),
            "right_arm_gripper_position": np.asarray([0.0], dtype=np.float32),
        },
        "language_instruction": "pick up the two bottles simultaneously",
    }


def test_pack_r1_state_order() -> None:
    result = pack_r1_state(np.arange(6), [0.05], np.arange(6, 12), [0.0])
    np.testing.assert_allclose(result, np.asarray([*range(6), 0.05, *range(6, 12), 0.0], dtype=np.float32))


def test_observation_adapter_returns_online_contract() -> None:
    result = GalaxeaObservationAdapter()(_observation())
    assert result["state"].shape == (14,)
    assert result["images"]["cam_high"].shape == (224, 224, 3)
    assert result["images"]["cam_high"].dtype == np.uint8
    assert result["prompt"] == "pick up the two bottles simultaneously"


def test_image_preprocessing_matches_aspect_padding_contract() -> None:
    image = np.ones((10, 20, 4), dtype=np.uint8) * 255
    result = prepare_rgb_image(image, 224, 224, name="test")
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[:56], 0)
    np.testing.assert_array_equal(result[56:168], 255)
    np.testing.assert_array_equal(result[168:], 0)


def test_action_adapter_accepts_single_action_only() -> None:
    action = np.arange(14, dtype=np.float32)
    np.testing.assert_array_equal(GalaxeaActionAdapter()({"actions": action}), action)


class _Policy:
    def __init__(self, result):
        self.result = result

    def infer(self, observation):
        return self.result

    def reset(self):
        pass


def test_action_chunk_contract_requires_15_by_14() -> None:
    policy = ActionChunkContractPolicy(
        _Policy({"actions": np.zeros((15, 14), dtype=np.float32)})
    )
    result = policy.infer({})
    assert result["actions"].shape == (15, 14)

    list_policy = ActionChunkContractPolicy(
        _Policy({"actions": np.zeros((15, 14), dtype=np.float32).tolist()})
    )
    assert isinstance(list_policy.infer({})["actions"], np.ndarray)

    bad_policy = ActionChunkContractPolicy(
        _Policy({"actions": np.zeros((50, 14), dtype=np.float32)})
    )
    try:
        bad_policy.infer({})
    except ValueError as error:
        assert "action_horizon=15" in str(error)
    else:  # pragma: no cover - assertion is the test
        raise AssertionError("a 50-step chunk must not pass the 15-step contract")


def test_policy_metadata_contract_is_explicit() -> None:
    validate_galaxea_policy_metadata(
        {
            "robot": "r1",
            "controller": "bimanual_joint_position",
            "action_dim": 14,
            "action_horizon": 15,
            "execute_horizon": 10,
        }
    )


def test_safety_filter_limits_rate_and_holds_on_nonfinite() -> None:
    safety = make_r1_safety_filter(R1_URDF, max_joint_delta=0.1, max_gripper_delta=0.01)
    current = np.zeros(14, dtype=np.float32)

    safe, report = safety.filter(np.full(14, 100.0, dtype=np.float32), current)
    assert report.joint_limit_clipped
    assert report.rate_limited
    np.testing.assert_allclose(safe[[0, 1, 3, 4, 5]], 0.1)
    np.testing.assert_allclose(safe[[7, 8, 10, 11, 12]], 0.1)
    assert safe[2] == 0.0 and safe[9] == 0.0
    np.testing.assert_allclose(safe[6], 0.01)
    np.testing.assert_allclose(safe[13], 0.01)

    held, report = safety.filter(np.full(14, np.nan, dtype=np.float32), safe)
    assert report.rejected and report.nonfinite
    np.testing.assert_array_equal(held, safe)


def test_safety_filter_hard_clamps_grippers() -> None:
    safety = make_r1_safety_filter(R1_URDF)
    current = np.zeros(14, dtype=np.float32)
    action = current.copy()
    action[6] = -1.0
    action[13] = 1.0
    safe, report = safety.filter(action, current)
    assert report.joint_limit_clipped
    assert 0.0 <= safe[6] <= 0.05
    assert 0.0 <= safe[13] <= 0.05
