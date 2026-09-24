"""为 R1 Pro 端到端闭环测试提供安全的保持位姿策略。

这不是 π0.5 模型，也不用于评估任务成功率。它经过与真实策略相同的
三图像/16维输入校验，然后把当前关节状态复制成15步动作块。用途是先验证
WebSocket、metadata、动作切片、R1 Pro 环境和安全层是否真正闭环，而不会
让随机动作驱动机器人。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from openpi.policies.galaxea_r1_pro_policy import GalaxeaR1ProInputs
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client import base_policy

from galaxea_sim.integrations.openpi.r1_pro_contract import (
    ACTION_DIM,
    CONTROL_FREQUENCY,
    EXECUTE_ACTION_HORIZON,
    POLICY_ACTION_HORIZON,
    STATE_ORDER,
)


class HoldPositionPolicy(base_policy.BasePolicy):
    """返回当前16维状态，使仿真机器人保持当前位置。"""

    def __init__(self) -> None:
        self._inputs = GalaxeaR1ProInputs()

    def infer(self, obs: dict) -> dict:
        self._inputs(obs)
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.all(np.isfinite(state)):
            raise ValueError(f"R1 Pro hold policy requires finite ({ACTION_DIM},) state")
        return {"actions": np.repeat(state[None, :], POLICY_ACTION_HORIZON, axis=0)}

    def reset(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    metadata = {
        "protocol_version": "galaxea-r1-pro-openpi-v1",
        "policy": "hold_position_smoke_test",
        "warning": "not pi05 inference; transport and simulation closed-loop test only",
        "robot": "r1_pro",
        "controller": "bimanual_joint_position",
        "action_dim": ACTION_DIM,
        "action_horizon": POLICY_ACTION_HORIZON,
        "execute_horizon": EXECUTE_ACTION_HORIZON,
        "control_frequency_hz": CONTROL_FREQUENCY,
        "state_order": list(STATE_ORDER),
    }
    server = WebsocketPolicyServer(
        policy=HoldPositionPolicy(),
        host="127.0.0.1",
        port=args.port,
        metadata=metadata,
        api_key=os.environ.get("OPENPI_API_KEY"),
    )
    print(f"R1 Pro hold-policy server listening on ws://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
