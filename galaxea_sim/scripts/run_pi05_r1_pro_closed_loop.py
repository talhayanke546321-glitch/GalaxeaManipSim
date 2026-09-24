"""运行 Galaxea R1 Pro 的 OpenPI π0.5 仿真闭环。

模型在独立 GPU 进程中通过 WebSocket 提供策略；本进程创建 R1 Pro
Gym/SAPIEN 任务，把三路图像和16维状态发送给策略，并把15步动作块逐步
执行。服务端 metadata 必须明确声明 ``r1_pro`` 和15x16协议，否则闭环在
进入仿真控制器前直接拒绝连接。
"""

from __future__ import annotations

import dataclasses
import logging

import tyro
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
from openpi_client.runtime import runtime
from openpi_client.runtime.agents import policy_agent

from galaxea_sim.integrations.openpi.r1_pro_chunk_contract import (
    R1ProActionChunkContractPolicy,
    validate_r1_pro_policy_metadata,
)
from galaxea_sim.integrations.openpi.r1_pro_contract import EXECUTE_ACTION_HORIZON
from galaxea_sim.integrations.openpi.r1_pro_environment import GalaxeaR1ProSimEnvironment


@dataclasses.dataclass
class Args:
    """R1 Pro任务、策略地址、动作块和安全边界参数。"""

    env_name: str = "R1ProDualBottlesPickEasy-v0"
    seed: int = 0
    num_episodes: int = 1
    max_episode_steps: int | None = None
    control_freq: int = 15
    headless: bool = True
    ray_tracing: bool = False

    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    api_key: str | None = None
    execute_horizon: int = EXECUTE_ACTION_HORIZON

    max_joint_delta: float = 0.12
    max_gripper_delta: float = 0.01
    prompt: str | None = None


def main(args: Args) -> None:
    """装配 R1 Pro 环境、WebSocket策略、动作块缓存和 Runtime。"""
    if args.execute_horizon != EXECUTE_ACTION_HORIZON:
        raise ValueError(
            f"R1 Pro v1 protocol requires execute_horizon={EXECUTE_ACTION_HORIZON}"
        )
    environment = GalaxeaR1ProSimEnvironment(
        env_name=args.env_name,
        seed=args.seed,
        control_freq=args.control_freq,
        headless=args.headless,
        ray_tracing=args.ray_tracing,
        max_episode_steps=args.max_episode_steps,
        max_joint_delta=args.max_joint_delta,
        max_gripper_delta=args.max_gripper_delta,
        default_prompt=args.prompt,
    )
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
            api_key=args.api_key,
        )
        metadata = client.get_server_metadata()
        logging.info("R1 Pro policy server metadata: %s", metadata)
        validate_r1_pro_policy_metadata(metadata)
        chunk_policy = R1ProActionChunkContractPolicy(client)
        broker = action_chunk_broker.ActionChunkBroker(
            chunk_policy,
            action_horizon=args.execute_horizon,
        )
        loop = runtime.Runtime(
            environment=environment,
            agent=policy_agent.PolicyAgent(broker),
            subscribers=[],
            max_hz=args.control_freq,
            num_episodes=args.num_episodes,
            max_episode_steps=environment.max_episode_steps,
        )
        loop.run()
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
