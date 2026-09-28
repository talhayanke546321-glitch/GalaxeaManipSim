"""显式运行标准 R1 的 OpenPI π0.5 仿真闭环。

这个入口本身不加载模型。模型在另一个 OpenPI 进程中通过 WebSocket 提供；
本进程只负责创建 Gym/SAPIEN 任务、把观测转换成线上协议、消费动作块，
并把动作安全地送回仿真器。

每轮调用关系是：

``Runtime -> GalaxeaSimEnvironment.get_observation``
``-> PolicyAgent -> ActionChunkBroker -> WebsocketClientPolicy``
``-> GalaxeaSimEnvironment.apply_action -> gym.step``。

默认的 15/10 协议表示模型预测 15 步，但每次只执行前 10 步后重新规划。

项目的通用闭环入口 ``run_pi05_closed_loop`` 默认使用 R1 Pro；只有明确需要
兼容标准 R1 checkpoint 时才使用本模块。
"""

from __future__ import annotations

import dataclasses
import logging

import tyro
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
from openpi_client.runtime import runtime
from openpi_client.runtime.agents import policy_agent

from galaxea_sim.integrations.openpi.environment import GalaxeaSimEnvironment
from galaxea_sim.integrations.openpi.chunk_contract import (
    ActionChunkContractPolicy,
    validate_galaxea_policy_metadata,
)
from galaxea_sim.integrations.openpi.contract import (
    ACTION_DIM,
    EXECUTE_ACTION_HORIZON,
    POLICY_ACTION_HORIZON,
)
from galaxea_sim.utils.render_environment import prepare_render_environment


@dataclasses.dataclass
class Args:
    """闭环启动参数。

    ``env_name`` 选择真正的仿真任务；``policy_host/port`` 选择 OpenPI
    服务；``execute_horizon`` 控制动作块缓存多久。安全阈值只影响仿真端
    最终动作，不会改变模型输出。
    """

    env_name: str = "R1DualBottlesPickEasy-v0"
    seed: int = 0
    num_episodes: int = 1
    # None uses the task's registered Gym limit (150-450 for current R1 tasks).
    max_episode_steps: int | None = None
    control_freq: int = 15
    headless: bool = True
    gui: bool = False
    ray_tracing: bool = False

    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    api_key: str | None = None
    # The policy returns 15 actions; execute the first 10, then re-observe and
    # replan. This is the project's intended 15/10 closed-loop protocol.
    execute_horizon: int = EXECUTE_ACTION_HORIZON

    max_joint_delta: float = 0.12
    max_gripper_delta: float = 0.01
    # The simulator's language_instruction is preferred. This is only a
    # fallback for tasks that do not provide one.
    prompt: str | None = None


def main(args: Args) -> None:
    """组装仿真环境、WebSocket 客户端和 Runtime，并启动 episode。"""
    show_gui = args.gui or not args.headless
    prepare_render_environment(require_display=show_gui)
    if args.execute_horizon <= 0:
        raise ValueError("execute_horizon must be positive")
    if args.execute_horizon != EXECUTE_ACTION_HORIZON:
        raise ValueError(f"execute_horizon must be {EXECUTE_ACTION_HORIZON} for the 15/{EXECUTE_ACTION_HORIZON} protocol")

    # 任务环境在本地进程创建；OpenPI 服务端只看到适配后的图像、状态和
    # prompt，不会直接访问 SAPIEN 场景或物体真值。
    environment = GalaxeaSimEnvironment(
        env_name=args.env_name,
        seed=args.seed,
        control_freq=args.control_freq,
        headless=not show_gui,
        ray_tracing=args.ray_tracing,
        max_episode_steps=args.max_episode_steps,
        max_joint_delta=args.max_joint_delta,
        max_gripper_delta=args.max_gripper_delta,
        default_prompt=args.prompt,
    )
    # 建立长连接后先读取 server metadata。metadata 校验通过，才允许进入
    # action broker，避免把其它机器人/控制器的 checkpoint 接进 R1 仿真。
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
        api_key=args.api_key,
    )
    metadata = client.get_server_metadata()
    logging.info("Policy server metadata: %s", metadata)
    validate_galaxea_policy_metadata(metadata)
    policy = ActionChunkContractPolicy(client, action_horizon=POLICY_ACTION_HORIZON, action_dim=ACTION_DIM)
    # Broker 把服务端返回的 15x14 动作块逐步切成单步 14 维动作，并在
    # 10 步耗尽后再次请求策略。Runtime 因此无需理解 action chunk 细节。
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=args.execute_horizon)
    agent = policy_agent.PolicyAgent(broker)
    loop = runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],
        max_hz=args.control_freq,
        num_episodes=args.num_episodes,
        max_episode_steps=environment.max_episode_steps,
    )
    try:
        loop.run()
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
