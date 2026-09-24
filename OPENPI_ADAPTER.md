# Galaxea R1 ↔ OpenPI π0.5 多资产适配

## 已验证范围

适配对象是标准 R1、`bimanual_joint_position` 控制器、15 Hz、14 维状态与动作：

```text
[left_arm_joint_1..6, left_gripper,
 right_arm_joint_1..6, right_gripper]
```

`R1_MULTI_ASSET_V1` 包含八类已跑通专家闭环的任务：

```text
R1DiverseBottlesPick-v0
R1ShoePlace-v0
R1ContainerPlace-v0
R1EmptyCupPlace-v0
R1MugHangingEasy-v0
R1PutAppleCabinet-v0
R1ToolAdjust-v0
R1BlockHammerBeat-v0
```

seed=0 的专家验证结果：

| 任务 | 成功步数 | 规划失败 |
| --- | ---: | ---: |
| Diverse bottles | 113 | 0 |
| Shoe place | 188 | 0 |
| Container place | 123 | 0 |
| Empty cup place | 117 | 0 |
| Mug hanging easy | 323 | 0 |
| Put apple cabinet | 167 | 0 |
| Tool adjust | 108 | 0 |
| Block hammer beat | 119 | 0 |

套件定义在 `galaxea_sim/integrations/openpi/task_suite.py`。`R1DualShoesPlace-v0`
暂未纳入：右臂最后两个放置规划步骤会失败；同一鞋资产的单鞋任务正常，因此不是
资产整体不可用。

## 数据和 prompt

在线观察由三路 224×224 RGB、14 维状态和环境的 `language_instruction` 组成。
离线转换器同样使用等比例缩放加黑边，不再把原图拉伸成正方形。

多任务转换时，每个 HDF5 episode 的 `language_instruction` 会写入 LeRobot 的
`task` 字段。OpenPI 配置 `pi05_galaxea_r1_multitask` 使用
`prompt_from_task=True`，因此训练和在线推理使用相同自然语言。

HDF5 记录采用 `pre_action_v1`：每一帧是执行动作前的 `state_t`，并在同一帧写入将要
执行的 `action_t`。记录还固定 `camera_resolution_scale=4`，与在线推理一致。OpenPI
专用转换器会检查这些标记以及 `bimanual_joint_position` 控制器；没有标记的旧轨迹会被
拒绝，必须重新采集/回放，不能静默转换成训练数据。

当前正式重建数据位于：

```text
datasets_multi_asset_v2/*/expert_pre_action_v1/demo_*.h5
datasets/galaxea_r1_multi_asset_v1/lerobot
```

当前数据为八类任务各 50 条，共 400 episodes / 62,252 frames；每个任务的自然语言指令
保存在 LeRobot 的 `task` 字段。此前没有时序契约标记的历史 HDF5/LeRobot 文件已移入回收站，
不能直接用于新的 OpenPI 转换/训练。LeRobot 数据必须
使用 OpenPI 环境固定的依赖版本生成；仿真环境的 `datasets==5.0.1` 输出不能被
OpenPI 的 `datasets==3.6.0` 读取。

## 变换与 stats

数据流顺序是：

```text
字段拼接
-> GalaxeaInputs（夹爪 0.00/0.05 m 转成模型 1/0）
-> DeltaActions（仅 12 个手臂关节，夹爪保持绝对值）
-> Normalize
-> π0.5
```

因此 stats 必须从最终正式多任务数据集计算，并且是在夹爪转换、关节动作增量化之后
统计。当前 stats 已从上述 400 条新数据全量计算，输出到：

```text
/home/vipuser/robotics/openpi/assets/pi05_galaxea_r1_multitask/galaxea_r1_multi_asset_v1/
```

如重新采集数据，使用下面的命令重算：

```bash
cd /home/vipuser/robotics/openpi
export GALAXEA_MULTITASK_LEROBOT_ROOT=/home/vipuser/robotics/GalaxeaManipSim/datasets/galaxea_r1_multi_asset_v1/lerobot
uv run scripts/compute_norm_stats.py --config-name pi05_galaxea_r1_multitask
```

## 环境与闭环

不需要额外执行 `activate_robotics.sh`。仿真端使用：

```bash
conda activate galaxea-sim
```

OpenPI 端在仓库内使用 `uv run`。有正式 stats 和 checkpoint 后启动服务：

```bash
cd /home/vipuser/robotics/openpi
export GALAXEA_MULTITASK_LEROBOT_ROOT=/home/vipuser/robotics/GalaxeaManipSim/datasets/galaxea_r1_multi_asset_v1/lerobot
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_galaxea_r1_multitask \
  --policy.dir=/path/to/checkpoint
```

客户端按任务启动，例如：

```bash
cd /home/vipuser/robotics/GalaxeaManipSim
conda activate galaxea-sim
python -m galaxea_sim.scripts.run_pi05_r1_closed_loop \
  --env-name R1ToolAdjust-v0 \
  --policy-host 127.0.0.1 \
  --policy-port 8000
```

默认 episode 时限来自各 Gym 任务注册值（目前 150–450），不再统一硬编码 200。
π0.5 模型输出 15 步 action chunk；客户端执行前 10 步后重新观测并请求下一段。
`action_horizon=15`（模型输出长度）和 `num_steps=10`（flow-matching 求解迭代数）不是同一个参数；本项目的
`execute_horizon=10` 是闭环执行长度。

旧的 `upright_bottles_v1_lora_8k/7999` checkpoint 没有保存协议元数据，且它的内嵌 action 统计量是按旧的
50 步 horizon 计算的，不能当作当前 15/10 协议的正确 checkpoint。重新计算 stats 后需要重新训练；客户端应
拒绝没有匹配协议元数据的服务端。

## 资源说明

`pi05_base` 和 LoRA 训练/推理的显存、内存需求取决于当前云服务器配置。
不要把旧机器上的 OOM 结论当作当前环境结论；启动训练或服务前，应先在目标环境检查
可用显存、内存和 checkpoint 类型。仿真、专家数据、转换和数据加载不依赖完整模型恢复。
