# R1 Pro × OpenPI π0.5 仿真闭环

这套闭环使用 R1 Pro 的双7轴关节位置控制协议：三路 RGB 图像 + 16维状态
经 WebSocket 发送到 OpenPI，策略返回 `15×16` 动作块，仿真端执行前10步
后重新观测和规划。

## 1. 安全地验证闭环线路

先启动保持位姿服务。它不是 π0.5，只用于验证完整网络和控制链路，输出
始终等于当前状态，因此不会产生随机关节运动。

```bash
cd /home/vipuser/robotics/openpi
export OPENPI_API_KEY='r1pro-local-test'
export PYTHONPATH=/home/vipuser/robotics/GalaxeaManipSim:$PYTHONPATH
uv run python -m galaxea_sim.scripts.run_r1_pro_hold_policy_server --port 8000
```

另开终端启动 R1 Pro 仿真客户端：

```bash
conda activate galaxea-sim
cd /home/vipuser/robotics/GalaxeaManipSim
export PYTHONPATH="$PWD:/home/vipuser/robotics/openpi/packages/openpi-client/src:$PYTHONPATH"
python -m galaxea_sim.scripts.run_pi05_closed_loop \
  --env-name R1ProDualBottlesPickEasy-v0 \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --api-key r1pro-local-test
```

可替换的已注册任务包括 `R1ProBlocksStackEasy-v0` 和
`R1ProBlocksStackHard-v0`。策略 checkpoint 必须由相同任务分布的数据训练；
换 Gym ID 并不会自动让单任务策略获得新技能。

## 2. 从 pi05_base 训练 R1 Pro 策略

`pi05_base` 没有 R1 Pro 的16维归一化统计，不能裸部署。完整流程是：采集或
回放 R1 Pro 专家轨迹、转换 LeRobot 数据、计算统计量、微调、再启动策略服务。

### 2.1 生成关节位置控制数据

```bash
conda activate galaxea-sim
cd /home/vipuser/robotics/GalaxeaManipSim
python -m galaxea_sim.scripts.collect_demos \
  --env-name R1ProDualBottlesPickEasy-v0 \
  --num-demos 100

python -m galaxea_sim.scripts.replay_demos \
  --env-name R1ProDualBottlesPickEasy-v0 \
  --target-controller-type bimanual_joint_position \
  --num-demos 100

python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot \
  --task R1ProDualBottlesPickEasy-v0 \
  --tag replayed \
  --robot r1_pro
```

转换后的数据必须包含 metadata：`galaxea_robot=r1_pro`、`action_dim=16`、
`controller=bimanual_joint_position`、`fps=15`。配置会在读取数据前强制校验。

### 2.2 计算统计量并 LoRA 微调

```bash
cd /home/vipuser/robotics/openpi
export GALAXEA_R1PRO_LEROBOT_ROOT='/绝对路径/R1ProDualBottlesPickEasy-v0'
export OPENPI_PI05_BASE_PARAMS='/home/vipuser/robotics/openpi-data/openpi-assets/checkpoints/pi05_base/params'
export OPENPI_R1PRO_ASSETS_DIR='/home/vipuser/robotics/openpi/assets/pi05_galaxea_r1_pro'

uv run scripts/compute_norm_stats.py --config-name pi05_galaxea_r1_pro_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_galaxea_r1_pro_lora \
  --exp-name=r1pro_dual_bottles \
  --overwrite
```

如果10GB显存仍不足，应降低训练配置中的 batch size 或使用梯度累积；不要通过
缩短16维动作、复用14维 R1 stats 来规避显存问题。

## 3. 启动真实 π0.5 checkpoint

```bash
cd /home/vipuser/robotics/openpi
export OPENPI_API_KEY='替换为随机长密钥'
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_galaxea_r1_pro_lora \
  --policy.dir=checkpoints/pi05_galaxea_r1_pro_lora/r1pro_dual_bottles/30000 \
  --host=127.0.0.1 \
  --port=8000 \
  --warmup
```

随后按第1节启动默认的 `run_pi05_closed_loop`（等价于显式入口
`run_pi05_r1_pro_closed_loop`）。客户端会检查服务 metadata，
标准 R1 的 `15×14` checkpoint、错误关节顺序或其它控制器都会被拒绝。

## 4. 数据和动作契约

```text
state/action:
[left_joint1..7, left_gripper,
 right_joint1..7, right_gripper]

arm unit: radian
sim gripper: 0.00 m closed, 0.05 m open
control frequency: 15 Hz
policy horizon: 15
execute horizon: 10
images: head + left wrist + right wrist, RGB HWC uint8, 224×224
```

这份适配层针对仿真器。R1 Pro 真机夹爪是0~100行程，真机部署时必须另加
ROS2硬件映射和通信看门狗，不能把仿真的0~0.05米指令直接发布到真机话题。
