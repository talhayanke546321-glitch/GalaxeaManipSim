# R1 Pro 瓶子抓取放置任务交接

更新时间：2026-09-28（Asia/Shanghai）

本文用于把当前开发服务器上的工作交接到另一台 GPU 推理电脑。后续必须
延续现有项目，不重新创建工程。

## 1. 最终部署架构（重要）

用户明确要求：**项目不部署到机器人本体**。

```text
当前开发服务器
  代码、训练数据、checkpoint
             │ Git + rsync
             ▼
另一台 GPU 推理电脑
  ├─ OpenPI 模型服务（Python 3.11，GPU，监听 127.0.0.1:8000）
  └─ R1 Pro ROS2 客户端（Python 3.10 + ROS2 Humble）
             │
             │ ROS2 DDS，有线局域网
             ▼
Galaxea R1 Pro
  只运行原厂 HDAS、相机、关节控制和夹爪控制节点
```

模型服务与 ROS2 客户端都运行在同一台 GPU 推理电脑上，两者通过
`127.0.0.1:8000` 通信。机器人只提供 ROS2 反馈并接收 ROS2 命令，不克隆
`openpi` 或 `GalaxeaManipSim`。

这与“在机器人上运行 Python 客户端”不同。因为客户端跨局域网访问机器人的
ROS2 图，所以机器人和推理电脑必须同时设置：

```bash
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=72
```

所有机器人原厂节点和推理电脑客户端必须使用相同的 `ROS_DOMAIN_ID`，并尽量
使用相同的 `RMW_IMPLEMENTATION`。OpenPI 端口不需要对局域网开放。

## 2. Git 仓库

### GalaxeaManipSim

- 本地：`/home/vipuser/robotics/GalaxeaManipSim`
- GitHub：`git@github.com:talhayanke546321-glitch/GalaxeaManipSim.git`
- 分支：`main`
- 真机部署功能基线提交：`7a9e645`
- 详细部署说明：[`r1_pro_openpi_real_deployment.md`](r1_pro_openpi_real_deployment.md)

### openpi

- 本地：`/home/vipuser/robotics/openpi`
- GitHub：`git@github.com:talhayanke546321-glitch/openpi.git`
- 分支：`main`
- R1 Pro 服务功能基线提交：`85a5677`

另一台电脑首次拉取：

```bash
mkdir -p ~/robotics
cd ~/robotics
git clone git@github.com:talhayanke546321-glitch/GalaxeaManipSim.git
git clone git@github.com:talhayanke546321-glitch/openpi.git
```

如果仓库已存在：

```bash
cd ~/robotics/GalaxeaManipSim
git pull --ff-only origin main

cd ~/robotics/openpi
git pull --ff-only origin main
```

交接文件所在的提交会晚于上面两个功能基线提交；拉取后用下面命令确认实际
最新提交：

```bash
git log -1 --oneline
```

## 3. 任务和数据契约

- 环境：`R1ProBottlePickPlace-v0`
- 文本指令：`pick up the bottle and place it on the plate`
- 控制器：双臂绝对关节位置控制
- 控制频率：15Hz
- 状态/动作：16维
- 顺序：左臂7 + 左夹爪1 + 右臂7 + 右夹爪1
- 模型预测块：15×16
- 每次实际消费：前10步，然后重新推理
- 图片：头部、左腕、右腕三路 RGB HWC `uint8`，输入尺寸 224×224
- 手臂单位：弧度
- 模型夹爪：0闭合，0.05张开
- 真机夹爪：0闭合，100张开

训练初始状态在全部120条轨迹中完全一致：

```text
left arm:  [-0.4,  1.3, -0.7, -1.57,  1.3, -0.4, -0.8]
left grip:  0.0
right arm: [-0.4, -1.3,  0.7, -1.57, -1.3, -0.4,  0.8]
right grip: 0.0
```

真机执行代码会检查该初始状态；默认手臂最大偏差 `0.15rad`、模型夹爪最大
偏差 `0.005`。不满足时 execute 在发布动作前退出。

## 4. 专家轨迹

原始 HDF5 数据在当前开发服务器：

```text
/home/vipuser/robotics/GalaxeaManipSim/datasets_r1pro_bottle_place_high_v1
```

- 大小约 531MB
- 120条成功轨迹
- 每个高度20条
- 高度：`0.900, 0.950, 1.000, 1.025, 1.0375, 1.050m`
- 瓶子每个 episode 随机 XY，但保持直立
- 仿真瓶子约在机器人基坐标 `x=0.48~0.62m, y=-0.25~-0.10m`
- 盘子中心约为 `x=0.65m, y=-0.16m`
- 桌高没有覆盖1.20m，真实部署不要使用1.20m

收集脚本：

```text
galaxea_sim/scripts/collect_r1pro_bottle_place_100.sh
```

LeRobot 数据在当前开发服务器：

```text
/home/vipuser/robotics/lerobot-data/galaxea/R1ProBottlePickPlace-v0
```

- 大小约 463MB
- 120 episodes
- 8758 frames
- 15fps
- 三路图像 + 16维状态 + 16维动作

数据没有进入 Git。仅做推理时不需要复制数据；需要重新训练或改良模型时才
复制这两个目录。

## 5. LoRA 训练配置

配置名：

```text
pi05_galaxea_r1_pro_bottle_place_lora
```

实验名：

```text
r1pro_bottle_place_high_v1_bs32_10k
```

关键参数：

| 参数 | 值 |
|---|---:|
| 初始化权重 | `pi05_base` |
| batch size | 32 |
| 训练步数 | 10000 |
| 保存步数 | 6000、8000、10000 |
| seed | 42 |
| data workers | 2 |
| EMA | 关闭 |
| 优化器 | AdamW |
| AdamW b1 / b2 | 0.9 / 0.95 |
| gradient clip norm | 1.0 |
| warmup | 1000步 |
| peak LR | 2.5e-5 |
| LoRA：PaliGemma attention/FFN | rank 16，alpha 16 |
| LoRA：action expert attention/FFN | rank 32，alpha 32 |

配置使用 LoRA-only 冻结规则：只有参数路径包含 `lora` 的权重训练，基础权重
冻结。不要把已经删除的旧4000步训练恢复回来。

基础模型当前开发服务器路径：

```text
/home/vipuser/robotics/packages/pi05_base/params
```

整个 `pi05_base` 包约12GB。只做现有 checkpoint 推理时不需要复制
`pi05_base`；重新训练时必须复制，并设置 `OPENPI_PI05_BASE_PARAMS`。

## 6. Checkpoint

当前三个 checkpoint：

```text
/home/vipuser/robotics/openpi/checkpoints/pi05_galaxea_r1_pro_bottle_place_lora/r1pro_bottle_place_high_v1_bs32_10k/6000
/home/vipuser/robotics/openpi/checkpoints/pi05_galaxea_r1_pro_bottle_place_lora/r1pro_bottle_place_high_v1_bs32_10k/8000
/home/vipuser/robotics/openpi/checkpoints/pi05_galaxea_r1_pro_bottle_place_lora/r1pro_bottle_place_high_v1_bs32_10k/10000
```

每个约5.4GB，不在 Git 中。默认部署服务使用10000步。每个目录应包含：

```text
_CHECKPOINT_METADATA
assets
params
train_state
```

如果另一台电脑只负责推理，至少复制10000目录：

```bash
rsync -avP \
  /home/vipuser/robotics/openpi/checkpoints/pi05_galaxea_r1_pro_bottle_place_lora/r1pro_bottle_place_high_v1_bs32_10k/10000/ \
  <INFERENCE_USER>@<INFERENCE_IP>:~/robotics/openpi/checkpoints/pi05_galaxea_r1_pro_bottle_place_lora/r1pro_bottle_place_high_v1_bs32_10k/10000/
```

如需比较模型，再复制6000和8000。不要把 checkpoint 提交到 Git。

## 7. 30种子仿真结果

| checkpoint | 成功 | 碰撞episode | 越界计数 |
|---|---:|---:|---:|
| 6000 | 2/30（6.67%） | 17 | 56 |
| 8000 | 1/30（3.33%） | 17 | 218 |
| 10000 | 1/30（3.33%） | 20 | 5 |

结果文件：

```text
evaluations/pi05_r1pro_bottle_place_30seeds_bs32_6k_gui.json
evaluations/pi05_r1pro_bottle_place_30seeds_bs32_8k_gui.json
evaluations/pi05_r1pro_bottle_place_30seeds_bs32_10k.json
```

结论：三个模型效果都不足以直接驱动真机。6000成功数略高，但仍只有2/30；
10000只是当前默认服务目标，不代表它已通过效果验证。

## 8. 已实现的部署代码

GalaxeaManipSim：

```text
galaxea_sim/integrations/openpi/r1_pro_real_adapter.py
galaxea_sim/integrations/openpi/r1_pro_ros2_bridge.py
galaxea_sim/scripts/run_pi05_r1_pro_real.py
tests/test_openpi_r1_pro_real_adapter.py
docs/r1_pro_openpi_real_deployment.md
```

功能包括：

- `preflight`：只订阅，不连接模型，不创建动作发布器
- `shadow`：连接模型并记录预测，不创建动作发布器
- `execute`：双重明确确认后才创建动作发布器
- 三路图像预处理与16维状态拼接
- 真机夹爪0~100和模型0~0.05双向转换
- URDF关节限位、单步限速、左臂保持
- 观测陈旧/跨流偏差检查
- WebSocket真实接收超时
- 初始姿态门控
- 异常时单次发布当前测量位置保持

openpi：

```text
scripts/serve_galaxea_r1_pro_bottle.py
scripts/smoke_test_galaxea_r1_pro_server.py
deploy/galaxea-r1-pro-bottle-openpi.service.example
packages/openpi-client/src/openpi_client/websocket_client_policy.py
```

默认服务 checkpoint 是10000步，服务预热后才监听端口，API key 必须存在。

## 9. 已完成验证

- GalaxeaManipSim 全量测试：23 passed
- openpi-client 测试：25 passed
- R1 Pro 服务端相关测试：7 passed
- 10000 checkpoint 文件布局验证通过
- 轻量导入确认不会加载 SAPIEN
- 没有运行 GPU 模型冒烟测试：用户说明显卡尚未更换
- 没有进行真实 ROS2 网络、相机或动作测试

## 10. 推理电脑需要两个 Python 环境

### 环境A：OpenPI 模型服务

- Python 3.11+
- `~/robotics/openpi/.venv`
- 使用 GPU/JAX
- 只监听 `127.0.0.1:8000`

```bash
cd ~/robotics/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 环境B：ROS2 真机客户端

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- `--system-site-packages`，以便使用系统 `rclpy`
- 不加载模型，不需要 PyTorch/JAX

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages ~/venvs/r1pro-ros-client
source ~/venvs/r1pro-ros-client/bin/activate
python -m pip install 'numpy<2' pillow opencv-python tyro websockets msgpack
python -m pip install -e ~/robotics/openpi/packages/openpi-client
```

验证客户端环境：

```bash
source /opt/ros/humble/setup.bash
source ~/venvs/r1pro-ros-client/bin/activate
python - <<'PY'
import cv2
import rclpy
import openpi_client
print("ROS2 client environment OK")
PY
```

## 11. 后续执行顺序

### A. 新显卡安装后

```bash
nvidia-smi
cd ~/robotics/openpi
uv run python -c 'import jax; print(jax.devices())'
```

目标硬件是4090系列48GB显存；目前尚未在目标推理电脑验证。JAX 必须显示
CUDA device 后才能继续。

### B. 启动本地模型服务

```bash
cd ~/robotics/openpi
mkdir -p ~/.config/openpi
umask 077
test -s ~/.config/openpi/r1pro_api_key || openssl rand -hex 32 > ~/.config/openpi/r1pro_api_key

export OPENPI_API_KEY="$(< ~/.config/openpi/r1pro_api_key)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/serve_galaxea_r1_pro_bottle.py \
  --host 127.0.0.1 \
  --port 8000
```

另开终端执行 GPU 冒烟测试：

```bash
cd ~/robotics/openpi
export OPENPI_API_KEY="$(< ~/.config/openpi/r1pro_api_key)"
uv run scripts/smoke_test_galaxea_r1_pro_server.py \
  --host 127.0.0.1 \
  --port 8000
```

预期输出动作形状为 `(15, 16)`。

### C. 机器人只启动原厂节点

在 R1 Pro 上启动原厂：

- HDAS R1 Pro driver
- 头部和左右腕相机
- `r1_pro_jointTrackerdemo.py`
- R1 Pro gripper controller

不要在机器人上克隆本项目。启动前确保机器人和推理电脑都使用：

```bash
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=72
```

具体 launch 文件名以机器人已安装 SDK 版本为准。

### D. 在推理电脑验证远程 ROS2 图

```bash
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=72
ros2 topic list | sort
```

必须看到：

```text
/hdas/feedback_arm_left
/hdas/feedback_arm_right
/hdas/feedback_gripper_left
/hdas/feedback_gripper_right
/hdas/camera_head/left_raw/image_raw_color/compressed
/hdas/camera_wrist_left/color/image_raw/compressed
/hdas/camera_wrist_right/color/image_raw/compressed
```

如果看不到，不要启动客户端。先检查同网段、有线网络、防火墙、DDS multicast、
`ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION`。

### E. 在推理电脑运行 preflight

```bash
cd ~/robotics/GalaxeaManipSim
source /opt/ros/humble/setup.bash
source ~/venvs/r1pro-ros-client/bin/activate
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=72
export GALAXEA_SKIP_SIM_REGISTRATION=1
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python -m galaxea_sim.scripts.run_pi05_r1_pro_real --mode preflight
```

检查输出目录中的三张 PNG、流时间、夹爪范围、关节顺序和训练初始状态。尤其
必须通过图片确认左右腕相机没有互换。

### F. 在推理电脑运行 shadow

```bash
export OPENPI_API_KEY="$(< ~/.config/openpi/r1pro_api_key)"

python -m galaxea_sim.scripts.run_pi05_r1_pro_real \
  --mode shadow \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --max-steps 300
```

shadow 不创建动作发布器。检查 `steps.jsonl` 中没有 NaN/Inf、reject、异常关节
裁剪或超过2秒的推理。

### G. 当前停止点

当前 checkpoint 仿真成功率过低，完成 shadow 后应停止，不进入 execute。
建议先把仿真成功率提升到至少24/30，并消除桌面碰撞和越界，再讨论5步真机
短动作试验。

## 12. ROS2 话题

订阅：

```text
/hdas/feedback_arm_left
/hdas/feedback_arm_right
/hdas/feedback_gripper_left
/hdas/feedback_gripper_right
/hdas/camera_head/left_raw/image_raw_color/compressed
/hdas/camera_wrist_left/color/image_raw/compressed
/hdas/camera_wrist_right/color/image_raw/compressed
```

execute 模式发布：

```text
/motion_target/target_joint_state_arm_left
/motion_target/target_joint_state_arm_right
/motion_target/target_position_gripper_left
/motion_target/target_position_gripper_right
```

消息类型均为标准 `sensor_msgs/msg/JointState` 或
`sensor_msgs/msg/CompressedImage`。官方话题依据：

- <https://docs.galaxea-ai.com/Guide/R1Pro/software_introduction/R1Pro_Software_Guide_ROS2/>
- <https://github.com/OpenGalaxea/GalaxeaVLA/tree/main/experiments/r1pro>

## 13. 安全状态

策略 metadata 明确为：

```text
training_domain=simulation
real_robot_validated=false
```

execute 需要同时提供：

```text
--allow-sim-policy-on-real-robot
--execution-confirmation R1PRO_SIM_POLICY_EXECUTION_ACKNOWLEDGED
```

存在这些参数不代表当前模型适合真机；它们只是防止误启动。当前不要使用。

## 14. 尚未完成/必须在现场确认

- 新4090 48GB显卡及驱动识别
- 10000 checkpoint 在目标电脑上的实际加载和推理延迟
- 推理电脑与 R1 Pro 的 DDS 跨机发现
- 机器人 SDK 的准确版本和 launch 文件名
- 三路真实相机是否存在、左右映射、色彩和方向
- 真机关节顺序/符号是否与仿真完全一致
- 真机夹爪反馈是否严格为0~100
- 训练初始姿态在真机上是否安全且可达
- 真实桌面、瓶子和盘子的视觉域差距
- preflight 和 shadow 现场验收
- 改良模型并重新完成30种子评估

## 15. 给下一台电脑/下一次 Codex 会话的起始指令

可直接发送：

```text
继续 /home/<用户>/robotics/GalaxeaManipSim/docs/
r1_pro_bottle_handoff_2026-09-28.md 中的 R1 Pro 瓶子抓取任务。
项目和 ROS2 客户端都运行在这台 GPU 推理电脑，机器人只运行原厂 SDK，
不要把项目部署到机器人。先检查 Git、checkpoint、NVIDIA/JAX，再做本地
模型冒烟测试、跨机 ROS2 topic 检查、preflight 和 shadow。当前不允许
execute，除非我另行明确要求且安全条件满足。
```
