# Galaxea 仿真资产映射

## 当前来源

- 来源：RoboTwin 1.0 官方数据集中的 `main_models.zip`
- SHA-256：`33d434c27252bb0d320d98ecf7df603f89daf958df5f8a64992815ef27d160ae`
- 接入目录：`galaxea_sim/assets/robotwin_models`

## 瓶子任务映射

Galaxea 的 `R1DualBottlesPickEasy/Hard` 使用以下原始资产：

| 任务对象 | 模型文件 | 元数据文件 | scale |
| --- | --- | --- | --- |
| red bottle | `001_bottles/base13.glb` | `001_bottles/model_data13.json` | `[0.132, 0.132, 0.132]` |
| green bottle | `001_bottles/base16.glb` | `001_bottles/model_data16.json` | `[0.132, 0.132, 0.132]` |

代码会从元数据读取缩放；困难任务开启 `model_z_val` 时，会根据 `extents * scale` 计算模型放置高度。当前不改写 `target_pose`、`contact_pose` 或 `trans_matrix`，以保留 RoboTwin 原始抓取坐标系。

## 多资产校验状态

- `main_models.zip`：压缩包测试通过；当前接入的是 RoboTwin 原始主资产包。
- 已实际创建并执行动力学：瓶子、鞋、容器/盘子、杯子/杯垫、杯子/杯架、
  苹果/柜体、工具、锤子/积木。
- 上述八类均已用内置专家完成至少一个 seed=0 成功 episode，并保存三相机 pilot；这些
  历史 pilot 在时序契约加固前生成，只能作资产/专家冒烟参考，不能直接作为新的 OpenPI
  训练数据。
- NVIDIA Vulkan/GLVND 已通过 Conda 激活钩子自动加载；`conda activate galaxea-sim`
  后 SAPIEN 可直接创建渲染设备，不需要额外激活脚本。
- `R1DualShoesPlace-v0` 的右臂末段规划仍失败；`R1ShoePlace-v0` 使用同一鞋资产
  可成功，因此该问题暂归类为专家规划路径问题，而不是模型锚点整体错误。
