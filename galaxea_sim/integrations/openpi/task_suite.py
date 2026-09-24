"""OpenPI 使用的、带版本号的标准 R1 多任务集合。

这个列表不是 Gym 注册表，而是数据重建、训练和评测时使用的“任务白名单”。
给集合加版本号可以让数据集、normalization stats 和 checkpoint 明确知道
自己对应哪一批任务，避免后来增删任务后仍误用旧统计量。
"""

from __future__ import annotations


# One task from each currently verified asset/semantic family.  Keeping this
# list versioned makes dataset statistics and checkpoints reproducible.
R1_MULTI_ASSET_V1 = (
    "R1DiverseBottlesPick-v0",
    "R1ShoePlace-v0",
    "R1ContainerPlace-v0",
    "R1EmptyCupPlace-v0",
    "R1MugHangingEasy-v0",
    "R1PutAppleCabinet-v0",
    "R1ToolAdjust-v0",
    "R1BlockHammerBeat-v0",
)


__all__ = ["R1_MULTI_ASSET_V1"]
