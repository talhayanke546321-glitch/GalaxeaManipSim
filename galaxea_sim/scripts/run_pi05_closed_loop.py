"""默认运行 Galaxea R1 Pro 的 OpenPI π0.5 仿真闭环。

通用入口固定转发到 R1 Pro 实现，避免无参数启动时误用标准 R1 的 14 维
协议。标准 R1 兼容入口保留为 ``run_pi05_r1_closed_loop``。
"""

from __future__ import annotations

import logging

import tyro

from galaxea_sim.scripts.run_pi05_r1_pro_closed_loop import Args, main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
