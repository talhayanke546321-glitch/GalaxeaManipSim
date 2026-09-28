"""Pre-render and smoothly display an expert trajectory on remote desktops.

SAPIEN's native ``RenderWindow`` can block on servers whose NVIDIA GPU has no
active scanout.  This viewer keeps simulation and rendering in SAPIEN, but
presents the resulting RGB frames through a regular Tk/X11 window.  Frames are
pre-rendered first so playback follows the dataset control frequency rather
than the server's image-rendering throughput.
"""

from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
import tqdm
import tyro

import galaxea_sim.envs  # noqa: F401 - registers Gym environments
from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.scripts.replay_demos import (
    _demo_index,
    _read_actions,
    _read_source_reset_info,
)
from galaxea_sim.utils.render_environment import prepare_render_environment
from galaxea_sim.utils.sapien_utils import add_camera_to_scene


def _find_source_demo(
    dataset_dir: Path,
    env_name: str,
    demo_index: int,
    source_subdir: str | None,
) -> Path:
    task_dir = dataset_dir / env_name
    if source_subdir:
        candidate = task_dir / source_subdir / f"demo_{demo_index}.h5"
        if candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for path in task_dir.glob(f"*/demo_{demo_index}.h5")
        if path.parent.name not in {"final", "replayed"}
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find demo_{demo_index}.h5 below {task_dir}"
        )
    if len(candidates) > 1:
        names = ", ".join(path.parent.name for path in candidates)
        raise ValueError(
            f"Multiple source demos match index {demo_index}: {names}. "
            "Set --source-subdir explicitly."
        )
    return candidates[0]


def _capture_rgb(camera) -> np.ndarray:
    camera.take_picture()
    rgba = camera.get_picture("Color")
    return (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)


def _display_frames(
    frames: list[np.ndarray],
    *,
    control_freq: int,
    window_scale: int,
    title: str,
    loop: bool,
    end_pause_seconds: float,
) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title(title)
    root.lift()
    root.attributes("-topmost", True)
    root.after(1500, lambda: root.attributes("-topmost", False))
    label = tk.Label(root, borderwidth=0, highlightthickness=0)
    label.pack()
    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    root.protocol("WM_DELETE_WINDOW", close)
    frame_period = 1.0 / control_freq

    try:
        while not closed:
            deadline = time.perf_counter()
            for index, frame in enumerate(frames):
                if closed:
                    break
                height, width = frame.shape[:2]
                ppm = f"P6 {width} {height} 255\n".encode() + frame.tobytes()
                image = tk.PhotoImage(data=ppm, format="PPM")
                if window_scale > 1:
                    image = image.zoom(window_scale, window_scale)
                label.configure(image=image)
                label.image = image
                root.title(f"{title}  [{index + 1}/{len(frames)}]")
                root.update_idletasks()
                root.update()

                deadline += frame_period
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline = time.perf_counter()
            if not loop:
                end_time = time.perf_counter() + end_pause_seconds
                while not closed and time.perf_counter() < end_time:
                    root.update_idletasks()
                    root.update()
                    time.sleep(0.02)
                break
    except tk.TclError:
        pass
    finally:
        if not closed:
            root.destroy()


def main(
    env_name: str = "R1DiverseBottlesPick-v0",
    dataset_dir: Path = Path("datasets_multi_asset_v2"),
    demo_index: int = 12,
    source_subdir: str | None = "expert_pre_action_v1",
    controller_type: str = "bimanual_relaxed_ik",
    control_freq: int = 15,
    width: int = 320,
    height: int = 180,
    window_scale: int = 2,
    max_steps: int | None = None,
    loop: bool = True,
    table_height: float | None = None,
    end_pause_seconds: float = 1.0,
    native_viewer: bool = False,
) -> None:
    """Show an expert demo at its original control frequency.

    Args:
        env_name: Registered Galaxea simulation environment.
        dataset_dir: Dataset root containing one directory per environment.
        demo_index: Numeric index of the source expert episode.
        source_subdir: Preferred source directory below the task directory.
        controller_type: Controller contract used by the source actions.
        control_freq: Playback frequency in Hz.
        width: Offscreen render width before integer window scaling.
        height: Offscreen render height before integer window scaling.
        window_scale: Integer display scaling applied by Tk.
        max_steps: Optional action limit for a quick check.
        loop: Replay continuously until the window is closed.
        table_height: Override the HDF5 table_height_m metadata when needed.
        end_pause_seconds: Keep the final frame visible after one-shot playback.
        native_viewer: Replay directly in SAPIEN's interactive 3D viewer.
    """
    if control_freq <= 0:
        raise ValueError("control_freq must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if window_scale <= 0:
        raise ValueError("window_scale must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if table_height is not None and not 0.70 <= table_height <= 1.10:
        raise ValueError("table_height must be between 0.70 and 1.10 meters")
    if end_pause_seconds < 0:
        raise ValueError("end_pause_seconds must be non-negative")

    prepare_render_environment(require_display=True)

    source_h5 = _find_source_demo(
        dataset_dir, env_name, demo_index, source_subdir
    )
    source_demo_index = _demo_index(source_h5)
    assert source_demo_index is not None

    import h5py

    with h5py.File(source_h5, "r") as h5_file:
        actions = _read_actions(h5_file, controller_type)
        recorded_height = h5_file.attrs.get("table_height_m")
    if table_height is None and recorded_height is not None:
        table_height = float(recorded_height)
    if max_steps is not None:
        actions = actions[:max_steps]

    env = gym.make(
        env_name,
        control_freq=control_freq,
        headless=not native_viewer,
        obs_mode="state",
        controller_type=controller_type,
        ray_tracing=False,
        include_depth=False,
        table_height_override=table_height,
    )
    assert isinstance(env.unwrapped, BimanualManipulationEnv)
    raw_env = env.unwrapped

    reset_info = _read_source_reset_info(source_h5, source_demo_index)
    if reset_info:
        env.reset(options={"reset_info": reset_info})
    else:
        env.reset()

    if native_viewer:
        print(
            f"Playing {source_h5} in the native SAPIEN viewer; "
            f"table height={table_height!r} m at {control_freq} Hz."
        )
        frame_period = 1.0 / control_freq
        env.render()
        for action in actions:
            started = time.perf_counter()
            _, _, terminated, truncated, _ = env.step(action)
            env.render()
            if raw_env.viewer is not None and raw_env.viewer.closed:
                break
            remaining = frame_period - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
            if terminated or truncated:
                break

        if raw_env.viewer is not None and not raw_env.viewer.closed:
            pause_deadline = time.perf_counter() + end_pause_seconds
            while time.perf_counter() < pause_deadline:
                env.render()
                if raw_env.viewer.closed:
                    break
                time.sleep(0.02)
        env.close()
        return

    camera_pose = sapien.Pose()
    camera_pose.set_p([1.8, -0.9, 1.0])
    camera_pose.set_rpy([0.0, 0.0, 2.64])
    camera = add_camera_to_scene(
        raw_env._scene,
        "smooth_expert_view",
        pose=camera_pose,
        width=width,
        height=height,
    )

    frames: list[np.ndarray] = []
    raw_env._scene.update_render()
    frames.append(_capture_rgb(camera))
    for action in tqdm.tqdm(actions, desc="Pre-rendering expert trajectory"):
        _, _, terminated, truncated, _ = env.step(action)
        frames.append(_capture_rgb(camera))
        if terminated or truncated:
            break

    env.close()
    print(
        f"Pre-rendered {len(frames)} frames from {source_h5}; "
        f"table height={table_height!r} m, playing at {control_freq} Hz."
    )
    height_label = "default" if table_height is None else f"{table_height:.4f} m"
    _display_frames(
        frames,
        control_freq=control_freq,
        window_scale=window_scale,
        title=f"SAPIEN expert: {height_label} / demo_{demo_index}",
        loop=loop,
        end_pause_seconds=end_pause_seconds,
    )


if __name__ == "__main__":
    tyro.cli(main)
