"""Prepare Vulkan and X11 for SAPIEN on replaceable cloud GPUs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import subprocess


_NVIDIA_VERSION_RE = re.compile(r"Kernel Module\s+(\d+(?:\.\d+)+)")
_REPAIR_VERSION_RE = re.compile(r"nvidia-repair-(\d+(?:\.\d+)+)")


def _loaded_nvidia_version() -> str | None:
    try:
        text = Path("/proc/driver/nvidia/version").read_text()
    except OSError:
        return None
    match = _NVIDIA_VERSION_RE.search(text)
    return match.group(1) if match else None


def clear_stale_nvidia_overrides() -> list[str]:
    """Remove copied NVIDIA libraries when they do not match the live driver."""
    loaded = _loaded_nvidia_version()
    if loaded is None:
        return []

    removed: list[str] = []
    for name in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES", "__EGL_VENDOR_LIBRARY_FILENAMES"):
        value = os.environ.get(name, "")
        match = _REPAIR_VERSION_RE.search(value)
        if match and match.group(1) != loaded:
            os.environ.pop(name, None)
            removed.append(name)

    entries = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    kept: list[str] = []
    changed = False
    for entry in entries:
        match = _REPAIR_VERSION_RE.search(entry)
        if match and match.group(1) != loaded:
            changed = True
            continue
        if entry:
            kept.append(entry)
    if changed:
        if kept:
            os.environ["LD_LIBRARY_PATH"] = ":".join(kept)
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)
        removed.append("LD_LIBRARY_PATH")

    if removed:
        logging.warning(
            "Removed stale NVIDIA userspace overrides; loaded driver is %s: %s",
            loaded,
            ", ".join(removed),
        )
    return removed


def _gnome_x11_environment() -> dict[str, str] | None:
    """Return the active user's GNOME X11 environment from /proc."""
    uid = os.getuid()
    proc = Path("/proc")
    for process_dir in proc.iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            if process_dir.stat().st_uid != uid:
                continue
            if (process_dir / "comm").read_text().strip() != "gnome-shell":
                continue
            raw = (process_dir / "environ").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        values: dict[str, str] = {}
        for item in raw.split(b"\0"):
            if b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            values[key.decode(errors="replace")] = value.decode(errors="replace")
        if values.get("DISPLAY"):
            return values
    return None


def prepare_render_environment(*, require_display: bool) -> str | None:
    """Select matching Vulkan libraries and, for GUI mode, the GNOME X display."""
    clear_stale_nvidia_overrides()
    if not require_display:
        return os.environ.get("DISPLAY")

    desktop = _gnome_x11_environment()
    if desktop:
        for name in ("DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
            if desktop.get(name):
                os.environ[name] = desktop[name]

    runtime_dir = f"/run/user/{os.getuid()}"
    os.environ.setdefault("XDG_RUNTIME_DIR", runtime_dir)
    authority = Path(runtime_dir) / "gdm" / "Xauthority"
    if not os.environ.get("XAUTHORITY") and authority.is_file():
        os.environ["XAUTHORITY"] = str(authority)
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus")

    display = os.environ.get("DISPLAY")
    if not display:
        raise RuntimeError(
            "GUI requested but no active GNOME X11 display was found. "
            "Log in through NoMachine first, then rerun the command."
        )
    try:
        subprocess.run(
            ["xrandr", "--display", display, "--current"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5,
            env=os.environ,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"GUI requested but X11 display {display!r} is not usable") from exc

    logging.info("SAPIEN GUI will use X11 display %s", display)
    return display


__all__ = ["clear_stale_nvidia_overrides", "prepare_render_environment"]
