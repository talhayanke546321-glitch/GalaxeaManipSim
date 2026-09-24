"""Small image helpers used by simulator-side data tools.

These helpers deliberately do not import ``openpi_client``. The simulator's
demo collection and LeRobot conversion commands should remain usable in the
standalone ``galaxea-sim`` environment; OpenPI is only required by the
integration and policy-serving paths.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_PIL_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def convert_to_uint8(images: np.ndarray) -> np.ndarray:
    """Convert camera pixels to the uint8 convention expected by OpenPI.

    Camera backends normally return uint8 pixels, but a few data paths expose
    floating-point pixels in either ``[0, 1]`` or ``[0, 255]``. Handle both
    explicitly and reject non-finite values instead of allowing uint8 wraparound.
    """

    images = np.asarray(images)
    if np.issubdtype(images.dtype, np.floating):
        if not np.all(np.isfinite(images)):
            raise ValueError("images must contain only finite values")
        if images.size and float(np.max(images)) <= 1.0 + 1e-6:
            images = images * 255.0
    return np.clip(images, 0, 255).astype(np.uint8, copy=False)


def resize_with_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize HWC images (or a batch of them) without changing aspect ratio.

    This mirrors ``openpi_client.image_tools.resize_with_pad``: PIL bilinear
    interpolation, centred zero padding, and floor-based resized dimensions.
    """

    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}")
    images = np.asarray(images)
    if images.ndim < 3:
        raise ValueError(f"images must have at least 3 dimensions, got {images.shape}")
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    flat_images = images.reshape(-1, *original_shape[-3:])
    resized_images = []
    for image in flat_images:
        cur_height, cur_width, channels = image.shape
        if channels not in (1, 3, 4):
            raise ValueError(f"images must have 1, 3, or 4 channels, got {image.shape}")
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized = np.asarray(
            Image.fromarray(image).resize(
                (resized_width, resized_height), resample=_PIL_BILINEAR
            )
        )
        if channels == 1 and resized.ndim == 2:
            resized = resized[..., None]
        canvas = np.zeros((height, width, channels), dtype=resized.dtype)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        canvas[pad_height : pad_height + resized_height, pad_width : pad_width + resized_width] = resized
        resized_images.append(canvas)

    return np.stack(resized_images).reshape(*original_shape[:-3], height, width, original_shape[-1])


def prepare_rgb_image(image: object, height: int, width: int, *, name: str = "image") -> np.ndarray:
    """Apply the complete RGB preprocessing contract used online and offline."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"{name} must be an HWC RGB/RGBA image, got {array.shape}")
    array = convert_to_uint8(array[..., :3])
    result = resize_with_pad(array, height, width)
    if result.shape != (height, width, 3):
        raise ValueError(f"{name} conversion produced unexpected shape {result.shape}")
    return np.ascontiguousarray(result)


__all__ = ["convert_to_uint8", "prepare_rgb_image", "resize_with_pad"]
