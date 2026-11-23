"""Decode untrusted image bytes into a BGR uint8 array."""
from __future__ import annotations

import cv2
import numpy as np


class InvalidImageError(ValueError):
    pass


class ImageTooLargeError(InvalidImageError):
    pass


def decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    if not data:
        raise InvalidImageError("Empty file")
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("File could not be decoded as an image")
    h, w = image.shape[:2]
    if h * w > max_pixels:
        raise ImageTooLargeError(f"Image is {w}x{h}; the limit is {max_pixels:,} pixels")
    return image
