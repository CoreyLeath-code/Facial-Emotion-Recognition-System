"""Validation helpers for the untrusted HTTP inference boundary."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


def validate_image_upload(
    payload: bytes,
    content_type: str | None,
    max_bytes: int,
) -> Image.Image:
    """Validate type, size, decoding, and dimensions before model inference."""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError("Only JPEG, PNG, and WebP images are accepted")
    if not payload:
        raise ValueError("Image payload is empty")
    if len(payload) > max_bytes:
        raise ValueError(f"Image exceeds the {max_bytes}-byte upload limit")

    try:
        image = Image.open(BytesIO(payload))
        image.verify()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Image payload is invalid or corrupted") from exc

    width, height = image.size
    if width < 16 or height < 16:
        raise ValueError("Image dimensions must be at least 16x16 pixels")
    if width * height > 25_000_000:
        raise ValueError("Decoded image dimensions are too large")
    return image
