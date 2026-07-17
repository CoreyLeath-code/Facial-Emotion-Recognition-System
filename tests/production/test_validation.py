from io import BytesIO

import pytest
from PIL import Image

from src.api.validation import validate_image_upload


def image_bytes(size=(48, 48), image_format="PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color="white").save(output, format=image_format)
    return output.getvalue()


def test_accepts_valid_png():
    image = validate_image_upload(image_bytes(), "image/png", 100_000)
    assert image.mode == "RGB"
    assert image.size == (48, 48)


@pytest.mark.parametrize("content_type", [None, "text/plain", "image/gif"])
def test_rejects_unsupported_content_type(content_type):
    with pytest.raises(ValueError, match="Only JPEG"):
        validate_image_upload(image_bytes(), content_type, 100_000)


def test_rejects_empty_payload():
    with pytest.raises(ValueError, match="empty"):
        validate_image_upload(b"", "image/png", 100)


def test_rejects_oversized_payload():
    with pytest.raises(ValueError, match="upload limit"):
        validate_image_upload(image_bytes(), "image/png", 10)


def test_rejects_corrupt_image():
    with pytest.raises(ValueError, match="invalid or corrupted"):
        validate_image_upload(b"not an image", "image/png", 100)


def test_rejects_tiny_dimensions():
    with pytest.raises(ValueError, match="at least"):
        validate_image_upload(image_bytes((8, 8)), "image/png", 100_000)


def test_rejects_decompression_bomb_dimensions():
    payload = image_bytes((5001, 5000))
    with pytest.raises(ValueError, match="too large"):
        validate_image_upload(payload, "image/png", len(payload) + 1)
