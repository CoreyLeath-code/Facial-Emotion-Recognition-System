from io import BytesIO

from PIL import Image

from src.api.validation import validate_image_upload


def test_image_validation_latency(benchmark):
    output = BytesIO()
    Image.new("RGB", (224, 224), color="white").save(output, format="JPEG")
    payload = output.getvalue()
    result = benchmark(validate_image_upload, payload, "image/jpeg", 1_000_000)
    assert result.size == (224, 224)
