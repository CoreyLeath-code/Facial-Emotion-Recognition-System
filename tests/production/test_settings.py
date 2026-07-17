import pytest

from src.src.config.settings import Settings


def test_default_settings_are_valid():
    Settings().validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("DECISION_THRESHOLD", -0.1),
        ("DECISION_THRESHOLD", 1.1),
        ("MAX_UPLOAD_BYTES", 0),
        ("PORT", 0),
        ("PORT", 65536),
    ],
)
def test_invalid_settings_are_rejected(field, value):
    candidate = Settings()
    setattr(candidate, field, value)
    with pytest.raises(ValueError):
        candidate.validate()
