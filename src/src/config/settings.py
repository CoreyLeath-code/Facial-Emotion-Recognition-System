"""
===============================================================================
FACIAL EMOTION RECOGNITION â€” CONFIGURATION (CV SYSTEM ONLY)

Purpose:
    Centralized configuration for model artifacts and inference parameters.

Design Principles:
    - Single source of truth
    - Environment variable overrides supported
    - No unrelated ML/LLM/RAG configuration
    - Clean separation of concerns

Future Improvements:
    - Replace static class with Pydantic BaseSettings for validation
===============================================================================
"""

import os


class Settings:
    # --------------------------------------------------------------------------
    # MODEL ARTIFACTS
    # --------------------------------------------------------------------------
    MODEL_PATH: str = os.getenv("MODEL_PATH", "artifacts/models/emotion_model.pt")

    DECISION_THRESHOLD: float = float(os.getenv("DECISION_THRESHOLD", "0.6"))
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
    ALLOWED_ORIGINS: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
        if origin.strip()
    )

    # --------------------------------------------------------------------------
    # API SERVER CONFIG
    # --------------------------------------------------------------------------
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    def validate(self) -> None:
        if not 0.0 <= self.DECISION_THRESHOLD <= 1.0:
            raise ValueError("DECISION_THRESHOLD must be between 0 and 1")
        if self.MAX_UPLOAD_BYTES <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be greater than zero")
        if not 1 <= self.PORT <= 65535:
            raise ValueError("PORT must be between 1 and 65535")


settings = Settings()
settings.validate()
