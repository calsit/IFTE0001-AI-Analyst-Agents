from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError

# Project root directory (the folder that contains this settings.py)
PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env_files() -> None:
    """
    Load environment variable files in order (first match wins):
      - .env
      - API.env
      - .env.local
    """
    for name in (".env", "API.env", ".env.local"):
        p = PROJECT_ROOT / name
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


_load_env_files()


class Settings(BaseModel):
    # OpenAI API key
    OPENAI_API_KEY: SecretStr = Field(...)

    # model & runtime parameters (defaults will be used if not provided)
    OPENAI_MODEL: str = Field(default="gpt-4.1-mini")
    OPENAI_MODEL_STRONG: str = Field(default="gpt-4.1")
    OPENAI_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    OPENAI_TIMEOUT_S: int = Field(default=60, ge=5, le=600)

    # logging level
    LOG_LEVEL: str = Field(default="INFO")


def load_settings() -> Settings:
    """
    Build Settings from environment variables.

    Make sure OPENAI_API_KEY is available via:
      - .env / API.env / .env.local, or
      - system environment variables
    """
    try:
        # Pydantic will pick needed fields from env dict
        return Settings.model_validate(os.environ)
    except ValidationError as e:
        raise RuntimeError(
            "Invalid or missing environment configuration.\n"
            "Required: OPENAI_API_KEY\n"
            "Tip: create a .env file at the project root with:\n"
            '  OPENAI_API_KEY="your_key_here"\n'
            f"\nDetails:\n{e}"
        ) from e


settings = load_settings()
