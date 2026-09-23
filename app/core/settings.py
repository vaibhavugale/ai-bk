from fastapi import datastructures
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

ENVIRONMENT = os.getenv("ENVIRONMENT","dev")


class Settings(BaseSettings):
    APP_NAME: str = "SELCO"

    OPENSEARCH_HOST :str = "9200-01krc32prg8r3e6sd3v76vscg9.cloudspaces.litng.ai"
    OPENSEARCH_PORT :int = 443


    model_config = SettingsConfigDict(
        env_file=f".env.{ENVIRONMENT}",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()