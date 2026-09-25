from pydantic import computed_field
from fastapi import datastructures
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

ENVIRONMENT = os.getenv("ENVIRONMENT","dev")


class Settings(BaseSettings):
    APP_NAME: str = "SELCO"

    OPENSEARCH_HOST :str = "9200-01krc32prg8r3e6sd3v76vscg9.cloudspaces.litng.ai"
    OPENSEARCH_PORT :int = 443

    LITNG_EMBEDDING_API:str = ""
    LITNG_EMBEDDING_API_KEY:str = ""

     # Database connection parameters
    postgres_server: str
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_port: int

    @computed_field
    @property
    def sqlalchemy_database_uri(self) -> str:
        """URI for primary (write) database"""
        return f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}?sslmode=require"
    
    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{ENVIRONMENT}"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()