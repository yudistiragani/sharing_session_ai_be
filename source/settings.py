from pydantic_settings import BaseSettings, SettingsConfigDict


class MasterSettings(BaseSettings):
    # AGENT
    LLM_API_KEY: str
    LLM_BASE_URL: str
    EMBEDDING_BASE_URL: str
    VECTOR_DB: str

    # REDIS
    REDIS_HOST: str
    REDIS_PORT: str
    REDIS_DB: int
    REDIS_PASSWORD: str
    REDIS_URL: str

    # POSTGRES DB
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = MasterSettings()


__all__ = ["settings"]
