import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "alt_credit")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "alt_credit_secret")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "alt_credit_vault")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))

    AES_SECRET_KEY: str = os.getenv(
        "AES_SECRET_KEY",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    API_KEY: str = os.getenv("API_KEY", "")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    AUTO_SEED_ON_STARTUP: str = os.getenv("AUTO_SEED_ON_STARTUP", "false")

    # Single connection string (e.g. Render/Railway/Fly provide this). Takes
    # precedence over the individual POSTGRES_* vars when set.
    DATABASE_URL_ENV: str = os.getenv("DATABASE_URL", "")

    @property
    def DATABASE_URL(self) -> str:
        if self.DATABASE_URL_ENV:
            return self._normalize_async_url(self.DATABASE_URL_ENV)
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @staticmethod
    def _normalize_async_url(url: str) -> str:
        # Managed hosts hand out sync-style URLs (postgres:// or
        # postgresql://). SQLAlchemy's async engine needs the asyncpg driver.
        if url.startswith("postgresql+asyncpg://"):
            return url
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
