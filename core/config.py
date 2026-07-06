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
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Server-side speech-to-text for voice answers in the psychometric assessment.
    # "sarvam" (Indian-language tuned, primary) | "gemini" (fallback) | "none"
    # (browser Web Speech API only, no server key needed).
    SPEECH_STT_PROVIDER: str = os.getenv("SPEECH_STT_PROVIDER", "none")
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    API_KEY: str = os.getenv("API_KEY", "")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    AUTO_SEED_ON_STARTUP: str = os.getenv("AUTO_SEED_ON_STARTUP", "false")

    # Zero-friction demo: seed mock borrowers into an empty DB on startup so the
    # dashboard is populated the instant the server boots. Defaults to true.
    SEED_ON_STARTUP: str = os.getenv("SEED_ON_STARTUP", "true")

    # Simulated 5-facet mode on psychometric survey completion (for testing/demo only).
    # MUST default to false for safety so real production files aren't contaminated.
    SIMULATE_ALL_FACETS: str = os.getenv("SIMULATE_ALL_FACETS", "false")

    # Time limits for the psychometric assessment (in seconds)
    PSYCHOMETRIC_TIME_LIMIT_SECONDS: int = int(os.getenv("PSYCHOMETRIC_TIME_LIMIT_SECONDS", "420"))
    PSYCHOMETRIC_EXTENSION_SECONDS: int = int(os.getenv("PSYCHOMETRIC_EXTENSION_SECONDS", "120"))

    # Application throttle. A borrower may submit at most APPLICATION_LIMIT_COUNT
    # completed assessments per rolling APPLICATION_LIMIT_WINDOW_DAYS window. This
    # is an anti-gaming control: a repeat applicant already knows the psychometric
    # questionnaire from earlier sittings, so unlimited re-applications would let
    # them rehearse answers. Keyed on the borrower's user_id.
    APPLICATION_LIMIT_ENABLED: str = os.getenv("APPLICATION_LIMIT_ENABLED", "true")
    APPLICATION_LIMIT_COUNT: int = int(os.getenv("APPLICATION_LIMIT_COUNT", "1"))
    APPLICATION_LIMIT_WINDOW_DAYS: int = int(os.getenv("APPLICATION_LIMIT_WINDOW_DAYS", "30"))

    # Storage backend. Defaults to Postgres (the deployed/Render configuration).
    # Set USE_SQLITE=true for an optional zero-dependency local run (a SQLite
    # file, no Docker/Postgres needed), useful for an offline laptop demo.
    # A provided DATABASE_URL always takes precedence over both.
    USE_SQLITE: str = os.getenv("USE_SQLITE", "false")
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", "./alt_credit.db")

    # Single connection string (e.g. Render provides this). Takes
    # precedence over USE_SQLITE and the individual POSTGRES_* vars when set.
    DATABASE_URL_ENV: str = os.getenv("DATABASE_URL", "")

    @property
    def using_sqlite(self) -> bool:
        url = self.DATABASE_URL
        return url.startswith("sqlite")

    @property
    def DATABASE_URL(self) -> str:
        if self.DATABASE_URL_ENV:
            return self._normalize_async_url(self.DATABASE_URL_ENV)
        if self.USE_SQLITE.strip().lower() in {"1", "true", "yes"}:
            return f"sqlite+aiosqlite:///{self.SQLITE_PATH}"
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
        # Local SQLite, sync-style -> async driver.
        if url.startswith("sqlite+aiosqlite://"):
            return url
        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return url

    @property
    def seed_on_startup_enabled(self) -> bool:
        return self.SEED_ON_STARTUP.strip().lower() in {"1", "true", "yes"}

    @property
    def simulate_all_facets_enabled(self) -> bool:
        return self.SIMULATE_ALL_FACETS.strip().lower() in {"1", "true", "yes"}

    @property
    def application_limit_enabled(self) -> bool:
        return self.APPLICATION_LIMIT_ENABLED.strip().lower() in {"1", "true", "yes"}

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
