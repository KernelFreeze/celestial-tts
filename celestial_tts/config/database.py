from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class DatabaseConfig(BaseSettings):
    """Configuration for database."""

    model_config = SettingsConfigDict(env_prefix="CELESTIAL_DATABASE_")

    url: str = Field(
        default="sqlite+aiosqlite:///database.db",
        description="Database URL. Supports sqlite and postgres (e.g., postgresql+asyncpg://user:pass@localhost/db)",
    )
