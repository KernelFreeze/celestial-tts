from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class BootstrapConfig(BaseSettings):
    """Configuration for bootstrap operations on first startup."""

    model_config = SettingsConfigDict(env_prefix="CELESTIAL_BOOTSTRAP_")

    create_token: bool = Field(
        default=False,
        description="Automatically create a bootstrap auth token on startup if no tokens exist",
    )
