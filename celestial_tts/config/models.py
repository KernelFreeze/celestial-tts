from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class IntegratedModelsConfig(BaseSettings):
    """Configuration for integrated models."""

    model_config = SettingsConfigDict(env_prefix="CELESTIAL_INTEGRATED_MODELS_")

    enabled: bool = Field(
        default=True,
        description="Whether to enable integrated models",
    )

    max_loaded_models: int = Field(
        default=2,
        ge=1,
        description="Maximum number of integrated models to keep loaded in memory simultaneously",
    )

    device_map: str = Field(
        default="cpu",
        description="Device map for integrated models",
    )
