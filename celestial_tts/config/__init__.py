from pathlib import Path

import tomli_w
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from celestial_tts.config.bootstrap import BootstrapConfig
from celestial_tts.config.database import DatabaseConfig
from celestial_tts.config.logging import LoggingConfig
from celestial_tts.config.models import IntegratedModelsConfig


class Config(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CELESTIAL_",
        env_nested_delimiter="__",
        toml_file=[
            "config.toml",
            Path.home() / ".config" / "celestial-tts" / "config.toml",
        ],
    )

    bootstrap: BootstrapConfig = Field(
        default_factory=BootstrapConfig,
        description="Configuration for bootstrap operations on first startup",
    )

    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Configuration for database",
    )

    integrated_models: IntegratedModelsConfig = Field(
        default_factory=IntegratedModelsConfig,
        description="Configuration for integrated models",
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Configuration for logging",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Check if any TOML file exists, if not create default
        toml_files_value = settings_cls.model_config.get("toml_file", [])
        if isinstance(toml_files_value, (str, Path)):
            toml_files: list[str | Path] = [toml_files_value]
        elif isinstance(toml_files_value, (list, tuple)):
            toml_files = list(toml_files_value)
        else:
            toml_files = []

        toml_paths = [p if isinstance(p, Path) else Path(p) for p in toml_files]
        if toml_paths and not any(p.exists() for p in toml_paths):
            create_default_config(toml_paths[0])

        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )


def create_default_config(path: Path | None = None) -> Path:
    """
    Create a default configuration file.

    Args:
        path: Where to create the config file. If None, creates in current directory as config.toml.

    Returns:
        Path to the created config file.
    """
    if path is None:
        path = Path("config.toml")

    path.parent.mkdir(parents=True, exist_ok=True)
    default_config = Config.model_construct(
        bootstrap=BootstrapConfig.model_construct(),
        database=DatabaseConfig.model_construct(),
        integrated_models=IntegratedModelsConfig.model_construct(),
        logging=LoggingConfig.model_construct(),
    )
    path.write_text(tomli_w.dumps(default_config.model_dump()), encoding="utf-8")
    return path
