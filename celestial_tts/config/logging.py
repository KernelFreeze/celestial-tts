import logging
import sys

from pydantic import Field
from pydantic_settings import BaseSettings


class LoggingConfig(BaseSettings):
    """Configuration for application logging."""

    model_config = {"env_prefix": "CELESTIAL_LOGGING_"}

    level: str = Field(
        default="INFO",
        description="Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )

    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for log messages",
    )

    # Package-specific log levels
    fastapi_level: str = Field(
        default="INFO",
        description="Log level for FastAPI",
    )

    starlette_level: str = Field(
        default="INFO",
        description="Log level for Starlette",
    )

    uvicorn_level: str = Field(
        default="INFO",
        description="Log level for Uvicorn",
    )

    hypercorn_level: str = Field(
        default="INFO",
        description="Log level for Hypercorn",
    )

    transformers_level: str = Field(
        default="WARNING",
        description="Log level for Transformers library",
    )

    torch_level: str = Field(
        default="WARNING",
        description="Log level for PyTorch",
    )

    sqlalchemy_level: str = Field(
        default="WARNING",
        description="Log level for SQLAlchemy",
    )

    httpx_level: str = Field(
        default="WARNING",
        description="Log level for HTTPX",
    )

    httpcore_level: str = Field(
        default="WARNING",
        description="Log level for HTTPCore",
    )

    # Request logging configuration
    log_requests: bool = Field(
        default=True,
        description="Enable HTTP request logging middleware",
    )


def setup_logging(config: LoggingConfig | None = None) -> None:
    """
    Configure logging for the application.

    Args:
        config: Logging configuration. If None, uses default configuration.
    """
    if config is None:
        config = LoggingConfig()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level.upper()))

    formatter = logging.Formatter(
        fmt=config.format,
        datefmt=config.date_format,
    )
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # Configure package-specific loggers
    logger_configs: dict[str, str] = {
        "fastapi": config.fastapi_level,
        "starlette": config.starlette_level,
        "uvicorn": config.uvicorn_level,
        "uvicorn.access": config.uvicorn_level,
        "uvicorn.error": config.uvicorn_level,
        "hypercorn": config.hypercorn_level,
        "hypercorn.access": config.hypercorn_level,
        "hypercorn.error": config.hypercorn_level,
        "transformers": config.transformers_level,
        "torch": config.torch_level,
        "sqlalchemy": config.sqlalchemy_level,
        "sqlalchemy.engine": config.sqlalchemy_level,
        "sqlalchemy.pool": config.sqlalchemy_level,
        "httpx": config.httpx_level,
        "httpcore": config.httpcore_level,
    }

    for logger_name, level in logger_configs.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = True

    # Log the initialization
    logging.getLogger(__name__).info("Logging configured with level: %s", config.level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.

    Args:
        name: Name for the logger (typically __name__ of the module).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
