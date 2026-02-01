import asyncio
import logging

import typer
from hypercorn.asyncio import serve
from hypercorn.config import Config
from transformers import torch

from celestial_tts import app
from celestial_tts.config.logging import setup_logging

cli_app = typer.Typer()


@cli_app.command()
def main(
    host: str = typer.Option("localhost", help="Host to bind the server to"),
    port: int = typer.Option(8080, help="Port to bind the server to"),
) -> None:
    # Initialize logging with the application's logging configuration
    setup_logging(app.state.config.logging)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Celestial TTS server on {host}:{port}")

    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        major, minor = torch.cuda.get_device_capability()
        logger.info(f"CUDA device major version: {major}")
        logger.info(f"CUDA device minor version: {minor}")

    config = Config()
    config.bind = [f"{host}:{port}"]

    asyncio.run(serve(app, config))  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    cli_app()
