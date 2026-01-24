import asyncio

import typer
from hypercorn.asyncio import serve
from hypercorn.config import Config

from celestial_tts import app

cli_app = typer.Typer()


@cli_app.command()
def main(
    host: str = typer.Option("localhost", help="Host to bind the server to"),
    port: int = typer.Option(8080, help="Port to bind the server to"),
) -> None:
    config = Config()
    config.bind = [f"{host}:{port}"]

    asyncio.run(serve(app, config))  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    cli_app()
