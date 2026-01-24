#!/usr/bin/env python3
"""CLI script to create bootstrap auth tokens."""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import typer

# An ugly hack to suppress flash-attn warning during import
# (There's currently no better way to do this, sorry :P)
_old_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
try:
    from celestial_tts.config import Config
    from celestial_tts.database import Database
    from celestial_tts.database.controller.auth_token import create_auth_token
finally:
    sys.stdout.close()
    sys.stdout = _old_stdout

app = typer.Typer(help="Create auth tokens for Celestial TTS")


@app.command()
def create(
    name: str = typer.Option(..., "--name", "-n", help="Name for the token"),
    expires_in_days: Optional[int] = typer.Option(
        None,
        "--expires-in",
        "-e",
        help="Token expiration in days (omit for no expiration)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Only output the token without additional information",
    ),
) -> None:
    """Create a new auth token and print it to stdout."""
    asyncio.run(_create_token(name, expires_in_days, quiet))


async def _create_token(name: str, expires_in_days: Optional[int], quiet: bool) -> None:
    config = Config()
    database = Database(config.database.url)

    await database.init_db()

    expires_at = None
    if expires_in_days is not None:
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

    token, secret = await create_auth_token(database, name, expires_at)
    token_string = token.encode_token(secret)

    await database.close()

    if quiet:
        typer.echo(token_string)
    else:
        typer.echo("Token created successfully!\n")
        typer.echo(f"  ID:         {token.id}")
        typer.echo(f"  Name:       {token.name}")
        typer.echo(f"  Created:    {token.created_at.isoformat()}")
        if expires_at:
            typer.echo(f"  Expires:    {expires_at.isoformat()}")
        else:
            typer.echo("  Expires:    Never")
        typer.echo(f"\n  Token:      {token_string}")
        typer.echo("\nStore this token securely - it cannot be retrieved later.")


if __name__ == "__main__":
    app()
