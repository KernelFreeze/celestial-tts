from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlmodel import select

from celestial_tts.database import Database
from celestial_tts.database.model.auth_token import AuthToken


async def create_auth_token(
    database: Database,
    name: str,
    expires_at: Optional[datetime] = None,
) -> tuple[AuthToken, str]:
    """
    Create a new auth token with a generated secret.
    Returns the token model and the plaintext secret (only available at creation time).
    """
    secret = AuthToken.generate_secret()
    secret_hash = AuthToken.hash_secret(secret)

    token = AuthToken(
        name=name,
        secret_hash=secret_hash,
        expires_at=expires_at,
    )

    async with database.async_session() as session:
        session.add(token)
        await session.commit()
        await session.refresh(token)

    return token, secret


async def select_auth_token_by_id(
    database: Database, token_id: UUID
) -> Optional[AuthToken]:
    """Get an auth token by its ID."""
    async with database.async_session() as session:
        result = await session.exec(
            select(AuthToken).where(AuthToken.id == token_id).limit(1)
        )
        return result.first()


async def select_all_auth_tokens(database: Database) -> list[AuthToken]:
    """Get all auth tokens (without secrets)."""
    async with database.async_session() as session:
        result = await session.exec(select(AuthToken))
        return list(result.all())


async def verify_token(database: Database, token_string: str) -> Optional[AuthToken]:
    """
    Verify a token string and return the token if valid.
    Also updates the last_used_at timestamp.
    Returns None if the token is invalid, expired, or revoked.
    """
    decoded = AuthToken.decode_token(token_string)
    if decoded is None:
        return None

    token_id, secret = decoded

    async with database.async_session() as session:
        result = await session.exec(
            select(AuthToken).where(AuthToken.id == token_id).limit(1)
        )
        token = result.first()

        if token is None:
            return None

        if not token.is_valid():
            return None

        if not AuthToken.verify_secret(secret, token.secret_hash):
            return None

        # Update last used timestamp
        token.last_used_at = datetime.utcnow()
        session.add(token)
        await session.commit()
        await session.refresh(token)

        return token


async def revoke_auth_token(database: Database, token_id: UUID) -> bool:
    """
    Revoke an auth token by its ID.
    Returns True if the token was found and revoked, False otherwise.
    """
    async with database.async_session() as session:
        result = await session.exec(
            select(AuthToken).where(AuthToken.id == token_id).limit(1)
        )
        token = result.first()

        if token is None:
            return False

        token.revoked = True
        session.add(token)
        await session.commit()

        return True


async def delete_auth_token(database: Database, token_id: UUID) -> bool:
    """
    Permanently delete an auth token by its ID.
    Returns True if the token was found and deleted, False otherwise.
    """
    async with database.async_session() as session:
        result = await session.exec(
            select(AuthToken).where(AuthToken.id == token_id).limit(1)
        )
        token = result.first()

        if token is None:
            return False

        await session.delete(token)
        await session.commit()

        return True
