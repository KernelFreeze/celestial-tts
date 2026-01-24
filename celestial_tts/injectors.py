from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.database.controller.auth_token import verify_token
from celestial_tts.database.model.auth_token import AuthToken
from celestial_tts.model import ModelState

# Security scheme for Bearer token authentication
_bearer_scheme = HTTPBearer()


def get_config(request: Request) -> Config:
    """Inject config into route handlers."""
    return request.app.state.config


def get_models(request: Request) -> ModelState:
    """Inject model state into route handlers."""
    return request.app.state.models


def get_database(request: Request) -> Database:
    """Inject database into route handlers."""
    return request.app.state.database


async def get_authenticated_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> AuthToken:
    """
    Validate the Bearer token and return the authenticated token.

    Usage in route handlers:
        @router.get("/protected")
        async def protected_route(token: AuthenticatedToken):
            # token is the validated AuthToken model
            return {"message": f"Hello, {token.name}"}
    """
    database: Database = request.app.state.database
    token = await verify_token(database, credentials.credentials)

    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token


# Type alias for cleaner route handler signatures (similar to Axum's extractor pattern)
AuthenticatedToken = Annotated[AuthToken, Depends(get_authenticated_token)]
