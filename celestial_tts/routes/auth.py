from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field

from celestial_tts.database import Database
from celestial_tts.database.controller.auth_token import (
    create_auth_token,
    delete_auth_token,
    revoke_auth_token,
    select_all_auth_tokens,
    select_auth_token_by_id,
    verify_token,
)
from celestial_tts.injectors import get_database

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

Status = Literal["ok", "error"]


class CreateTokenRequest(BaseModel):
    name: str = Field(description="A human-readable name for the token")
    expires_at: Optional[datetime] = Field(
        default=None, description="Optional expiration timestamp (ISO 8601)"
    )


class TokenInfo(BaseModel):
    id: UUID = Field(description="The unique identifier of the token")
    name: str = Field(description="The human-readable name of the token")
    created_at: datetime = Field(description="When the token was created")
    last_used_at: Optional[datetime] = Field(
        default=None, description="When the token was last used"
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="When the token expires"
    )
    revoked: bool = Field(description="Whether the token has been revoked")


class CreateTokenResponse(BaseModel):
    status: Status = Field(description="The status of the request")
    token: str = Field(
        description="The full token string (only shown once at creation)"
    )
    info: TokenInfo = Field(description="Token metadata")


class ListTokensResponse(BaseModel):
    status: Status = Field(description="The status of the request")
    tokens: list[TokenInfo] = Field(description="List of all tokens")


class TokenResponse(BaseModel):
    status: Status = Field(description="The status of the request")
    info: TokenInfo = Field(description="Token metadata")


class VerifyTokenRequest(BaseModel):
    token: str = Field(description="The full token string to verify")


class VerifyTokenResponse(BaseModel):
    status: Status = Field(description="The status of the request")
    valid: bool = Field(description="Whether the token is valid")
    info: Optional[TokenInfo] = Field(
        default=None, description="Token metadata if valid"
    )


class MessageResponse(BaseModel):
    status: Status = Field(description="The status of the request")
    message: str = Field(description="A human-readable message")


def token_to_info(token) -> TokenInfo:
    """Convert an AuthToken model to TokenInfo response."""
    return TokenInfo(
        id=token.id,
        name=token.name,
        created_at=token.created_at,
        last_used_at=token.last_used_at,
        expires_at=token.expires_at,
        revoked=token.revoked,
    )


@router.post("/tokens", response_model=CreateTokenResponse)
async def create_token(
    request: CreateTokenRequest,
    database: Database = Depends(get_database),
) -> CreateTokenResponse:
    """
    Create a new auth token.

    The token string is only returned once at creation time.
    Store it securely as it cannot be retrieved later.
    """
    token, secret = await create_auth_token(
        database,
        name=request.name,
        expires_at=request.expires_at,
    )

    token_string = token.encode_token(secret)

    return CreateTokenResponse(
        status="ok",
        token=token_string,
        info=token_to_info(token),
    )


@router.get("/tokens", response_model=ListTokensResponse)
async def list_tokens(
    database: Database = Depends(get_database),
) -> ListTokensResponse:
    """List all auth tokens (without secrets)."""
    tokens = await select_all_auth_tokens(database)
    return ListTokensResponse(
        status="ok",
        tokens=[token_to_info(t) for t in tokens],
    )


@router.get("/tokens/{token_id}", response_model=TokenResponse)
async def get_token(
    token_id: UUID,
    database: Database = Depends(get_database),
) -> TokenResponse:
    """Get a specific auth token by ID."""
    token = await select_auth_token_by_id(database, token_id)
    if token is None:
        raise HTTPException(status_code=404, detail="Token not found")

    return TokenResponse(
        status="ok",
        info=token_to_info(token),
    )


@router.post("/tokens/verify", response_model=VerifyTokenResponse)
async def verify_token_endpoint(
    request: VerifyTokenRequest,
    database: Database = Depends(get_database),
) -> VerifyTokenResponse:
    """
    Verify a token string.

    Returns whether the token is valid and its metadata if so.
    """
    token = await verify_token(database, request.token)

    if token is None:
        return VerifyTokenResponse(
            status="ok",
            valid=False,
            info=None,
        )

    return VerifyTokenResponse(
        status="ok",
        valid=True,
        info=token_to_info(token),
    )


@router.post("/tokens/{token_id}/revoke", response_model=MessageResponse)
async def revoke_token(
    token_id: UUID,
    database: Database = Depends(get_database),
) -> MessageResponse:
    """
    Revoke an auth token.

    Revoked tokens cannot be used for authentication but remain in the database.
    """
    success = await revoke_auth_token(database, token_id)
    if not success:
        raise HTTPException(status_code=404, detail="Token not found")

    return MessageResponse(
        status="ok",
        message="Token revoked successfully",
    )


@router.delete("/tokens/{token_id}", response_model=MessageResponse)
async def delete_token(
    token_id: UUID,
    database: Database = Depends(get_database),
) -> MessageResponse:
    """Permanently delete an auth token."""
    success = await delete_auth_token(database, token_id)
    if not success:
        raise HTTPException(status_code=404, detail="Token not found")

    return MessageResponse(
        status="ok",
        message="Token deleted successfully",
    )
