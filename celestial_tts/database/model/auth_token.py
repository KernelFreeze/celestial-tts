import base64
import secrets
from datetime import datetime
from typing import Optional
from uuid import UUID

from passlib.hash import argon2
from sqlmodel import Field, SQLModel
from uuid_utils import uuid7

TOKEN_PREFIX = "sk-ct-v1-"
SECRET_BYTE_LENGTH = 32


class AuthToken(SQLModel, table=True):
    __tablename__ = "auth_token"  # pyright: ignore[reportAssignmentType]

    id: UUID = Field(default_factory=uuid7, primary_key=True)
    name: str = Field(description="A human-readable name for the token")
    secret_hash: str = Field(description="PHC-encoded hash of the token secret")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = Field(default=None)
    expires_at: Optional[datetime] = Field(default=None)
    revoked: bool = Field(default=False)

    @staticmethod
    def generate_secret() -> str:
        """Generate a cryptographically secure random secret."""
        return secrets.token_urlsafe(SECRET_BYTE_LENGTH)

    @staticmethod
    def hash_secret(secret: str) -> str:
        """Hash a secret using Argon2 (PHC format)."""
        return argon2.using(rounds=4).hash(secret)

    @staticmethod
    def verify_secret(secret: str, secret_hash: str) -> bool:
        """Verify a secret against its hash."""
        return argon2.verify(secret, secret_hash)

    def encode_token(self, secret: str) -> str:
        """
        Encode the token ID and secret into the final token format.
        Format: sk-ct-v1-<base64(id:secret)>
        """
        token_data = f"{self.id}:{secret}"
        encoded = base64.urlsafe_b64encode(token_data.encode()).decode()
        return f"{TOKEN_PREFIX}{encoded}"

    @staticmethod
    def decode_token(token: str) -> tuple[UUID, str] | None:
        """
        Decode a token string into its ID and secret components.
        Returns None if the token format is invalid.
        """
        if not token.startswith(TOKEN_PREFIX):
            return None

        encoded_part = token[len(TOKEN_PREFIX) :]
        try:
            decoded = base64.urlsafe_b64decode(encoded_part).decode()
            id_str, secret = decoded.split(":", 1)
            return UUID(id_str), secret
        except (ValueError, UnicodeDecodeError):
            return None

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if the token is valid (not revoked and not expired)."""
        return not self.revoked and not self.is_expired()
