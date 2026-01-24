from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


Status = Literal["ok", "error"]


class HealthResponse(BaseModel):
    status: Status
    """The status of the request"""
    message: str
    """The message of the request"""


@router.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", message="All systems are operational")
