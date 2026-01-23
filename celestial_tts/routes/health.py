from fastapi import APIRouter, Depends

from celestial_tts.config import Config
from celestial_tts.injectors import get_config

router = APIRouter()


@router.get("/health")
async def health_check(config: Config = Depends(get_config)):
    """Health check endpoint that shows current configuration."""
    return {
        "status": "ok",
    }
