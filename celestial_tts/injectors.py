from fastapi import Request

from celestial_tts.config import Config
from celestial_tts.model import ModelState


def get_config(request: Request) -> Config:
    """Inject config into route handlers."""
    return request.app.state.config


def get_models(request: Request) -> ModelState:
    """Inject model state into route handlers."""
    return request.app.state.models
