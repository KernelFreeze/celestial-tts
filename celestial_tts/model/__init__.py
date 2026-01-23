from typing import Optional

from pydantic import BaseModel

from celestial_tts.config import Config
from celestial_tts.model.local.model_cache import ModelCache


class LocalModelState(BaseModel):
    """
    Holds the state of the local TTS models.
    """

    model_cache: ModelCache

    def __init__(self, max_loaded_models: int):
        super().__init__(model_cache=ModelCache(max_loaded_models))


class ModelState(BaseModel):
    """
    Holds the state for local and remote TTS models.
    """

    local_state: Optional[LocalModelState] = None

    def __init__(self, config: Config):
        local_state = None
        if config.integrated_models.enabled:
            local_state = LocalModelState(config.integrated_models.max_loaded_models)
        super().__init__(local_state=local_state)
