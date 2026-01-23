from typing import Optional

from pydantic import BaseModel

from celestial_tts.config import Config
from celestial_tts.model.local.model_cache import LocalModelCache
from celestial_tts.model.remote import RemoteTTSModel


class LocalModelState(BaseModel):
    """
    Holds the state of the local TTS models.
    """

    model_cache: LocalModelCache

    def __init__(self, max_loaded_models: int):
        super().__init__(model_cache=LocalModelCache(max_loaded_models))


class RemoteModelState(BaseModel):
    """
    Holds the state of the remote TTS models.
    """

    model_map: dict[str, RemoteTTSModel] = {}

    def __init__(self):
        super().__init__()


class ModelState(BaseModel):
    """
    Holds the state for local and remote TTS models.
    """

    local_state: Optional[LocalModelState]
    remote_state: RemoteModelState

    def __init__(self, config: Config):
        remote_state = RemoteModelState()

        local_state = None
        if config.integrated_models.enabled:
            local_state = LocalModelState(config.integrated_models.max_loaded_models)

        super().__init__(local_state=local_state, remote_state=remote_state)
