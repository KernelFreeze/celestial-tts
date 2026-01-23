from collections import OrderedDict
from typing import Callable, Optional

from pydantic import BaseModel

from celestial_tts.model.local import LocalTTSModel
from celestial_tts.model.local.factory import LocalTTSType


class LocalModelCache(BaseModel):
    max_loaded_models: int
    _cache: OrderedDict[LocalTTSType, LocalTTSModel]

    def __init__(self, max_loaded_models: int):
        super().__init__(max_loaded_models=max_loaded_models)
        self._cache = OrderedDict()

    def get(self, model_type: LocalTTSType) -> Optional[LocalTTSModel]:
        """
        Get a model from the cache. Marks it as recently used.

        Args:
            model_type: Type of the TTS model

        Returns:
            The TTSModel if found in cache, None otherwise
        """
        if model_type in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(model_type)
            return self._cache[model_type]
        return None

    def get_or_put(
        self, model_type: LocalTTSType, loader: Callable[[], LocalTTSModel]
    ) -> LocalTTSModel:
        """
        Get a model from the cache, or load it using the callback if not present.

        Args:
            model_type: Type of the TTS model
            loader: Callback function that loads and returns the model if not in cache

        Returns:
            The TTSModel from cache or newly loaded
        """
        model = self.get(model_type)
        if model is not None:
            return model

        # Load the model using the callback
        model = loader()
        self.put(model_type, model)
        return model

    def put(self, model_type: LocalTTSType, model: LocalTTSModel) -> None:
        """
        Add or update a model in the cache. If cache is full, evicts the LRU model.

        Args:
            model_type: Type of the TTS model
            model: The TTSModel instance to cache
        """
        if model_type in self._cache:
            # Update existing entry and mark as recently used
            self._cache.move_to_end(model_type)
            self._cache[model_type] = model
        else:
            # Check if we need to evict
            if len(self._cache) >= self.max_loaded_models:
                # Remove least recently used (first item)
                lru_model_type, lru_model = self._cache.popitem(last=False)
                # Optionally: add cleanup logic here if models need explicit unloading

            # Add new model
            self._cache[model_type] = model

    def remove(self, model_type: LocalTTSType) -> Optional[LocalTTSModel]:
        """
        Remove a model from the cache.

        Args:
            model_type: Type of the TTS model

        Returns:
            The removed TTSModel if it existed, None otherwise
        """
        return self._cache.pop(model_type, None)

    def clear(self) -> None:
        """Clear all models from the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of models currently in cache."""
        return len(self._cache)

    def __contains__(self, model_type: LocalTTSType) -> bool:
        """Check if a model is in the cache."""
        return model_type in self._cache
