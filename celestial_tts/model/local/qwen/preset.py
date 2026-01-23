from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from qwen_tts import Qwen3TTSModel

from celestial_tts.model.local import LocalTTSModel, NonEmptyStr

Language = Literal["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]
Speaker = Literal[
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]


class QwenTTSPreset(LocalTTSModel[Language, Speaker]):
    """Qwen3 TTS using Preset voices"""

    model_config = {"arbitrary_types_allowed": True}
    model: Qwen3TTSModel
    loaded: bool = True

    def __init__(
        self,
        model: Qwen3TTSModel,
    ):
        super().__init__(model=model)

    def generate_voice(
        self,
        text: Union[NonEmptyStr, List[NonEmptyStr]],
        language: Language,
        speaker: Speaker,
        instruct: Optional[NonEmptyStr] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        if not self.loaded:
            raise ValueError("Model is not loaded")

        return self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def unload(self):
        if not self.loaded:
            return

        import gc

        import torch

        # Check if model exists before deleting
        if hasattr(self, "model"):
            del self.model

        # Clear CUDA cache only if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: synchronize to ensure operations complete
            torch.cuda.synchronize()

        gc.collect()
        self.loaded = False
