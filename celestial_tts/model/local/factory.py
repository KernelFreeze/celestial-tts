from enum import Enum

import torch
from qwen_tts import Qwen3TTSModel

from celestial_tts.model.local import LocalTTSModel
from celestial_tts.model.local.qwen.custom import QwenTTSCustom
from celestial_tts.model.local.qwen.preset import QwenTTSPreset


class LocalTTSType(Enum):
    QWEN_PRESET = "preset"
    QWEN_CUSTOM = "custom"


class LocalTTSFactory:
    """Factory for creating TTS model instances."""

    @staticmethod
    def create(model_type: LocalTTSType, device_map: str = "cuda:0") -> LocalTTSModel:
        """
        Create a TTS model instance on CUDA.

        Args:
            model_type: The type of TTS model to create.

        Returns:
            A TTSModel instance based on the model_type.

        Raises:
            ValueError: If an unknown model type is provided.
        """
        if model_type == LocalTTSType.QWEN_PRESET:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=device_map,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            return QwenTTSPreset(model=model)
        elif model_type == LocalTTSType.QWEN_CUSTOM:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map=device_map,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            return QwenTTSCustom(model=model)
        else:
            raise ValueError(f"Unknown Qwen TTS model type: {model_type}")
