from enum import Enum

import torch
from fastapi.exceptions import HTTPException
from qwen_tts import Qwen3TTSModel

from celestial_tts.model.local import LocalTTSModel
from celestial_tts.model.local.qwen.clone import QwenTTSClone
from celestial_tts.model.local.qwen.design import QwenTTSDesign
from celestial_tts.model.local.qwen.preset import QwenTTSPreset


class LocalTTSType(Enum):
    QWEN_PRESET = "qwen3-tts-preset"
    QWEN_VOICE_CLONE = "qwen3-tts-voice-clone"
    QWEN_VOICE_DESIGN = "qwen3-tts-voice-design"

    @classmethod
    def from_str(cls, value: str) -> "LocalTTSType":
        for member in cls:
            if member.value == value:
                return member
        raise HTTPException(400, f"Unknown local TTS type: {value}")


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
                attn_implementation="sdpa",
            )
            return QwenTTSPreset(model=model)
        elif model_type == LocalTTSType.QWEN_VOICE_CLONE:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map=device_map,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            return QwenTTSClone(model=model)
        elif model_type == LocalTTSType.QWEN_VOICE_DESIGN:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map=device_map,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            return QwenTTSDesign(model=model)
        else:
            raise ValueError(f"Unknown Qwen TTS model type: {model_type}")
