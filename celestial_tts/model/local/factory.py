from enum import Enum

import torch
from fastapi.exceptions import HTTPException
from qwen_tts import Qwen3TTSModel

from celestial_tts.model.local import LocalTTSModel
from celestial_tts.model.local.qwen.clone import QwenTTSClone
from celestial_tts.model.local.qwen.design import QwenTTSDesign
from celestial_tts.model.local.qwen.preset import QwenTTSPreset


class LocalTTSType(Enum):
    QWEN_PRESET_1_7B = "qwen3-tts-1.7b-preset"
    QWEN_PRESET_0_6B = "qwen3-tts-0.6b-preset"
    QWEN_VOICE_CLONE_1_7B = "qwen3-tts-1.7b-voice-clone"
    QWEN_VOICE_CLONE_0_6B = "qwen3-tts-0.6b-voice-clone"
    QWEN_VOICE_DESIGN_1_7B = "qwen3-tts-1.7b-voice-design"

    @classmethod
    def from_str(cls, value: str) -> "LocalTTSType":
        for member in cls:
            if member.value == value:
                return member
        raise HTTPException(400, f"Unknown local TTS type: {value}")


_HF_MODEL_NAMES: dict[LocalTTSType, str] = {
    LocalTTSType.QWEN_PRESET_1_7B: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    LocalTTSType.QWEN_PRESET_0_6B: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    LocalTTSType.QWEN_VOICE_CLONE_1_7B: "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    LocalTTSType.QWEN_VOICE_CLONE_0_6B: "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    LocalTTSType.QWEN_VOICE_DESIGN_1_7B: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

_PRESET_TYPES = {LocalTTSType.QWEN_PRESET_1_7B, LocalTTSType.QWEN_PRESET_0_6B}
_CLONE_TYPES = {LocalTTSType.QWEN_VOICE_CLONE_1_7B, LocalTTSType.QWEN_VOICE_CLONE_0_6B}
_DESIGN_TYPES = {
    LocalTTSType.QWEN_VOICE_DESIGN_1_7B,
}


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
        hf_name = _HF_MODEL_NAMES.get(model_type)
        if hf_name is None:
            raise ValueError(f"Unknown Qwen TTS model type: {model_type}")

        model = Qwen3TTSModel.from_pretrained(
            hf_name,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
            if torch.cuda.is_available()
            else "sdpa",
        )

        if model_type in _PRESET_TYPES:
            return QwenTTSPreset(model=model)
        elif model_type in _CLONE_TYPES:
            return QwenTTSClone(model=model)
        elif model_type in _DESIGN_TYPES:
            return QwenTTSDesign(model=model)
        else:
            raise ValueError(f"Unknown Qwen TTS model type: {model_type}")
