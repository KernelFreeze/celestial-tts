import copyreg
import importlib
import importlib.util
import logging
from enum import Enum

import torch
from fastapi.exceptions import HTTPException
from qwen_tts import Qwen3TTSModel
from transformers import BitsAndBytesConfig

from celestial_tts.model.local import LocalTTSModel
from celestial_tts.model.local.qwen.clone import QwenTTSClone
from celestial_tts.model.local.qwen.design import QwenTTSDesign
from celestial_tts.model.local.qwen.preset import QwenTTSPreset

# HACK: Upstream qwen_tts stores ``dict.keys()`` views as model attributes
# (e.g. ``self.supported_speakers = config...spk_id.keys()``).
# The bitsandbytes 4-bit quantiser calls ``deepcopy(model)`` during
# ``from_pretrained``, which fails because ``dict_keys`` is not picklable.
# Register a reducer so ``copy``/``pickle`` can handle the type.
# The copy materialises as a plain ``list``, which is fine — the
# attribute is only used for iteration and membership tests.
copyreg.pickle(type({}.keys()), lambda obj: (list, (list(obj),)))  # pyright: ignore[reportArgumentType]


def _is_flash_attn_available() -> bool:
    """Check if flash-attn is available for use."""
    return importlib.util.find_spec("flash_attn") is not None


def _is_nvfp4_available() -> bool:
    """Check if NVFP4 quantization is available (requires Blackwell GPU + fp_quant)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        return False
    try:
        transformers = importlib.import_module("transformers")

        return hasattr(transformers, "FPQuantConfig")
    except ImportError:
        return False


def _build_4bit_quant_config() -> object | None:
    """Build the best available 4-bit quantization config for the current hardware.

    Returns a transformers-compatible quantization config, or None if
    no 4-bit quantization backend is available.

    Preference order: NVFP4 (Blackwell) > NF4 (bitsandbytes).
    """
    if _is_nvfp4_available():
        from transformers import FPQuantConfig

        logging.info(
            "Using NVFP4 4-bit quantization (A Blackwell or newer GPU was detected)"
        )
        return FPQuantConfig(forward_dtype="nvfp4", pseudoquantization=False)

    logging.info("Using NF4 4-bit quantization (bitsandbytes)")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )


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
    def create(
        model_type: LocalTTSType,
        device_map: str = "cuda:0",
        *,
        quantize_4bit: bool = False,
    ) -> LocalTTSModel:
        """
        Create a TTS model instance on CUDA.

        Args:
            model_type: The type of TTS model to create.
            device_map: Target device for the model.
            quantize_4bit: Load with 4-bit quantization (NVFP4 or NF4 fallback).

        Returns:
            A TTSModel instance based on the model_type.

        Raises:
            ValueError: If an unknown model type is provided.
        """
        hf_name = _HF_MODEL_NAMES.get(model_type)
        if hf_name is None:
            raise ValueError(f"Unknown Qwen TTS model type: {model_type}")

        # Determine attention implementation: use flash_attention_2 only if
        # CUDA is available AND flash-attn package is installed
        if torch.cuda.is_available() and _is_flash_attn_available():
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"

        logging.info(f"Using attention implementation: {attn_impl}")

        extra_kwargs: dict = {}
        if quantize_4bit:
            quant_config = _build_4bit_quant_config()
            if quant_config is not None:
                extra_kwargs["quantization_config"] = quant_config

        model = Qwen3TTSModel.from_pretrained(
            hf_name,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            **extra_kwargs,
        )

        try:
            model.model = torch.compile(  # pyright: ignore[reportAttributeAccessIssue]
                model.model, mode="max-autotune", fullgraph=True
            )
            logging.info("Model compiled successfully")
        except Exception as e:
            logging.warning(f"Failed to compile model: {e}. Using slow path")

        try:
            model.model.speech_tokenizer.model = torch.compile(  # pyright: ignore[reportOptionalMemberAccess]
                model.model.speech_tokenizer.model,  # pyright: ignore[reportOptionalMemberAccess]
                mode="max-autotune",
                fullgraph=True,
            )
            logging.info("Speech tokenizer compiled successfully")
        except Exception as e:
            logging.warning(f"Failed to compile speech tokenizer: {e}. Using slow path")

        if model_type in _PRESET_TYPES:
            return QwenTTSPreset(model=model)
        elif model_type in _CLONE_TYPES:
            return QwenTTSClone(model=model)
        elif model_type in _DESIGN_TYPES:
            return QwenTTSDesign(model=model)
        else:
            raise ValueError(f"Unknown Qwen TTS model type: {model_type}")
