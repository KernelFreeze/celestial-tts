from typing import Optional

import torch
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from celestial_tts.database.model.custom_speaker import QwenCustomSpeaker


def serialize_tensor(tensor: torch.Tensor, key: str = "data") -> bytes:
    """Serialize a PyTorch tensor to bytes using safetensors."""
    # Ensure tensor is contiguous before saving (required by safetensors)
    return safetensors_save({key: tensor.contiguous()})


def deserialize_tensor(data: bytes, key: str = "data") -> torch.Tensor:
    """Deserialize bytes back to a PyTorch tensor using safetensors."""
    tensors = safetensors_load(data)
    return tensors[key]


def qwen_speaker_from_prompt(
    name: str, item: VoiceClonePromptItem
) -> QwenCustomSpeaker:
    """Create a Speaker database model from a VoiceClonePromptItem."""
    return QwenCustomSpeaker(
        name=name,
        ref_code=serialize_tensor(item.ref_code, "ref_code")
        if item.ref_code is not None
        else None,
        ref_spk_embedding=serialize_tensor(item.ref_spk_embedding, "ref_spk_embedding"),
        x_vector_only_mode=item.x_vector_only_mode,
        icl_mode=item.icl_mode,
        ref_text=item.ref_text,
    )


def prompt_from_qwen_speaker(
    speaker: QwenCustomSpeaker, device: Optional[torch.device] = None
) -> VoiceClonePromptItem:
    """Reconstruct a VoiceClonePromptItem from a Speaker database model."""
    ref_code = None
    if speaker.ref_code is not None:
        ref_code = deserialize_tensor(speaker.ref_code, "ref_code")
        if device is not None:
            ref_code = ref_code.to(device)

    ref_spk_embedding = deserialize_tensor(
        speaker.ref_spk_embedding, "ref_spk_embedding"
    )
    if device is not None:
        ref_spk_embedding = ref_spk_embedding.to(device)

    return VoiceClonePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=ref_spk_embedding,
        x_vector_only_mode=speaker.x_vector_only_mode,
        icl_mode=speaker.icl_mode,
        ref_text=speaker.ref_text,
    )
