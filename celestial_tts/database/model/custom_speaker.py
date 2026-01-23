from datetime import datetime
from typing import Optional
from uuid import UUID

import torch
from sqlmodel import Field, SQLModel
from uuid_utils import uuid7

from celestial_tts.database.utils import prompt_from_qwen_speaker


class QwenCustomSpeaker(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid7, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Voice clone prompt data (serialized from VoiceClonePromptItem)
    ref_code: Optional[bytes] = Field(default=None)  # Serialized torch.Tensor
    ref_spk_embedding: bytes  # Serialized torch.Tensor (required)
    x_vector_only_mode: bool = Field(default=False)
    icl_mode: bool = Field(default=True)
    ref_text: Optional[str] = Field(default=None)

    def to_voice_clone_prompt(self, device: Optional[torch.device] = None):
        prompt_from_qwen_speaker(self, device)
