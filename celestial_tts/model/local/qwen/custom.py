import asyncio
from typing import Any, List, Literal, Optional, Set, Tuple, Union, get_args
from uuid import UUID

import numpy as np
from qwen_tts import Qwen3TTSModel

from celestial_tts.database import Database
from celestial_tts.database.controller.custom_speaker import (
    select_qwen_custom_speaker_by_id,
    select_qwen_custom_speakers,
)
from celestial_tts.database.model.custom_speaker import QwenCustomSpeaker
from celestial_tts.database.utils import qwen_speaker_from_prompt
from celestial_tts.model.local import LocalTTSModel, NonEmptyStr

Language = Literal["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

SUPPORTED_LANGUAGES: Set[Language] = set(get_args(Language))


class QwenTTSCustom(LocalTTSModel[Language, QwenCustomSpeaker]):
    """Qwen3 TTS using Custom voices"""

    model_config = {"arbitrary_types_allowed": True}
    model: Qwen3TTSModel
    loaded: bool = True
    lock = asyncio.Lock()

    @classmethod
    def generate_voice_design(
        cls,
        name: str,
        design_model: Qwen3TTSModel,
        clone_model: Qwen3TTSModel,
        text: Union[NonEmptyStr, List[NonEmptyStr]],
        language: Language,
        instruct: Union[NonEmptyStr, List[NonEmptyStr]],
    ):
        wavs, sr = design_model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )

        voice_clone_prompt = clone_model.create_voice_clone_prompt(
            ref_audio=(
                wavs[0],
                sr,
            ),
            ref_text=text,  # pyright: ignore[reportArgumentType]
        )[0]

        return qwen_speaker_from_prompt(name, voice_clone_prompt)

    def __init__(
        self,
        model: Qwen3TTSModel,
    ):
        super().__init__(model=model)

    async def str_to_language(
        self, database: Database, language_str: str
    ) -> Optional[Language]:
        if language_str not in SUPPORTED_LANGUAGES:
            return None
        return language_str

    async def str_to_speaker(
        self, database: Database, speaker_str: str
    ) -> Optional[QwenCustomSpeaker]:
        # Try to parse the speaker_str into an UUID
        try:
            uuid = UUID(speaker_str)
        except ValueError:
            raise ValueError(f"Invalid speaker UUID: {speaker_str}")
        return await select_qwen_custom_speaker_by_id(database, uuid)

    async def get_supported_languages(self, database: Database) -> Set[Language]:
        return SUPPORTED_LANGUAGES

    async def get_supported_speakers(
        self, database: Database
    ) -> Optional[Set[QwenCustomSpeaker]]:
        return set(await select_qwen_custom_speakers(database))

    async def generate_voice(
        self,
        database: Database,
        text: Union[NonEmptyStr, List[NonEmptyStr]],
        language: Language,
        speaker: QwenCustomSpeaker,
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

        def run():
            return self.model.generate_voice_clone(
                voice_clone_prompt=speaker.to_voice_clone_prompt(self.model.device),
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

        async with self.lock:  # Prevent concurrent use of the model
            return await asyncio.to_thread(run)

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
