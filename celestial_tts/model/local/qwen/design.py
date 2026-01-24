import asyncio
from typing import ClassVar, List, Literal, Optional, Set, Tuple, Union, get_args

import numpy as np
from fastapi import HTTPException
from qwen_tts import Qwen3TTSModel

from celestial_tts.database import Database
from celestial_tts.model.local import LocalTTSModel, NonEmptyStr

Language = Literal[
    "auto",
    "chinese",
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
]

Speaker = Literal["generated"]

SUPPORTED_LANGUAGES: Set[Language] = set(get_args(Language))

SUPPORTED_SPEAKERS: Set[Speaker] = set(get_args(Speaker))


class QwenTTSDesign(LocalTTSModel[Language, Speaker]):
    """Qwen3 TTS using designed voices by the model"""

    model_config = {"arbitrary_types_allowed": True}
    model: Qwen3TTSModel
    loaded: bool = True
    lock: ClassVar[asyncio.Lock] = asyncio.Lock()

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
    ) -> Optional[Speaker]:
        if speaker_str not in SUPPORTED_SPEAKERS:
            return None
        return speaker_str

    async def get_supported_languages(self, database: Database) -> Set[Language]:
        return SUPPORTED_LANGUAGES

    async def get_supported_speakers(
        self, database: Database
    ) -> Optional[Set[Tuple[Speaker, str]]]:
        return {(speaker, speaker) for speaker in SUPPORTED_SPEAKERS}

    async def generate_voice(
        self,
        database: Database,
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

        if not instruct:
            raise HTTPException(
                status_code=400,
                detail="Instruct is required for voice design. Please provide a valid instruction.",
            )

        def run():
            return self.model.generate_voice_design(
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
