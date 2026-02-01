from typing import Generic, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np
from qwen_tts.inference.qwen3_tts_model import AudioLike

from celestial_tts.database import Database
from celestial_tts.model.types import NonEmptyStr

LanguageT = TypeVar("LanguageT")
SpeakerT = TypeVar("SpeakerT")
InstructT = TypeVar("InstructT")


class LocalTTSModel(Generic[LanguageT, SpeakerT]):
    def supports_custom_speakers(self) -> bool:
        """Return whether this model supports creating custom speakers."""
        return False

    async def create_speaker(
        self,
        database: Database,
        name: str,
        ref_audio: AudioLike,
        ref_text: str,
    ) -> Tuple[SpeakerT, str]:
        """
        Create a new speaker for this model by using the reference audio and text.
        Most models that support custom speakers will use the reference audio and text to clone the speaker's voice.

        Returns:
            SpeakerData: The newly created speaker (Usually the speaker id) and the speaker name.
        """
        raise NotImplementedError()

    async def str_to_language(
        self, database: Database, language_str: str
    ) -> Optional[LanguageT]:
        """Convert a string into a LanguageT. This function might error if conversion is not supported, or None if not found."""
        raise NotImplementedError()

    async def str_to_speaker(
        self, database: Database, speaker_str: str
    ) -> Optional[SpeakerT]:
        """Convert a string into a SpeakerT. This function might error if conversion is not supported, or None if not found."""
        raise NotImplementedError()

    async def get_supported_languages(self, database: Database) -> Set[LanguageT]:
        """Return the set of supported language codes."""
        raise NotImplementedError()

    async def get_supported_speakers(
        self, database: Database
    ) -> Optional[Set[Tuple[SpeakerT, str]]]:
        """
        Return the set of supported speaker names.

        For models with dynamic/custom speakers, return an empty set
        and override validate_speaker() instead.

        Returns:
            Optional[Set[SpeakerData]]: The set of supported speaker names and their corresponding names.
        """
        raise NotImplementedError()

    async def generate_voice(
        self,
        database: Database,
        text: Union[NonEmptyStr, List[NonEmptyStr]],
        language: LanguageT,
        speaker: SpeakerT,
        instruct: Optional[NonEmptyStr] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            instruct:
                Instruction(s) describing desired voice/style. Empty string is allowed (treated as no instruction).
            speaker:
                Speaker name(s). Will be validated against `model.get_supported_speakers()` (case-insensitive).
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)
        """
        raise NotImplementedError()

    def unload(self):
        """
        Unload the model from memory.
        """
        raise NotImplementedError()
