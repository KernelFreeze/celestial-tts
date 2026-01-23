from typing import Annotated, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
from pydantic import BaseModel, StringConstraints

LanguageT = TypeVar("LanguageT")
SpeakerT = TypeVar("SpeakerT")

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class LocalTTSModel(BaseModel, Generic[LanguageT, SpeakerT]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_voice(
        self,
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
