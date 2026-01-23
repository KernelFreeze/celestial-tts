from typing import Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
from pydantic import BaseModel

from celestial_tts.model.types import NonEmptyStr

LanguageT = TypeVar("LanguageT")
SpeakerT = TypeVar("SpeakerT")


class RemoteTTSModel(BaseModel, Generic[LanguageT, SpeakerT]):
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
        Generate speech using a remote TTS API.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            instruct:
                Instruction(s) describing desired voice/style. Empty string is allowed (treated as no instruction).
            speaker:
                Speaker name(s).
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
                Any other keyword arguments supported by the remote API.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)
        """
        raise NotImplementedError()

    def close(self):
        """
        Close the connection to the remote API and clean up resources.
        """
        raise NotImplementedError()
