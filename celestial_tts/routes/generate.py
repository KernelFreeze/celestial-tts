import base64
import io
from typing import List, Literal, Optional, Union

import soundfile as sf
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.injectors import get_config, get_database, get_models
from celestial_tts.model import ModelState
from celestial_tts.model.local.factory import LocalTTSFactory, LocalTTSType
from celestial_tts.model.types import NonEmptyStr

Status = Literal["ok", "error"]


class GenerateRequest(BaseModel):
    model_id: str
    text: Union[NonEmptyStr, List[NonEmptyStr]]
    language: str
    speaker: str
    instruct: Optional[NonEmptyStr] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    provider: str = "local"


class GenerateResponse(BaseModel):
    status: Status
    """The status of the request"""
    wavs: List[str]
    """
    The base64-encoded audio data.
    Every entry in the list is a base64-encoded audio
    file for every matching input text.
    """
    sampling_rate: int
    """The sampling rate of the audio"""


router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    config: Config = Depends(get_config),
    models: ModelState = Depends(get_models),
    database: Database = Depends(get_database),
) -> GenerateResponse:
    """Generate a text-to-speech audio."""
    if request.provider == "local":
        if models.local_state is None:
            raise HTTPException(status_code=500, detail="Local state not initialized")

        model_type = LocalTTSType.from_str(request.model_id)
        model = models.local_state.model_cache.get_or_put(
            model_type,
            lambda: LocalTTSFactory.create(
                model_type, config.integrated_models.device_map
            ),
        )
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Validate language
        language_type = await model.str_to_language(database, request.language)
        if language_type is None:
            supported = await model.get_supported_languages(database)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{request.language}'. Supported: {sorted(supported)}",
            )

        # Validate speaker
        speaker_type = await model.str_to_speaker(database, request.speaker)
        if speaker_type is None:
            supported = await model.get_supported_speakers(database)
            if supported is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported speaker '{request.speaker}'. Supported: {sorted(supported)}",
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported speaker '{request.speaker}'",
                )

        wavs, sr = await model.generate_voice(
            database,
            request.text,
            language_type,
            speaker_type,
            request.instruct,
            request.top_k,
            request.top_p,
            request.temperature,
            request.repetition_penalty,
            request.max_new_tokens,
        )

        # Encode as proper WAV files and serialize to base64
        base64_wavs = []
        for wav in wavs:
            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format="WAV")
            buffer.seek(0)
            base64_wavs.append(base64.b64encode(buffer.read()).decode("utf-8"))
        return GenerateResponse(
            status="ok",
            wavs=base64_wavs,
            sampling_rate=sr,
        )
    else:
        raise HTTPException(
            status_code=400, detail="Remote models not implemented yet."
        )
