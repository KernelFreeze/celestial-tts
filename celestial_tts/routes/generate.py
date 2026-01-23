import base64
from typing import List, Optional, Union

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.injectors import get_config, get_database, get_models
from celestial_tts.model import ModelState
from celestial_tts.model.local.factory import LocalTTSFactory, LocalTTSType
from celestial_tts.model.types import NonEmptyStr

router = APIRouter()


@router.post("/generate")
async def generate(
    model_id: str,
    text: Union[NonEmptyStr, List[NonEmptyStr]],
    language: str,
    speaker: str,
    instruct: Optional[NonEmptyStr] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    provider: str = "local",
    config: Config = Depends(get_config),
    models: ModelState = Depends(get_models),
    database: Database = Depends(get_database),
):
    """Generate a text-to-speech audio."""
    if provider == "local":
        if models.local_state is None:
            raise HTTPException(status_code=500, detail="Local state not initialized")

        model_type = LocalTTSType.from_str(model_id)
        model = models.local_state.model_cache.get_or_put(
            model_type,
            lambda: LocalTTSFactory.create(
                model_type, config.integrated_models.device_map
            ),
        )
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Validate language
        language_type = await model.str_to_language(database, language)
        if language_type is None:
            supported = await model.get_supported_languages(database)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language}'. Supported: {sorted(supported)}",
            )

        # Validate speaker
        speaker_type = await model.str_to_speaker(database, speaker)
        if speaker_type is None:
            supported = await model.get_supported_speakers(database)
            if supported is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported speaker '{speaker}'. Supported: {sorted(supported)}",
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported speaker '{speaker}'",
                )

        wavs, sr = await model.generate_voice(
            database,
            text,
            language_type,
            speaker_type,
            instruct,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            max_new_tokens,
        )

        # Serialize wavs to base64
        base64_wavs = [base64.b64encode(wav).decode("utf-8") for wav in wavs]
        return {
            "status": "ok",
            "wavs": base64_wavs,
            "sampling_rate": sr,
        }
    else:
        raise HTTPException(
            status_code=400, detail="Remote models not implemented yet."
        )
