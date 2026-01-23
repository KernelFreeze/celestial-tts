from typing import List, Union

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.database.controller.custom_speaker import create_qwen_custom_speaker
from celestial_tts.injectors import get_config, get_database, get_models
from celestial_tts.model import ModelState
from celestial_tts.model.local.factory import LocalTTSFactory, LocalTTSType
from celestial_tts.model.types import NonEmptyStr

router = APIRouter()


@router.get("/speakers")
async def get_speakers(
    model_id: str,
    provider: str = "local",
    config: Config = Depends(get_config),
    models: ModelState = Depends(get_models),
    database: Database = Depends(get_database),
):
    """Get all speakers for a model."""
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

        speakers = await model.get_supported_speakers(database)
        if speakers is None:
            return {"status": "ok", "speakers": []}

        return {
            "status": "ok",
            "speakers": [
                {
                    "id": str(s.id),
                    "name": s.name,
                    "created_at": s.created_at.isoformat(),
                }
                for s in speakers
            ],
        }
    else:
        raise HTTPException(
            status_code=400, detail="Remote models not implemented yet."
        )


@router.post("/speakers")
async def create_speaker(
    model_id: str,
    name: str,
    text: Union[NonEmptyStr, List[NonEmptyStr]],
    language: str,
    instruct: Union[NonEmptyStr, List[NonEmptyStr]],
    provider: str = "local",
    config: Config = Depends(get_config),
    models: ModelState = Depends(get_models),
    database: Database = Depends(get_database),
):
    """Create a new custom speaker for a model."""
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

        # Check if the model supports custom speakers
        if not model.supports_custom_speakers():
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' does not support custom speakers.",
            )

        # Validate language
        language_type = await model.str_to_language(database, language)
        if language_type is None:
            supported = await model.get_supported_languages(database)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language}'. Supported: {sorted(supported)}",
            )

        # Create the speaker
        speaker = await model.create_speaker(name, text, language_type, instruct)

        # Save to database
        speaker = await create_qwen_custom_speaker(database, speaker)

        return {
            "status": "ok",
            "speaker": {
                "id": str(speaker.id),
                "name": speaker.name,
                "created_at": speaker.created_at.isoformat(),
            },
        }
    else:
        raise HTTPException(
            status_code=400, detail="Remote models not implemented yet."
        )
