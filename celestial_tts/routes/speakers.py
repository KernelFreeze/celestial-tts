from typing import List, Literal, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.database.controller.custom_speaker import (
    create_qwen_custom_speaker,
    delete_qwen_custom_speaker,
)
from celestial_tts.injectors import get_config, get_database, get_models
from celestial_tts.model import ModelState
from celestial_tts.model.local.factory import LocalTTSFactory, LocalTTSType
from celestial_tts.model.types import NonEmptyStr

Status = Literal["ok", "error"]


class CreateSpeakerRequest(BaseModel):
    model_id: str
    """The ID of the model to create the speaker for"""
    name: str
    """The name of the speaker"""
    text: Union[NonEmptyStr, List[NonEmptyStr]]
    """The text to use for the speaker"""
    language: str
    """The language of the speaker"""
    instruct: Union[NonEmptyStr, List[NonEmptyStr]]
    """The instructions for the speaker"""
    provider: str = "local"
    """The provider of the speaker"""


class DeleteSpeakerRequest(BaseModel):
    model_id: str
    """The ID of the model to delete the speaker for"""
    speaker_id: UUID
    """The ID of the speaker to delete"""
    provider: str = "local"
    """The provider of the speaker"""


class SpeakerInfo(BaseModel):
    id: UUID
    """The ID of the speaker"""
    name: str
    """The name of the speaker"""
    created_at: Optional[str] = None
    """The date and time the speaker was created"""


class GetSpeakersResponse(BaseModel):
    status: Status
    """The status of the request"""
    speakers: List[SpeakerInfo] | List[str]
    """The list of speakers"""


class SpeakerResponse(BaseModel):
    status: Status
    """The status of the request"""
    speaker: SpeakerInfo
    """The speaker information"""


class DeleteSpeakerResponse(BaseModel):
    status: Status
    """The status of the request"""
    message: str
    """The message of the response"""


router = APIRouter()


@router.get("/speakers", response_model=GetSpeakersResponse)
async def get_speakers(
    model_id: str,
    provider: str = "local",
    config: Config = Depends(get_config),
    models: ModelState = Depends(get_models),
    database: Database = Depends(get_database),
) -> GetSpeakersResponse:
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
            return GetSpeakersResponse(status="ok", speakers=[])

        # Check if speakers are strings or SpeakerInfo objects
        speaker = next(iter(speakers), None)
        if speaker is None:
            return GetSpeakersResponse(status="ok", speakers=[])

        if isinstance(speaker, str):
            return GetSpeakersResponse(status="ok", speakers=list(speakers))

        return GetSpeakersResponse(
            status="ok",
            speakers=[
                SpeakerInfo(
                    id=s.id,
                    name=s.name,
                    created_at=s.created_at.isoformat(),
                )
                for s in speakers
            ],
        )
    else:
        raise HTTPException(
            status_code=400, detail="Remote models not implemented yet."
        )


@router.post("/speakers", response_model=SpeakerResponse)
async def create_speaker(
    request: CreateSpeakerRequest,
    config: Config = Depends(get_config),
    models: ModelState = Depends(get_models),
    database: Database = Depends(get_database),
) -> SpeakerResponse:
    """Create a new custom speaker for a model."""
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

        # Check if the model supports custom speakers
        if not model.supports_custom_speakers():
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_id}' does not support custom speakers.",
            )

        # Validate language
        language_type = await model.str_to_language(database, request.language)
        if language_type is None:
            supported = await model.get_supported_languages(database)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{request.language}'. Supported: {sorted(supported)}",
            )

        # Create the speaker
        speaker = await model.create_speaker(
            request.name, request.text, language_type, request.instruct
        )

        # Save to database
        speaker = await create_qwen_custom_speaker(database, speaker)

        return SpeakerResponse(
            status="ok",
            speaker=SpeakerInfo(
                id=str(speaker.id),
                name=speaker.name,
                created_at=speaker.created_at.isoformat(),
            ),
        )
    else:
        raise HTTPException(
            status_code=400, detail="Remote models not implemented yet."
        )


@router.delete("/speakers/{speaker_id}", response_model=DeleteSpeakerResponse)
async def delete_speaker(
    request: DeleteSpeakerRequest,
    config: Config = Depends(get_config),
    models: ModelState = Depends(get_models),
    database: Database = Depends(get_database),
) -> DeleteSpeakerResponse:
    """Delete a custom speaker."""
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

        # Check if the model supports custom speakers
        if not model.supports_custom_speakers():
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_id}' does not support custom speakers.",
            )

        uuid = request.speaker_id

        # Delete the speaker
        deleted = await delete_qwen_custom_speaker(database, uuid)
        if not deleted:
            raise HTTPException(status_code=404, detail="Speaker not found")

        return DeleteSpeakerResponse(
            status="ok", message="Speaker deleted successfully"
        )
    else:
        raise HTTPException(
            status_code=400, detail="Remote models not implemented yet."
        )
