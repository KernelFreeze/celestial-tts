from typing import List, Literal, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field

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
    model_id: str = Field(description="The ID of the model to create the speaker for")
    name: str = Field(description="The name of the speaker")
    text: Union[NonEmptyStr, List[NonEmptyStr]] = Field(
        description="The reference text for voice design"
    )
    language: str = Field(description="The language code for the speaker")
    instruct: Union[NonEmptyStr, List[NonEmptyStr]] = Field(
        description="Voice design instructions (e.g., 'A young female with energetic tone')"
    )
    provider: str = Field(default="local", description="The model provider")


class DeleteSpeakerRequest(BaseModel):
    model_id: str = Field(description="The ID of the model to delete the speaker for")
    speaker_id: UUID = Field(description="The UUID of the speaker to delete")
    provider: str = Field(default="local", description="The model provider")


class SpeakerInfo(BaseModel):
    id: UUID = Field(description="The unique identifier of the speaker")
    name: str = Field(description="The display name of the speaker")
    created_at: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp of when the speaker was created"
    )


class GetSpeakersResponse(BaseModel):
    status: Status = Field(description="The status of the request ('ok' or 'error')")
    speakers: List[SpeakerInfo] | List[str] = Field(
        description="List of available speakers (SpeakerInfo for custom, strings for preset)"
    )


class SpeakerResponse(BaseModel):
    status: Status = Field(description="The status of the request ('ok' or 'error')")
    speaker: SpeakerInfo = Field(description="The created speaker information")


class DeleteSpeakerResponse(BaseModel):
    status: Status = Field(description="The status of the request ('ok' or 'error')")
    message: str = Field(description="A human-readable message describing the result")


router = APIRouter(tags=["speakers"])


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
                id=speaker.id,
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
