import base64
import io
import urllib.request
from typing import List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import UUID

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.database.controller.custom_speaker import (
    delete_qwen_custom_speaker,
)
from celestial_tts.injectors import get_config, get_database, get_models
from celestial_tts.model import ModelState
from celestial_tts.model.local.factory import LocalTTSFactory, LocalTTSType
from celestial_tts.model.types import NonEmptyStr

Status = Literal["ok", "error"]

# Type alias for audio input - only accepts string (URL or base64) or numpy array with sample rate
AudioInput = str | Tuple[np.ndarray, int]


def _is_url(s: str) -> bool:
    """Check if a string is a valid HTTP(S) URL."""
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _is_probably_base64(s: str) -> bool:
    """Check if a string is likely base64 encoded audio data."""
    if s.startswith("data:audio"):
        return True
    # Base64 strings are typically long; try to validate by decoding a small sample
    if len(s) > 256:
        try:
            # Try decoding the first 100 chars (padded to valid length) to check validity
            sample = s[:100]
            # Pad to make it a valid base64 length (multiple of 4)
            padding = (4 - len(sample) % 4) % 4
            base64.b64decode(sample + "=" * padding, validate=True)
            return True
        except Exception:
            return False
    return False


def _decode_base64_to_wav_bytes(b64: str) -> bytes:
    """Decode base64 string to raw audio bytes, handling data URI scheme."""
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_to_np(x: str) -> Tuple[np.ndarray, int]:
    """
    Load audio from a URL or base64 string.

    Args:
        x: Either an HTTP(S) URL or a base64-encoded audio string.

    Returns:
        A tuple of (audio waveform as float32 numpy array, sample rate).

    Raises:
        ValueError: If the input is neither a valid URL nor base64 data.
    """
    if _is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif _is_probably_base64(x):
        wav_bytes = _decode_base64_to_wav_bytes(x)
        with io.BytesIO(wav_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        raise ValueError("Audio input must be an HTTP(S) URL or base64-encoded data. ")

    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    return audio.astype(np.float32), int(sr)


def normalize_audio_input(audio: AudioInput) -> Tuple[np.ndarray, int]:
    """
    Normalize audio input into (waveform, sample_rate) tuple.

    Supports:
      - str: URL or base64 audio string (NOT local file paths)
      - (np.ndarray, int): Pre-loaded waveform with sample rate

    Args:
        audio: Audio input in one of the supported formats.

    Returns:
        Tuple of (float32 waveform numpy array, sample rate).

    Raises:
        ValueError: If input format is invalid or unsupported.
        TypeError: If input type is not supported.
    """
    if isinstance(audio, str):
        return load_audio_to_np(audio)
    elif isinstance(audio, tuple) and len(audio) == 2:
        arr, sr = audio
        if isinstance(arr, np.ndarray) and isinstance(sr, int):
            result = arr.astype(np.float32)
            if result.ndim > 1:
                result = np.mean(result, axis=-1).astype(np.float32)
            return result, sr
        raise ValueError("Tuple must be (np.ndarray, int) for (waveform, sample_rate).")
    elif isinstance(audio, np.ndarray):
        raise ValueError("For numpy waveform input, pass a tuple (audio, sample_rate).")
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")


class CreateSpeakerRequest(BaseModel):
    provider: str = Field(default="local", description="The model provider")
    model_id: str = Field(description="The ID of the model to create the speaker for")
    name: str = Field(description="The name of the speaker")
    text: NonEmptyStr = Field(description="The reference text for the audio file")
    audio: str = Field(
        description="The audio to clone: HTTP(S) URL or base64-encoded audio data. "
        "Local file paths are not supported for security reasons."
    )


class DeleteSpeakerRequest(BaseModel):
    model_id: str = Field(description="The ID of the model to delete the speaker for")
    speaker_id: UUID = Field(description="The UUID of the speaker to delete")
    provider: str = Field(default="local", description="The model provider")


class SpeakerInfo(BaseModel):
    id: Union[UUID, str] = Field(description="The unique identifier of the speaker")
    name: str = Field(description="The display name of the speaker")


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
                model_type,
                config.integrated_models.device_map,
                quantize_4bit=config.integrated_models.quantize_4bit,
            ),
        )
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")

        speakers = await model.get_supported_speakers(database)
        if speakers is None:
            return GetSpeakersResponse(status="ok", speakers=[])

        return GetSpeakersResponse(
            status="ok",
            speakers=[
                SpeakerInfo(
                    id=speaker_id,
                    name=speaker_name,
                )
                for speaker_id, speaker_name in speakers
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
    """
    Create a new custom speaker for a model. This will usually take the provided
    audio and text and clone the speaker's voice.
    """
    if request.provider == "local":
        if models.local_state is None:
            raise HTTPException(status_code=500, detail="Local state not initialized")

        model_type = LocalTTSType.from_str(request.model_id)
        model = models.local_state.model_cache.get_or_put(
            model_type,
            lambda: LocalTTSFactory.create(
                model_type,
                config.integrated_models.device_map,
                quantize_4bit=config.integrated_models.quantize_4bit,
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

        try:
            audio_data = normalize_audio_input(request.audio)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create the speaker
        speaker_id, speaker_name = await model.create_speaker(
            database,
            request.name,
            audio_data,
            request.text,
        )

        return SpeakerResponse(
            status="ok",
            speaker=SpeakerInfo(
                id=speaker_id,
                name=speaker_name,
            ),
        )
    else:
        raise HTTPException(
            status_code=400, detail="Remote models not implemented yet."
        )


@router.delete("/speakers", response_model=DeleteSpeakerResponse)
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
                model_type,
                config.integrated_models.device_map,
                quantize_4bit=config.integrated_models.quantize_4bit,
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
