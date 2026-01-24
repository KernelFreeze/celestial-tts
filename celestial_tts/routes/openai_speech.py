import io
from typing import Optional

import soundfile as sf
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.injectors import get_config, get_database, get_models
from celestial_tts.model import ModelState
from celestial_tts.model.local.factory import LocalTTSFactory, LocalTTSType

router = APIRouter(tags=["openai-compatible"], prefix="/audio")

# OpenAI voice to internal speaker mapping
# Users can use OpenAI voice names or native speaker names
OPENAI_VOICE_MAP = {
    "alloy": "Vivian",
    "echo": "Dylan",
    "fable": "Serena",
    "onyx": "Eric",
    "nova": "Aiden",
    "shimmer": "Sohee",
}

# Audio format to MIME type mapping
AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# Soundfile format mapping (some formats need translation)
SF_FORMAT_MAP = {
    "mp3": "MP3",
    "opus": "OGG",  # soundfile uses OGG container for opus
    "aac": "WAV",  # fallback to WAV, AAC requires separate encoder
    "flac": "FLAC",
    "wav": "WAV",
    "pcm": "RAW",
}


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""

    model: str = Field(
        description="Model ID to use. Use 'tts-1' for preset voices or specify a model like 'qwen3-tts-preset'"
    )
    input: str = Field(
        max_length=4096, description="The text to generate audio for (max 4096 chars)"
    )
    voice: str = Field(
        description="Voice to use. Supports OpenAI voices (alloy, echo, fable, onyx, nova, shimmer) or native speakers"
    )
    response_format: str = Field(
        default="mp3",
        description="Audio format: mp3, opus, aac, flac, wav, or pcm",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed of generated audio (0.25 to 4.0). Note: speed adjustment may not be supported by all models",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Additional instructions for voice control (only supported by voice-design model)",
    )


def _map_model_id(model: str) -> LocalTTSType:
    """Map OpenAI model names to internal model types."""
    # Direct mapping for native model IDs
    try:
        return LocalTTSType.from_str(model)
    except HTTPException:
        pass

    # Map OpenAI-style model names
    model_lower = model.lower()
    if model_lower in ("tts-1", "tts-1-hd"):
        return LocalTTSType.QWEN_PRESET
    elif "clone" in model_lower or "custom" in model_lower:
        return LocalTTSType.QWEN_VOICE_CLONE
    elif "design" in model_lower:
        return LocalTTSType.QWEN_VOICE_DESIGN

    # Default to preset model
    return LocalTTSType.QWEN_PRESET


def _map_voice(voice: str) -> str:
    """Map OpenAI voice names to internal speaker names."""
    # Check if it's an OpenAI voice name
    if voice.lower() in OPENAI_VOICE_MAP:
        return OPENAI_VOICE_MAP[voice.lower()]
    # Otherwise return as-is (native speaker name or UUID)
    return voice


@router.post(
    "/speech",
    response_class=Response,
    responses={
        200: {
            "content": {
                "audio/mpeg": {},
                "audio/wav": {},
                "audio/opus": {},
                "audio/flac": {},
                "audio/aac": {},
                "audio/pcm": {},
            },
            "description": "Generated audio file",
        }
    },
)
async def create_speech(
    request: SpeechRequest,
    config: Config = Depends(get_config),
    models: ModelState = Depends(get_models),
    database: Database = Depends(get_database),
) -> Response:
    """
    Generate audio from text using an OpenAI-compatible API.

    This endpoint is compatible with OpenAI's /v1/audio/speech API.
    It accepts the same request format and returns audio directly.
    """
    # Validate response format
    response_format = request.response_format.lower()
    if response_format not in AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{request.response_format}'. Supported: {list(AUDIO_CONTENT_TYPES.keys())}",
        )

    # Validate input is not empty
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # Check local model state
    if models.local_state is None:
        raise HTTPException(status_code=500, detail="Local model state not initialized")

    # Map model and voice
    model_type = _map_model_id(request.model)
    speaker = _map_voice(request.voice)

    # Get or create the model
    model = models.local_state.model_cache.get_or_put(
        model_type,
        lambda: LocalTTSFactory.create(model_type, config.integrated_models.device_map),
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Validate speaker
    speaker_type = await model.str_to_speaker(database, speaker)
    if speaker_type is None:
        supported = await model.get_supported_speakers(database)
        if supported is not None:
            speaker_names = sorted([name for _, name in supported])
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported voice '{request.voice}'. Supported: {speaker_names + list(OPENAI_VOICE_MAP.keys())}",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported voice '{request.voice}'",
            )

    # Use English as default language (OpenAI API doesn't have language param)
    # Auto-detect could be added by using "auto" if the model supports it
    language_type = await model.str_to_language(database, "auto")
    if language_type is None:
        # Fallback to English if auto is not supported
        language_type = await model.str_to_language(database, "english")
        if language_type is None:
            raise HTTPException(
                status_code=500, detail="Could not determine language for synthesis"
            )

    # Generate audio
    wavs, sr = await model.generate_voice(
        database,
        request.input,
        language_type,
        speaker_type,
        request.instructions,
    )

    if not wavs:
        raise HTTPException(status_code=500, detail="No audio generated")

    # Get the first (and only) wav since we only pass a single string
    wav = wavs[0]

    # Encode to requested format
    buffer = io.BytesIO()
    sf_format = SF_FORMAT_MAP.get(response_format, "WAV")

    try:
        if response_format == "pcm":
            # For PCM, write raw 16-bit samples
            import numpy as np

            pcm_data = (wav * 32767).astype(np.int16)
            buffer.write(pcm_data.tobytes())
        else:
            sf.write(buffer, wav, sr, format=sf_format)
    except Exception:
        # Fallback to WAV if the requested format fails
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        response_format = "wav"

    buffer.seek(0)

    return Response(
        content=buffer.read(),
        media_type=AUDIO_CONTENT_TYPES[response_format],
        headers={
            "Content-Disposition": f"attachment; filename=speech.{response_format}"
        },
    )
