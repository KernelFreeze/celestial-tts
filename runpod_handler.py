"""RunPod serverless handler for Celestial TTS.

Bypasses FastAPI and calls the TTS business logic directly.
The container auto-detects the RunPod environment via RUNPOD_POD_ID
and switches to this handler instead of the HTTP server.
"""

import asyncio
import base64
import io
import logging

import runpod
import soundfile as sf
from fastapi.exceptions import HTTPException

from celestial_tts.config import Config
from celestial_tts.config.logging import setup_logging
from celestial_tts.database import Database
from celestial_tts.model import ModelState
from celestial_tts.model.local.factory import LocalTTSFactory, LocalTTSType
from celestial_tts.routes.openai_speech import (
    AUDIO_CONTENT_TYPES,
    SF_FORMAT_MAP,
    _map_model_id,
    _map_voice,
)

# Module-level initialization (replaces FastAPI lifespan)
config = Config()
setup_logging(config.logging)

logger = logging.getLogger(__name__)

database = Database(config.database.url)
asyncio.get_event_loop().run_until_complete(database.init_db())

models = ModelState(config=config)

logger.info("RunPod handler initialized")


async def _handle_generate(input_data: dict) -> dict:
    """Handle native TTS generation requests."""
    model_id = input_data.get("model_id")
    text = input_data.get("text")
    language = input_data.get("language")
    speaker = input_data.get("speaker")
    instruct = input_data.get("instruct")

    if not model_id or not text or not language or not speaker:
        return {"error": "Missing required fields: model_id, text, language, speaker"}

    try:
        model_type = LocalTTSType.from_str(model_id)
    except HTTPException as e:
        return {"error": e.detail}

    if models.local_state is None:
        return {"error": "Local model state not initialized"}

    model = models.local_state.model_cache.get_or_put(
        model_type,
        lambda: LocalTTSFactory.create(
            model_type, config.integrated_models.device_map
        ),
    )
    if model is None:
        return {"error": "Model not found"}

    # Validate language
    language_type = await model.str_to_language(database, language)
    if language_type is None:
        supported = await model.get_supported_languages(database)
        return {
            "error": f"Unsupported language '{language}'. Supported: {sorted(supported)}"
        }

    # Validate speaker
    speaker_type = await model.str_to_speaker(database, speaker)
    if speaker_type is None:
        supported = await model.get_supported_speakers(database)
        if supported is not None:
            speaker_names = sorted([name for _, name in supported])
            return {
                "error": f"Unsupported speaker '{speaker}'. Supported: {speaker_names}"
            }
        return {"error": f"Unsupported speaker '{speaker}'"}

    # Generate audio
    wavs, sr = await model.generate_voice(
        database,
        text,
        language_type,
        speaker_type,
        instruct,
        input_data.get("top_k"),
        input_data.get("top_p"),
        input_data.get("temperature"),
        input_data.get("repetition_penalty"),
        input_data.get("max_new_tokens"),
    )

    # Encode as WAV files and serialize to base64
    base64_wavs = []
    for wav in wavs:
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        buffer.seek(0)
        base64_wavs.append(base64.b64encode(buffer.read()).decode("utf-8"))

    return {"status": "ok", "wavs": base64_wavs, "sampling_rate": sr}


async def _handle_openai(input_data: dict) -> dict:
    """Handle OpenAI-compatible speech requests."""
    model_name = input_data.get("model", "tts-1")
    voice = input_data.get("voice")
    text = input_data.get("input")
    response_format = input_data.get("response_format", "wav").lower()
    instruct = input_data.get("instructions")

    if not voice or not text:
        return {"error": "Missing required fields: voice, input"}

    if not text.strip():
        return {"error": "Input text cannot be empty"}

    if response_format not in AUDIO_CONTENT_TYPES:
        return {
            "error": f"Unsupported response_format '{response_format}'. Supported: {list(AUDIO_CONTENT_TYPES.keys())}"
        }

    if models.local_state is None:
        return {"error": "Local model state not initialized"}

    # Map model and voice
    model_type = _map_model_id(model_name)
    speaker = _map_voice(voice)

    model = models.local_state.model_cache.get_or_put(
        model_type,
        lambda: LocalTTSFactory.create(
            model_type, config.integrated_models.device_map
        ),
    )
    if model is None:
        return {"error": "Model not found"}

    # Validate speaker
    speaker_type = await model.str_to_speaker(database, speaker)
    if speaker_type is None:
        supported = await model.get_supported_speakers(database)
        if supported is not None:
            speaker_names = sorted([name for _, name in supported])
            return {
                "error": f"Unsupported voice '{voice}'. Supported: {speaker_names}"
            }
        return {"error": f"Unsupported voice '{voice}'"}

    # Auto-detect language
    language_type = await model.str_to_language(database, "auto")
    if language_type is None:
        language_type = await model.str_to_language(database, "english")
        if language_type is None:
            return {"error": "Could not determine language for synthesis"}

    # Generate audio
    wavs, sr = await model.generate_voice(
        database, text, language_type, speaker_type, instruct
    )

    if not wavs:
        return {"error": "No audio generated"}

    wav = wavs[0]

    # Encode to requested format
    buffer = io.BytesIO()
    sf_format = SF_FORMAT_MAP.get(response_format, "WAV")

    try:
        if response_format == "pcm":
            import numpy as np

            pcm_data = (wav * 32767).astype(np.int16)
            buffer.write(pcm_data.tobytes())
        else:
            sf.write(buffer, wav, sr, format=sf_format)
    except Exception:
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        response_format = "wav"

    buffer.seek(0)
    audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "status": "ok",
        "audio": audio_b64,
        "format": response_format,
        "content_type": AUDIO_CONTENT_TYPES[response_format],
    }


async def handler(job: dict) -> dict:
    """RunPod serverless handler entry point."""
    try:
        input_data = job.get("input", {})

        if input_data.get("openai"):
            return await _handle_openai(input_data)
        else:
            return await _handle_generate(input_data)
    except Exception as e:
        logger.exception("Unhandled error in RunPod handler")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
