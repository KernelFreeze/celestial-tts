"""RunPod serverless handler for Celestial TTS."""

import asyncio
import base64
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor

# Configure HuggingFace to use RunPod's cached model volume.
# Models are pre-cached on the network volume, so from_pretrained()
# loads them directly without downloading.
RUNPOD_MODEL_CACHE = "/runpod-volume/huggingface-cache"
if os.path.isdir(RUNPOD_MODEL_CACHE):
    os.environ["HF_HOME"] = RUNPOD_MODEL_CACHE

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

# Module-level configuration
config = Config()
setup_logging(config.logging)

logger = logging.getLogger(__name__)

# Log configuration for debugging
logger.info(
    f"Configuration loaded: device_map={config.integrated_models.device_map}, max_loaded_models={config.integrated_models.max_loaded_models}"
)

# Thread pool for blocking model loading operations
_model_loading_executor = ThreadPoolExecutor(max_workers=1)

# Lazy initialization globals
_database = None
_models = None
_init_lock = asyncio.Lock()


async def _get_database():
    """Lazy async initialization of database."""
    global _database
    if _database is None:
        async with _init_lock:
            if _database is None:
                logger.info("Initializing database...")
                _database = Database(config.database.url)
                await _database.init_db()
                logger.info("Database initialized")
    return _database


def _get_models():
    """Lazy initialization of models (synchronous)."""
    global _models
    if _models is None:
        logger.info("Initializing model state...")
        _models = ModelState(config=config)
        logger.info(
            f"Model state initialized (local_enabled={_models.local_state is not None})"
        )
    return _models


def _load_model_sync(model_type, device_map):
    """Synchronous model loading to run in thread pool."""
    logger.info(f"Loading model {model_type.value} on device: {device_map}")
    model = LocalTTSFactory.create(
        model_type, device_map, quantize_4bit=config.integrated_models.quantize_4bit
    )
    logger.info(f"Model {model_type.value} loaded on device: {device_map}")
    return model


async def _get_or_load_model(model_state, model_type):
    """Get model from cache or load it in thread pool."""
    # Check if model is already cached
    model = model_state.local_state.model_cache.get(model_type)
    if model is not None:
        return model

    # Load model in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(
        _model_loading_executor,
        _load_model_sync,
        model_type,
        config.integrated_models.device_map,
    )
    model_state.local_state.model_cache.put(model_type, model)
    return model


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

    # Initialize services
    db = await _get_database()
    model_state = _get_models()

    if model_state.local_state is None:
        return {"error": "Local model state not initialized"}

    # Get or load model
    try:
        model = await _get_or_load_model(model_state, model_type)
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        return {"error": f"Failed to load model: {str(e)}"}

    # Validate language
    language_type = await model.str_to_language(db, language)
    if language_type is None:
        supported = await model.get_supported_languages(db)
        return {
            "error": f"Unsupported language '{language}'. Supported: {sorted(supported)}"
        }

    # Validate speaker
    speaker_type = await model.str_to_speaker(db, speaker)
    if speaker_type is None:
        supported = await model.get_supported_speakers(db)
        if supported is not None:
            speaker_names = sorted([name for _, name in supported])
            return {
                "error": f"Unsupported speaker '{speaker}'. Supported: {speaker_names}"
            }
        return {"error": f"Unsupported speaker '{speaker}'"}

    # Generate audio
    wavs, sr = await model.generate_voice(
        db,
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

    # Initialize services
    db = await _get_database()
    model_state = _get_models()

    if model_state.local_state is None:
        return {"error": "Local model state not initialized"}

    # Map model and voice
    model_type = _map_model_id(model_name)
    speaker = _map_voice(voice)

    # Get or load model
    try:
        model = await _get_or_load_model(model_state, model_type)
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        return {"error": f"Failed to load model: {str(e)}"}

    # Validate speaker
    speaker_type = await model.str_to_speaker(db, speaker)
    if speaker_type is None:
        supported = await model.get_supported_speakers(db)
        if supported is not None:
            speaker_names = sorted([name for _, name in supported])
            return {"error": f"Unsupported voice '{voice}'. Supported: {speaker_names}"}
        return {"error": f"Unsupported voice '{voice}'"}

    # Auto-detect language
    language_type = await model.str_to_language(db, "auto")
    if language_type is None:
        language_type = await model.str_to_language(db, "english")
        if language_type is None:
            return {"error": "Could not determine language for synthesis"}

    # Generate audio
    wavs, sr = await model.generate_voice(
        db, text, language_type, speaker_type, instruct
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


async def _handle_health() -> dict:
    """Handle health check requests."""
    try:
        model_state = _get_models()
        model_status = "ok" if model_state.local_state is not None else "initializing"

        return {
            "status": "healthy" if model_status == "ok" else "degraded",
            "checks": {"models": model_status},
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "unhealthy", "error": str(e)}


async def handler(job: dict) -> dict:
    """RunPod serverless handler entry point."""
    try:
        input_data = job.get("input", {})

        # Handle health check requests immediately
        if input_data.get("health"):
            return await _handle_health()

        if input_data.get("openai"):
            return await _handle_openai(input_data)
        else:
            return await _handle_generate(input_data)
    except Exception as e:
        logger.exception("Unhandled error in RunPod handler")
        return {"error": str(e)}


logger.info("RunPod handler module loaded, waiting for first request...")
runpod.serverless.start({"handler": handler})
