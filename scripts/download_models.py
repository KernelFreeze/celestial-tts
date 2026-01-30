#!/usr/bin/env python3
"""Pre-download Qwen3-TTS models for RunPod serverless.

This script downloads model weights during Docker build to eliminate cold start.
Uses huggingface_hub for lightweight downloading without loading models into memory.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(model_id: str, cache_dir: str) -> bool:
    """Download a single model from HuggingFace using huggingface_hub.

    Args:
        model_id: The HuggingFace model ID
        cache_dir: Directory to cache the model

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading model: {model_id}...")

        from huggingface_hub import snapshot_download

        # Download the entire model repository
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
        )

        logger.info(f"Successfully downloaded: {model_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")
        return False


def download_models():
    """Download all Qwen3-TTS models from HuggingFace."""
    try:
        # Set cache directory
        cache_dir = os.environ.get("HF_HOME", "/app/.cache/huggingface")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Using cache directory: {cache_dir}")

        # All three Qwen3-TTS model variants
        models_to_download = [
            ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "preset voice"),
            ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "voice clone"),
            ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "voice design"),
        ]

        success_count = 0
        failed_models = []

        for model_id, description in models_to_download:
            logger.info(f"Starting download for {description} ({model_id})")
            if download_model(model_id, cache_dir):
                success_count += 1
            else:
                failed_models.append((model_id, description))

        if success_count == len(models_to_download):
            logger.info("All models pre-downloaded successfully!")
            return 0
        elif success_count > 0:
            logger.warning(
                f"Downloaded {success_count}/{len(models_to_download)} models"
            )
            logger.warning(f"Failed models: {failed_models}")
            logger.warning(
                "Failed models will be downloaded at runtime (slower cold start)"
            )
            return 0
        else:
            logger.error("All model downloads failed")
            logger.warning("Models will be downloaded at runtime (cold start)")
            return 0

    except Exception as e:
        logger.exception(f"Unexpected error during model download: {e}")
        logger.warning("Models will be downloaded at runtime (cold start)")
        return 0


if __name__ == "__main__":
    sys.exit(download_models())
