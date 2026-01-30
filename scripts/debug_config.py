#!/usr/bin/env python3
"""Debug configuration loading for RunPod environment.

This script prints the configuration values to help debug
environment variable loading issues.
"""

import os
import sys


def print_env_vars():
    """Print relevant environment variables."""
    print("=== Environment Variables ===")
    relevant_prefixes = ("CELESTIAL_", "RUNPOD_", "HF_", "TRANSFORMERS_")

    for key, value in sorted(os.environ.items()):
        if key.startswith(relevant_prefixes):
            # Mask sensitive values
            if "TOKEN" in key or "KEY" in key or "PASSWORD" in key:
                value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            print(f"  {key}={value}")

    print()


def print_config():
    """Print loaded configuration."""
    print("=== Loaded Configuration ===")

    try:
        from celestial_tts.config import Config

        config = Config()
        print(f"  integrated_models.enabled = {config.integrated_models.enabled}")
        print(
            f"  integrated_models.device_map = {config.integrated_models.device_map!r}"
        )
        print(
            f"  integrated_models.max_loaded_models = {config.integrated_models.max_loaded_models}"
        )
        print(f"  database.url = {config.database.url!r}")
        print(f"  bootstrap.create_token = {config.bootstrap.create_token}")
        print(f"  logging.level = {config.logging.level}")
    except Exception as e:
        print(f"  ERROR loading config: {e}")
        import traceback

        traceback.print_exc()

    print()


def test_model_loading():
    """Test if we can load the model (without actually loading)."""
    print("=== Model Loading Test ===")

    try:
        from celestial_tts.config import Config

        config = Config()
        device_map = config.integrated_models.device_map

        print(f"  Would load model on device: {device_map!r}")

        if device_map == "cpu":
            print("  ⚠️ WARNING: device_map is 'cpu' - model will load on CPU!")
            print("     Expected: 'cuda:0' for GPU acceleration")
            return False
        elif "cuda" in device_map:
            print(f"  ✓ device_map is '{device_map}' - GPU will be used")
            return True
        else:
            print(f"  ? Unknown device_map value: {device_map!r}")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("Celestial TTS Configuration Debugger")
    print("=" * 60)
    print()

    print_env_vars()
    print_config()

    success = test_model_loading()

    print()
    print("=" * 60)
    if success:
        print("Configuration looks good!")
    else:
        print("Configuration issues detected - check env vars above")
        sys.exit(1)


if __name__ == "__main__":
    main()
