#!/usr/bin/env python3
"""Test configuration loading locally without Docker.

This script simulates the RunPod environment and verifies
that configuration is loaded correctly from environment variables.
"""

import os
import sys


def setup_runpod_env():
    """Set environment variables to simulate RunPod environment."""
    # RunPod simulation
    os.environ["RUNPOD_POD_ID"] = "local-test-pod-123"

    # Core config - NOTE: Double underscores for nested config!
    os.environ["CELESTIAL_INTEGRATED_MODELS__ENABLED"] = "true"
    os.environ["CELESTIAL_INTEGRATED_MODELS__DEVICE_MAP"] = "cuda:0"
    os.environ["CELESTIAL_INTEGRATED_MODELS__MAX_LOADED_MODELS"] = "1"
    os.environ["CELESTIAL_BOOTSTRAP_CREATE_TOKEN"] = "true"
    os.environ["CELESTIAL_LOGGING_LEVEL"] = "DEBUG"

    # HuggingFace
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"


def test_config():
    """Test configuration loading."""
    print("=" * 60)
    print("Testing Celestial TTS Configuration Loading")
    print("=" * 60)
    print()

    # Show what env vars we set
    print("Environment variables set:")
    for key in sorted(os.environ.keys()):
        if key.startswith("CELESTIAL_"):
            print(f"  {key}={os.environ[key]!r}")
    print()

    # Now load config
    print("Loading configuration...")
    print()

    try:
        from celestial_tts.config import Config

        config = Config()

        print("Configuration loaded successfully:")
        print(f"  integrated_models.enabled = {config.integrated_models.enabled}")
        print(
            f"  integrated_models.device_map = {config.integrated_models.device_map!r}"
        )
        print(
            f"  integrated_models.max_loaded_models = {config.integrated_models.max_loaded_models}"
        )
        print()

        # Check if config matches expectations
        device_map = config.integrated_models.device_map

        if device_map == "cuda:0":
            print("✓ SUCCESS: device_map is correctly set to 'cuda:0'")
            print("  The model will load on GPU")
            return True
        elif device_map == "cpu":
            print("✗ FAILURE: device_map is 'cpu'")
            print("  The model will load on CPU (slow!)")
            print()
            print("Troubleshooting:")
            print("  1. Check that env var uses DOUBLE underscores:")
            print("     CELESTIAL_INTEGRATED_MODELS__DEVICE_MAP (not single)")
            print("  2. Check that the env var is set BEFORE importing Config")
            return False
        else:
            print(f"? UNEXPECTED: device_map is {device_map!r}")
            return False

    except Exception as e:
        print(f"✗ ERROR loading configuration: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    setup_runpod_env()
    success = test_config()

    print()
    print("=" * 60)
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Configuration test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
