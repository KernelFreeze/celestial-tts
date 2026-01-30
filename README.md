# Celestial TTS

[![Runpod](https://api.runpod.io/badge/KernelFreeze/celestial-tts)](https://console.runpod.io/hub/KernelFreeze/celestial-tts)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/KernelFreeze/celestial-tts/publish-container.yml)
![GitHub Release](https://img.shields.io/github/v/release/KernelFreeze/celestial-tts)

A multi-lingual, multi-provider Text-to-Speech (TTS) REST API microservice built with FastAPI. Generate high-quality speech synthesis through local models with support for preset voices and custom voice cloning.

## Features

- **Multi-lingual support** - 11 languages including auto-detection
- **Multiple voice presets** - 9 built-in voices with distinct characteristics
- **Custom voice cloning** - Clone voices from audio samples
- **OpenAI-compatible API** - Drop-in replacement for OpenAI's TTS API
- **Async-first architecture** - Built on FastAPI with full async support
- **LRU model caching** - Efficient memory management for multiple models
- **Flexible configuration** - Environment variables, TOML files, or code

## Supported Languages

| Language   | Code       |
|------------|------------|
| Auto       | `auto`     |
| Chinese    | `chinese`  |
| English    | `english`  |
| French     | `french`   |
| German     | `german`   |
| Italian    | `italian`  |
| Japanese   | `japanese` |
| Korean     | `korean`   |
| Portuguese | `portuguese` |
| Russian    | `russian`  |
| Spanish    | `spanish`  |

## Preset Voices

- Vivian
- Serena
- Uncle_Fu
- Dylan
- Eric
- Ryan
- Aiden
- Ono_Anna
- Sohee

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended) or CPU
- ~3.5GB RAM/VRAM per loaded model

## Quick Start

### Using Pre-built Container (Recommended)

The easiest way to run Celestial TTS is using the pre-built container image:

```bash
# Pull the latest image
podman pull ghcr.io/kernelfreeze/celestial-tts:latest

# Run with GPU support (requires NVIDIA Container Toolkit)
podman run --gpus all -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/kernelfreeze/celestial-tts:latest

# Run on CPU only
podman run -p 8080:8080 \
  -e CELESTIAL_INTEGRATED_MODELS_DEVICE_MAP=cpu \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/kernelfreeze/celestial-tts:latest

# With Docker
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/kernelfreeze/celestial-tts:latest
```

**Persistent Configuration:**

To persist database and configuration:

```bash
podman run --gpus all -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  -e CELESTIAL_DATABASE_URL="sqlite+aiosqlite:///data/database.db" \
  ghcr.io/kernelfreeze/celestial-tts:latest
  
# With Docker
docker run --gpus all -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  -e CELESTIAL_DATABASE_URL="sqlite+aiosqlite:///data/database.db" \
  ghcr.io/kernelfreeze/celestial-tts:latest
```

### Building from Source

#### Installation

```bash
# Clone the repository
git clone https://github.com/CelesteLove/celestial-tts.git
cd celestial-tts

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

#### Running Locally

```bash
# Using uv
uv run celestial-tts

# Run with custom host/port
uv run celestial-tts --host 0.0.0.0 --port 8000

# Or using pip
python main.py

# Run with custom host/port
python main.py --host 0.0.0.0 --port 8000
```

#### Building Container from Source

```bash
# Build the container image
podman build -t celestial-tts .

# Or with Docker
docker build -t celestial-tts .

# Run your built image
podman run --gpus all -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  celestial-tts
```

## Authentication

Most API routes require a valid auth token. Tokens use the format `sk-ct-v1-<base64>` and are passed via the `Authorization` header as a Bearer token.

### Creating Your First Token

You have two options for creating an initial auth token:

#### Option 1: Automatic Bootstrap Token (Recommended for Containers)

Set the `CELESTIAL_BOOTSTRAP_CREATE_TOKEN` environment variable to automatically create a token on first startup:

```bash
# With podman/docker
podman run --gpus all -p 8080:8080 \
  -e CELESTIAL_BOOTSTRAP_CREATE_TOKEN=true \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/kernelfreeze/celestial-tts:latest

# The token will be printed to stdout on first startup
# Example output:
# 2026-01-30 12:00:00 - celestial_tts - INFO - Bootstrap token created: sk-ct-v1-...
```

The bootstrap token is only created if no tokens exist in the database. On subsequent restarts with persistent storage, token creation is skipped.

**With persistent storage:**

```bash
podman run --gpus all -p 8080:8080 \
  -e CELESTIAL_BOOTSTRAP_CREATE_TOKEN=true \
  -e CELESTIAL_DATABASE_URL="sqlite+aiosqlite:///data/database.db" \
  -v ./data:/app/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/kernelfreeze/celestial-tts:latest
```

#### Option 2: Manual Token Creation via CLI

When running from source, use the CLI tool to create tokens:

```bash
# Create a token that never expires
uv run celestial-tts-create-token --name "My API Token"

# Create a token that expires in 30 days
uv run celestial-tts-create-token --name "Temporary Token" --expires-in 30

# Quiet mode (only output the token)
uv run celestial-tts-create-token --name "Dev Token" --quiet

# Short flags
uv run celestial-tts-create-token -n "Dev Token" -e 7 -q
```

The command outputs the token details:

```
Token created successfully!

  ID:         01234567-89ab-cdef-0123-456789abcdef
  Name:       My API Token
  Created:    2026-01-23T12:00:00
  Expires:    Never

  Token:      sk-ct-v1-MDEyMzQ1NjctODlhYi...

Store this token securely - it cannot be retrieved later.
```

### Using Tokens

Include the token in your requests:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Authorization: Bearer sk-ct-v1-..." \
  -H "Content-Type: application/json" \
  -d '{"model_id": "qwen3-tts-preset", "text": "Hello!", "language": "english", "speaker": "Vivian"}'
```

### Public vs Protected Routes

The API documentation is available at:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Configuration

### Environment Variables

```bash
# Bootstrap
CELESTIAL_BOOTSTRAP_CREATE_TOKEN=false  # Auto-create token on first startup

# Database
CELESTIAL_DATABASE_URL="sqlite+aiosqlite:///database.db"
CELESTIAL_DATABASE_URL="postgresql+asyncpg://user:pass@host/db"  # Production

# Model settings
CELESTIAL_INTEGRATED_MODELS_ENABLED=true
CELESTIAL_INTEGRATED_MODELS_MAX_LOADED_MODELS=2
CELESTIAL_INTEGRATED_MODELS_DEVICE_MAP="cuda:0"  # Or "cpu"
```

### TOML Configuration

Create `config.toml` or `~/.config/celestial-tts/config.toml`:

```toml
[bootstrap]
create_token = false

[database]
url = "sqlite+aiosqlite:///database.db"

[integrated_models]
enabled = true
max_loaded_models = 2
device_map = "cuda:0"
```

## API Reference

Celestial TTS provides two API styles:

1. **OpenAI-compatible API** (`/v1/audio/speech`) - Drop-in replacement for OpenAI's TTS API
2. **Native API** (`/v1/generate`) - Full access to all features and customization

### Health Check

```http
GET /api/health
```

Returns service health status.

### OpenAI-Compatible Speech Synthesis

```http
POST /api/v1/audio/speech
Content-Type: application/json
Authorization: Bearer sk-ct-v1-...

{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Hello, world!",
  "response_format": "mp3",
  "speed": 1.0
}
```

This endpoint is fully compatible with OpenAI's `/v1/audio/speech` API, allowing you to use existing OpenAI client libraries with Celestial TTS.

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier: `tts-1`, `tts-1-hd`, or native model IDs |
| `input` | string | Yes | Text to synthesize (max 4096 characters) |
| `voice` | string | Yes | Voice name (OpenAI or native speaker names) |
| `response_format` | string | No | Audio format: `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | float | No | Playback speed (0.25 to 4.0, default: 1.0) |
| `instructions` | string | No | Voice control instructions (voice-design model only) |

**Voice Mapping:**

OpenAI voice names are automatically mapped to native speakers:

| OpenAI Voice | Native Speaker |
|--------------|----------------|
| `alloy` | Vivian |
| `echo` | Dylan |
| `fable` | Serena |
| `onyx` | Eric |
| `nova` | Aiden |
| `shimmer` | Sohee |

You can also use native speaker names directly (`Vivian`, `Ryan`, etc.) or custom speaker UUIDs.

**Model Mapping:**

| OpenAI Model | Native Model |
|--------------|--------------|
| `tts-1` | qwen3-tts-preset |
| `tts-1-hd` | qwen3-tts-preset |

**Response:**

Returns raw audio bytes with appropriate `Content-Type` header (`audio/mpeg`, `audio/wav`, etc.).

**Usage with OpenAI Python Client:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/api/v1",
    api_key="sk-ct-v1-..."  # Your Celestial TTS token
)

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Hello from Celestial TTS!"
)

response.stream_to_file("output.mp3")
```

**Usage with cURL:**

```bash
curl -X POST http://localhost:8080/api/v1/audio/speech \
  -H "Authorization: Bearer sk-ct-v1-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "alloy",
    "input": "Hello, world!",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Generate Speech (Native API)

```http
POST /generate
Content-Type: application/json

{
  "model_id": "qwen3-tts-preset",
  "text": "Hello, world!",
  "language": "english",
  "speaker": "Vivian",
  "provider": "local"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | `qwen3-tts-preset`, `qwen3-tts-voice-clone`, or `qwen3-tts-voice-design` |
| `text` | string/array | Yes | Text to synthesize |
| `language` | string | Yes | Language code (see supported languages) |
| `speaker` | string | Yes | Preset name or custom speaker UUID |
| `instruct` | string | No | Voice instruction for styling |
| `provider` | string | No | `local` (default) |
| `top_k` | int | No | Sampling top-k |
| `top_p` | float | No | Sampling top-p |
| `temperature` | float | No | Sampling temperature |
| `repetition_penalty` | float | No | Repetition penalty |
| `max_new_tokens` | int | No | Maximum tokens to generate |

**Response:**

```json
{
  "status": "ok",
  "wavs": ["base64-encoded-wav-data"],
  "sampling_rate": 24000
}
```

### List Speakers

```http
GET /speakers?model_id=qwen3-tts-preset
```

**Response:**

```json
{
  "status": "ok",
  "speakers": [
    {"id": "Vivian", "name": "Vivian", "created_at": "..."},
    {"id": "uuid", "name": "Custom Voice", "created_at": "..."}
  ]
}
```

### Create Custom Speaker (Voice Cloning)

```http
POST /speakers
Content-Type: application/json

{
  "model_id": "qwen3-tts-voice-clone",
  "name": "My Voice",
  "text": "The transcript of what is said in the audio",
  "audio": "https://example.com/reference.wav"
}
```

The `audio` field accepts either an HTTP(S) URL or base64-encoded audio data. Local file paths are not supported for security reasons.

**Response:**

```json
{
  "status": "ok",
  "speaker": {
    "id": "uuid7-string",
    "name": "My Voice",
    "created_at": "2024-01-01T12:00:00"
  }
}
```

### Generate with Voice Design

Voice design generates speech with a dynamically created voice based on a text description. Unlike voice cloning, no audio sample is needed—the voice is synthesized from the `instruct` parameter.

```http
POST /generate
Content-Type: application/json

{
  "model_id": "qwen3-tts-voice-design",
  "text": "Hello, this is my designed voice!",
  "language": "english",
  "speaker": "generated",
  "instruct": "A young female with an energetic and cheerful tone"
}
```

**Response:**

```json
{
  "status": "ok",
  "wavs": ["base64-encoded-wav-data"],
  "sampling_rate": 24000
}
```

### Delete Custom Speaker

```http
DELETE /speakers
Content-Type: application/json

{
  "model_id": "qwen3-tts-voice-clone",
  "speaker_id": "uuid7-string",
  "provider": "local"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Model ID that supports custom speakers |
| `speaker_id` | string | Yes | UUID of the custom speaker to delete |
| `provider` | string | No | `local` (default) |

**Response:**

```json
{
  "status": "ok",
  "message": "Speaker deleted successfully"
}
```

## Models

For now, only the following local models are supported. I plan to add support for remote API end-points and more local models in the future.

### qwen3-tts-preset

Uses Qwen3-TTS with fixed preset voices. Best for quick, consistent results.

### qwen3-tts-voice-clone

Supports voice cloning from audio samples. Create unique voices by providing a reference audio file and its transcript.

### qwen3-tts-voice-design

Supports custom voice design. Create unique voices by providing reference text and voice instructions describing the desired voice characteristics.

## Usage Examples

### OpenAI-Compatible API

#### Python with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/api/v1",
    api_key="sk-ct-v1-..."  # Your Celestial TTS token
)

# Generate speech
response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Welcome to Celestial TTS!",
    response_format="mp3"
)

# Save to file
response.stream_to_file("output.mp3")
```

#### cURL

```bash
curl -X POST http://localhost:8080/api/v1/audio/speech \
  -H "Authorization: Bearer sk-ct-v1-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "alloy",
    "input": "Hello from the command line!"
  }' \
  --output speech.mp3
```

### Native API

#### Python with requests

```python
import requests
import base64

TOKEN = "sk-ct-v1-..."  # Your auth token

response = requests.post(
    "http://localhost:8080/api/v1/generate",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={
        "model_id": "qwen3-tts-preset",
        "text": "Welcome to Celestial TTS!",
        "language": "english",
        "speaker": "Vivian"
    }
)

data = response.json()
audio_bytes = base64.b64decode(data["wavs"][0])

with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

#### cURL

```bash
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Authorization: Bearer sk-ct-v1-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-tts-preset",
    "text": "Hello from the command line!",
    "language": "english",
    "speaker": "Dylan"
  }' | jq -r '.wavs[0]' | base64 -d > output.wav
```

## Testing

Celestial TTS includes comprehensive test suites using both pytest (Python) and hurl (HTTP).

### Running Tests with pytest

The pytest test suite tests the OpenAI-compatible API using the official OpenAI Python client.

```bash
# Install test dependencies
uv sync --extra test

# Or with pip
pip install -e ".[test]"

# Start the server in one terminal
uv run celestial-tts

# In another terminal, set your token and run tests
export CELESTIAL_TTS_TOKEN="sk-ct-v1-..."
pytest tests/test_openai_client.py -v

# Run specific test classes
pytest tests/test_openai_client.py::TestVoiceMapping -v
pytest tests/test_openai_client.py::TestAudioFormats -v

# Run with custom base URL
CELESTIAL_TTS_BASE_URL="http://localhost:8000/api/v1" pytest tests/test_openai_client.py -v
```

**Test Coverage:**
- Basic speech generation (tts-1, tts-1-hd models)
- All OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer)
- All native speaker names (Vivian, Dylan, etc.)
- All audio formats (mp3, wav, opus, flac, pcm)
- Speed parameter (0.25 to 4.0)
- Multilingual content (10+ languages)
- Error handling (invalid inputs, authentication)
- Edge cases (long text, special characters, mixed languages)

### Running Tests with hurl

Hurl tests provide HTTP-level testing for all API endpoints.

```bash
# Install hurl (https://hurl.dev)
# On Linux:
curl -LO https://github.com/Orange-OpenSource/hurl/releases/download/5.0.1/hurl_5.0.1_amd64.deb
sudo dpkg -i hurl_5.0.1_amd64.deb

# On macOS:
brew install hurl

# Configure your token in tests/hurl/vars.env
echo 'token=sk-ct-v1-YOUR_TOKEN_HERE' > tests/hurl/vars.env
echo 'base_url=http://127.0.0.1:8080' >> tests/hurl/vars.env

# Start the server
uv run celestial-tts

# Run all hurl tests
hurl --variables-file tests/hurl/vars.env --test tests/hurl/*.hurl

# Run specific test files
hurl --variables-file tests/hurl/vars.env --test tests/hurl/test_openai_speech.hurl
hurl --variables-file tests/hurl/vars.env --test tests/hurl/test_generate.hurl
hurl --variables-file tests/hurl/vars.env --test tests/hurl/test_speakers.hurl

# Run with verbose output
hurl --variables-file tests/hurl/vars.env --test --verbose tests/hurl/test_openai_speech.hurl
```

**Available Test Files:**
- `test_openai_speech.hurl` - OpenAI-compatible /v1/audio/speech endpoint (48 tests)
- `test_generate.hurl` - Native /v1/generate endpoint with all models (34 tests)
- `test_speakers.hurl` - Speaker management and voice cloning (25 tests)

## License

MIT License
