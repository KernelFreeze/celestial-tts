# Celestial TTS

A multi-lingual, multi-provider Text-to-Speech (TTS) REST API microservice built with FastAPI. Generate high-quality speech synthesis through local models with support for preset voices and custom voice cloning.

## Features

- **Multi-lingual support** - 11 languages including auto-detection
- **Multiple voice presets** - 9 built-in voices with distinct characteristics
- **Custom voice cloning** - Create personalized voices via voice design
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

## Installation

```bash
# Clone the repository
git clone https://github.com/CelesteLove/celestial-tts.git
cd celestial-tts

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

## Quick Start

### Using uv
```bash
uv run celestial-tts

# Run with custom host/port
uv run celestial-tts --host 0.0.0.0 --port 8000
```

### Or using pip
```bash
# Run with defaults (localhost:8080)
python main.py

# Run with custom host/port
python main.py --host 0.0.0.0 --port 8000
```

## Authentication

Most API routes require a valid auth token. Tokens use the format `sk-ct-v1-<base64>` and are passed via the `Authorization` header as a Bearer token.

### Creating Your First Token

Before using protected endpoints, create a bootstrap token using the CLI:

```bash
# Create a token that never expires
uv run celestial-tts-create-token --name "My API Token"

# Create a token that expires in 30 days
uv run celestial-tts-create-token --name "Temporary Token" --expires-in 30

# Short flags
uv run celestial-tts-create-token -n "Dev Token" -e 7
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

| Route | Authentication |
|-------|----------------|
| `GET /health` | Public |
| `POST /generate` | Required |
| `GET /speakers` | Required |
| `POST /speakers` | Required |
| `DELETE /speakers/{id}` | Required |
| `GET /auth/tokens` | Required |
| `POST /auth/tokens` | Required |
| `POST /auth/tokens/verify` | Required |
| `POST /auth/tokens/{id}/revoke` | Required |
| `DELETE /auth/tokens/{id}` | Required |

The API documentation is available at:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Configuration

### Environment Variables

```bash
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
[database]
url = "sqlite+aiosqlite:///database.db"

[integrated_models]
enabled = true
max_loaded_models = 2
device_map = "cuda:0"
```

## API Reference

### Health Check

```http
GET /health
```

Returns service health status.

### Generate Speech

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
| `model_id` | string | Yes | `qwen3-tts-preset` or `qwen3-tts-custom` |
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

### Create Custom Speaker

```http
POST /speakers
Content-Type: application/json

{
  "model_id": "qwen3-tts-custom",
  "name": "My Voice",
  "text": "Reference text for voice design",
  "language": "english",
  "instruct": "A young female with an energetic tone"
}
```

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

### Delete Custom Speaker

```http
DELETE /speakers/{speaker_id}?model_id=qwen3-tts-custom
```

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

### qwen3-tts-custom

Supports custom voice cloning and voice design. Create unique voices by providing reference text and voice instructions.

## Usage Examples

### Python

```python
import requests
import base64

TOKEN = "sk-ct-v1-..."  # Your auth token

response = requests.post(
    "http://localhost:8080/generate",
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

### cURL

```bash
curl -X POST http://localhost:8080/generate \
  -H "Authorization: Bearer sk-ct-v1-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-tts-preset",
    "text": "Hello from the command line!",
    "language": "english",
    "speaker": "Dylan"
  }' | jq -r '.wavs[0]' | base64 -d > output.wav
```

## License

MIT License
