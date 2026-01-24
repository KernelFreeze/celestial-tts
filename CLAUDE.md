# CLAUDE.md - AI Assistant Guide for Celestial TTS

This document provides comprehensive guidance for AI assistants working on the Celestial TTS codebase.

## Project Overview

**celestial-tts** is a multi-lingual, multi-provider Text-to-Speech (TTS) REST API microservice built with FastAPI. It provides speech synthesis through local models (currently Qwen3-TTS) with support for preset voices and custom voice cloning.

- **Version:** 0.1.0
- **Python:** 3.12+
- **Package Manager:** uv (modern, fast Python package manager)
- **Architecture:** Async-first FastAPI microservice with SQLModel ORM

## Quick Stats

- **Lines of Code:** ~378
- **Python Files:** 21
- **API Endpoints (Defined):** 4 (only 1 currently wired to app)
- **Supported Languages:** 10 (zh, en, ja, ko, de, fr, ru, pt, es, it)
- **Local Model Types:** 2 (Qwen preset, Qwen custom)
- **Dependencies:** 22 production dependencies

---

## Project Structure

```
/home/user/celestial-tts/
├── main.py                          # CLI entry point (Typer)
├── pyproject.toml                   # Project metadata, dependencies
├── uv.lock                          # Dependency lock file
├── .gitignore
├── .python-version                  # 3.12
├── README.md                        # Currently empty
└── celestial_tts/                   # Main package
    ├── __init__.py                  # FastAPI app + lifespan management
    ├── config.py                    # Multi-source configuration
    ├── injectors.py                 # FastAPI dependency injection
    ├── database/
    │   ├── __init__.py              # Database engine, session mgmt
    │   ├── utils.py                 # Tensor serialization
    │   ├── model/
    │   │   └── custom_speaker.py    # QwenCustomSpeaker SQLModel
    │   └── controller/
    │       └── custom_speaker.py    # CRUD operations
    ├── model/                       # TTS model abstraction layer
    │   ├── __init__.py              # ModelState classes
    │   ├── types/
    │   │   └── __init__.py          # Type definitions
    │   ├── local/
    │   │   ├── __init__.py          # LocalTTSModel base class
    │   │   ├── factory.py           # LocalTTSFactory, LocalTTSType
    │   │   ├── model_cache.py       # LRU model cache
    │   │   └── qwen/
    │   │       ├── preset.py        # Fixed voices
    │   │       └── custom.py        # Voice cloning
    │   └── remote/
    │       └── __init__.py          # RemoteTTSModel (not implemented)
    └── routes/                      # API endpoints
        ├── __init__.py
        ├── health.py                # GET /health (wired)
        ├── generate.py              # POST /generate (NOT wired yet)
        └── speakers.py              # GET/POST /speakers (NOT wired yet)
```

---

## Technology Stack

### Core Framework
- **FastAPI** (>=0.128.0) - Async web framework with automatic OpenAPI docs
- **Hypercorn** (>=0.18.0) - ASGI HTTP server
- **Typer** (>=0.21.1) - CLI framework

### Database
- **SQLModel** (>=0.0.31) - SQL ORM with Pydantic integration
- **aiosqlite** (>=0.22.1) - Async SQLite driver (default)
- **asyncpg** (>=0.31.0) - Async PostgreSQL driver (production-ready)

### Machine Learning
- **Qwen TTS** (>=0.0.4) - Qwen3 Text-to-Speech models (1.7B parameters)
- **PyTorch** (>=2.10.0) - Deep learning framework
- **Flash Attention** (>=2.8.3) - Optimized attention mechanism
- **safetensors** (>=0.7.0) - Safe tensor serialization
- **soundfile** (>=0.13.1) - Audio file I/O

### Configuration
- **Pydantic Settings** (>=2.12.0) - Settings with validation
- **tomli-w** (>=1.2.0) - TOML serialization
- **uuid-utils** (>=0.14.0) - UUID7 support (time-ordered UUIDs)

---

## Key Architecture Patterns

### 1. Async-First Design
- All database operations use async/await
- FastAPI lifespan events manage startup/shutdown
- Thread pools used for synchronous ML model inference

### 2. Factory Pattern
- `LocalTTSFactory` creates model instances based on `LocalTTSType` enum
- Device mapping (CPU/GPU) configured at factory level

### 3. LRU Model Cache
- `LocalModelCache` implements LRU eviction policy
- Lazy loading via `get_or_put()`
- Explicit `unload()` calls on eviction to free GPU/CPU memory
- Configurable capacity via `max_loaded_models` setting

### 4. Dependency Injection
Three FastAPI dependencies in `injectors.py`:
```python
get_config()    # Injects Config from app.state
get_models()    # Injects ModelState from app.state
get_database()  # Injects Database from app.state
```

### 5. Abstract Base Classes
- `LocalTTSModel` - Generic base with typed language/speaker support
- `RemoteModelModel` - Placeholder for future remote providers
- Use generics for type-safe language and speaker parameters

### 6. Multi-Source Configuration
Priority order:
1. Explicit initialization arguments
2. Environment variables (with prefix)
3. TOML files (`config.toml` or `~/.config/celestial-tts/config.toml`)
4. Defaults

---

## Configuration Reference

### Environment Variables

All config values can be set via environment variables with `CELESTIAL_` prefix:

```bash
# Database configuration
CELESTIAL_DATABASE_URL="sqlite+aiosqlite:///database.db"
CELESTIAL_DATABASE_URL="postgresql+asyncpg://user:pass@host/db"  # Production

# Local models configuration
CELESTIAL_INTEGRATED_MODELS_ENABLED=true
CELESTIAL_INTEGRATED_MODELS_MAX_LOADED_MODELS=2
CELESTIAL_INTEGRATED_MODELS_DEVICE_MAP="cpu"       # Or "cuda:0", "cuda:1", etc.
```

Nested settings support `__` delimiter:
```bash
CELESTIAL_DATABASE__URL="sqlite+aiosqlite:///db.db"
```

### TOML Configuration

Location: `config.toml` or `~/.config/celestial-tts/config.toml`

```toml
[database]
url = "sqlite+aiosqlite:///database.db"

[integrated_models]
enabled = true
max_loaded_models = 2
device_map = "cpu"
```

Config file is auto-created with defaults if missing.

### Configuration Classes

See `celestial_tts/config.py:1`:
- `DatabaseConfig` - Database URL and SSL settings
- `IntegratedModelsConfig` - Local model settings
- `Config` - Root configuration class

---

## Database Schema

### QwenCustomSpeaker Table
Location: `celestial_tts/database/model/custom_speaker.py:8`

Stores custom voice clones with serialized tensor embeddings:

```python
class QwenCustomSpeaker(SQLModel, table=True):
    id: uuid.UUID                      # UUID7 (time-ordered), PK
    name: str                          # Speaker display name
    created_at: datetime               # Creation timestamp
    ref_code: bytes | None             # Serialized tensor (optional)
    ref_spk_embedding: bytes           # Serialized speaker embedding (required)
    x_vector_only_mode: bool           # Voice clone mode flag
    icl_mode: bool                     # In-context learning mode flag
    ref_text: str | None               # Reference text for voice clone
```

**Tensor Serialization**: Uses `safetensors` format (see `celestial_tts/database/utils.py:9`)

---

## API Endpoints

### Currently Wired
- `GET /health` - Health check endpoint

### Defined but NOT Wired Yet
⚠️ **IMPORTANT**: The following routes exist but are NOT included in the main app (`celestial_tts/__init__.py:39`):

- `POST /generate` - Text-to-speech generation
- `GET /speakers` - List available speakers
- `POST /speakers` - Create custom speaker

**To enable these routes**, add to `celestial_tts/__init__.py`:
```python
from celestial_tts.routes import health, generate, speakers

app.include_router(health.router)
app.include_router(generate.router)    # Add this
app.include_router(speakers.router)    # Add this
```

### POST /generate
Location: `celestial_tts/routes/generate.py:28`

Generate speech from text.

**Request Parameters:**
- `model_id` (str) - "qwen3-tts-preset" or "qwen3-tts-custom"
- `text` (str | list[str]) - Text to synthesize
- `language` (str) - Language code: zh, en, ja, ko, de, fr, ru, pt, es, it
- `speaker` (str) - Preset name (e.g., "Vivian") or UUID for custom
- `instruct` (str, optional) - Voice instruction
- `provider` (str) - "local" only (remote not implemented)
- Generation params: `top_k`, `top_p`, `temperature`, `repetition_penalty`, `max_new_tokens`

**Response:**
```json
{
  "audio": "base64-encoded-wav-data",
  "sampling_rate": 12000
}
```

### GET /speakers
Location: `celestial_tts/routes/speakers.py:14`

List available speakers for a model.

**Query Parameters:**
- `model_id` (str) - Model identifier

**Response:**
```json
{
  "speakers": [
    {"id": "Vivian", "name": "Vivian", "created_at": null},
    {"id": "uuid", "name": "Custom Voice", "created_at": "2024-01-01T00:00:00"}
  ]
}
```

### POST /speakers
Location: `celestial_tts/routes/speakers.py:44`

Create custom speaker via voice design.

**Request Parameters:**
- `model_id` (str) - Must support custom speakers (qwen3-tts-custom)
- `name` (str) - Speaker name
- `text` (str) - Reference text for voice design
- `language` (str) - Language code
- `instruct` (str) - Voice instruction (e.g., "A young female with energetic tone")

**Response:**
```json
{
  "id": "uuid7-string",
  "name": "My Custom Voice",
  "created_at": "2024-01-01T12:00:00"
}
```

---

## Model System

### Supported Models

#### 1. Qwen3-TTS-Preset
- **Model ID:** `qwen3-tts-preset`
- **Implementation:** `celestial_tts/model/local/qwen/preset.py:19`
- **Base Model:** Qwen3-TTS-1.7B-CustomVoice
- **Preset Voices:** Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
- **Languages:** All 10 supported languages
- **Sampling Rate:** 12kHz

#### 2. Qwen3-TTS-Custom
- **Model ID:** `qwen3-tts-custom`
- **Implementation:** `celestial_tts/model/local/qwen/custom.py:22`
- **Clone Model:** Qwen3-TTS-1.7B-Base
- **Design Model:** Qwen3-TTS-1.7B-VoiceDesign
- **Features:**
  - Custom voice cloning from reference audio
  - Voice design from text instructions
  - Database persistence of embeddings
  - UUID-based speaker lookup
- **Sampling Rate:** 12kHz

### Model Cache Behavior

Location: `celestial_tts/model/local/model_cache.py:7`

- **LRU Eviction**: Least recently used models evicted when cache full
- **Capacity**: Configured via `max_loaded_models` (default: 2, min: 1)
- **Memory Management**: Explicit `unload()` called on eviction
- **Thread Safety**: Uses asyncio locks (models aren't thread-safe)

**Important for GPU usage:**
- Model loading triggers CUDA cache clear
- Unloading calls garbage collection
- Models are loaded to `device_map` specified in config

---

## Development Workflows

### Running the Application

```bash
# Install dependencies
uv sync

# Run with defaults (localhost:8080)
python main.py

# Run with custom host/port
python main.py --host 0.0.0.0 --port 8000
```

### Entry Point
Location: `main.py:6`

Typer CLI that starts Hypercorn ASGI server with async event loop.

### Application Lifecycle

Location: `celestial_tts/__init__.py:1`

**Startup:**
1. Load configuration from TOML/env
2. Initialize database (SQLite/PostgreSQL)
3. Create tables via SQLModel metadata
4. Initialize model state (local + remote)
5. Attach to `app.state`

**Shutdown:**
1. Close database connections
2. Cleanup model resources

---

## Code Conventions

### Import Organization
Standard Python conventions:
1. Standard library
2. Third-party packages
3. Local imports

### Type Hints
- **Extensive type hints throughout** - this is a strongly typed codebase
- Use `from typing import ...` for generics
- SQLModel provides Pydantic validation
- Custom types in `celestial_tts/model/types/__init__.py:3` (e.g., `NonEmptyStr`)

### Async Patterns
- Use `async def` for all I/O operations
- Database sessions: `async with`
- Model cache locks: `async with self._lock`
- Thread pools for sync model inference: `asyncio.to_thread()`

### Error Handling
- FastAPI `HTTPException` for API errors (400, 404, 422, 500, 501)
- Database errors propagate to FastAPI error handlers
- Model validation via Pydantic raises 422 Unprocessable Entity

### Naming Conventions
- **Files:** `snake_case.py`
- **Classes:** `PascalCase`
- **Functions/Variables:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private attributes:** `_leading_underscore`

---

## Common Development Tasks

### Adding a New TTS Model Provider

#### For Local Models:
1. Create new directory in `celestial_tts/model/local/`
2. Implement `LocalTTSModel[LanguageType, SpeakerType]`
3. Define language and speaker types
4. Implement required methods:
   - `generate()` - Text-to-speech synthesis
   - `get_languages()` - Supported languages
   - `get_speakers()` - Available speakers
   - `validate_language()` - Language validation
   - `validate_speaker()` - Speaker validation
   - `unload()` - Cleanup resources
5. Add to `LocalTTSType` enum in `factory.py`
6. Add factory case in `LocalTTSFactory.create()`

#### For Remote Models:
1. Implement `RemoteModelModel[LanguageType, SpeakerType]` in `celestial_tts/model/remote/`
2. Add HTTP client for API calls
3. Update `RemoteModelState` initialization
4. Update route handlers to support remote provider

### Adding a New API Endpoint

1. Create route file in `celestial_tts/routes/`
2. Use FastAPI `APIRouter()`
3. Inject dependencies via `Depends(get_config)`, etc.
4. Add type hints for request/response models
5. **Import and wire router in `celestial_tts/__init__.py`**
6. Document in this CLAUDE.md file

### Database Migrations

⚠️ **No migration system currently in place**

Current approach:
- SQLModel creates tables from metadata on startup
- Schema changes require manual migration or database recreation

**Recommendations for future:**
- Add Alembic for migrations
- Create initial migration from current schema
- Version control migration files

### Modifying Configuration

1. Update config class in `celestial_tts/config.py`
2. Add Pydantic field with default value
3. Add environment variable to this doc
4. Config file auto-regenerates with new defaults

---

## Testing (TODO)

⚠️ **No testing infrastructure currently exists**

### Recommended Setup:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",  # For FastAPI testing
    "pytest-cov>=4.1.0",
]
```

### Suggested Test Structure:
```
tests/
├── conftest.py                 # Fixtures
├── test_config.py              # Configuration tests
├── test_database.py            # Database layer tests
├── test_model_cache.py         # Model cache LRU tests
├── test_qwen_preset.py         # Qwen preset model tests
├── test_qwen_custom.py         # Qwen custom model tests
└── test_routes/
    ├── test_health.py
    ├── test_generate.py
    └── test_speakers.py
```

### Key Test Patterns:
- Use `pytest-asyncio` for async tests
- Use FastAPI `TestClient` or `AsyncClient`
- Mock ML models to avoid loading 1.7B parameter models in tests
- Use in-memory SQLite for database tests
- Test configuration precedence (env > TOML > defaults)

---

## Performance Considerations

### Model Loading
- **Cold start:** 1.7B parameter model loading can take 10-60 seconds depending on hardware
- **Memory usage:** Each model requires ~3.5GB RAM (float16) or ~7GB RAM (float32)
- **GPU memory:** Similar to RAM, depends on precision
- **LRU cache:** Tune `max_loaded_models` based on available memory

### Database
- **SQLite:** Sufficient for development and low-traffic production
- **PostgreSQL:** Recommended for production with high concurrency
- **Connection pooling:** Automatic with asyncpg driver

### Async Performance
- FastAPI uses ASGI for true async support
- Thread pools prevent blocking event loop during model inference
- Database operations are fully async

---

## Security Considerations

### Input Validation
- Pydantic validates all API inputs
- Language and speaker validation before generation
- Model ID validation against supported types

### Database
- SQL injection protection via SQLModel/SQLAlchemy
- PostgreSQL SSL support available (`celestial_tts/database/__init__.py:20`)

### Dependencies
- Use `uv` lock file for reproducible builds
- Regular dependency updates recommended
- Monitor for security advisories in PyTorch, FastAPI

### Production Deployment
- Don't expose on 0.0.0.0 without authentication
- Use reverse proxy (nginx, Caddy) for HTTPS
- Implement rate limiting
- Add API key authentication
- Monitor disk usage (generated audio, model cache)

---

## Known Issues & Limitations

### 1. Routes Not Wired
**Status:** Critical
- `generate` and `speakers` routes defined but not included in app
- Only `health` endpoint currently accessible
- **Fix:** Add router imports to `celestial_tts/__init__.py`

### 2. Remote Models Not Implemented
**Status:** Placeholder
- Abstract classes exist
- All remote provider requests return 501 Not Implemented
- `RemoteModelState` class exists but unused

### 3. No Testing Infrastructure
**Status:** Missing
- Zero test files
- No CI/CD pipeline
- Manual testing required

### 4. Empty Documentation
**Status:** Missing
- README.md is empty
- No API documentation beyond code
- No deployment guides

### 5. No Migration System
**Status:** Risk
- Schema changes require manual intervention
- No version control for database schema

### 6. Single Model Family
**Status:** Limited
- Only Qwen3-TTS models supported
- Architecture supports multiple providers but none implemented

---

## Git Workflow

### Branch Naming
- Feature branches: `claude/claude-md-<session-id>-<description>`
- Must start with `claude/` and include session ID for push authorization

### Commit Messages
Follow conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Test additions
- `chore:` - Maintenance tasks

Examples:
```
feat(routes): wire generate and speakers endpoints
fix(model-cache): prevent memory leak on eviction
docs(readme): add quickstart guide
```

### Push Protocol
```bash
# Always use -u flag for branch tracking
git push -u origin claude/claude-md-mkrjkbu54v2wffdz-BKyRP

# Retry on network errors with exponential backoff (2s, 4s, 8s, 16s)
# Max 4 retries
```

---

## Important File Locations

### Configuration
- Main config: `celestial_tts/config.py:45` (Config class)
- Database config: `celestial_tts/config.py:10` (DatabaseConfig)
- Models config: `celestial_tts/config.py:28` (IntegratedModelsConfig)

### Application Entry
- CLI entry: `main.py:6` (Typer app)
- FastAPI app: `celestial_tts/__init__.py:1` (app initialization)
- Lifespan management: `celestial_tts/__init__.py:10` (async context manager)

### Database
- Engine setup: `celestial_tts/database/__init__.py:8` (Database class)
- Custom speaker model: `celestial_tts/database/model/custom_speaker.py:8`
- CRUD controller: `celestial_tts/database/controller/custom_speaker.py:9`
- Tensor utils: `celestial_tts/database/utils.py:9`

### Model System
- Model state: `celestial_tts/model/__init__.py:19` (LocalModelState, RemoteModelState, ModelState)
- Factory: `celestial_tts/model/local/factory.py:9` (LocalTTSFactory)
- Model cache: `celestial_tts/model/local/model_cache.py:7` (LocalModelCache)
- Qwen preset: `celestial_tts/model/local/qwen/preset.py:19`
- Qwen custom: `celestial_tts/model/local/qwen/custom.py:22`

### API Routes
- Health: `celestial_tts/routes/health.py:6`
- Generate: `celestial_tts/routes/generate.py:28`
- Speakers: `celestial_tts/routes/speakers.py:14` (GET), `:44` (POST)

### Dependency Injection
- Injectors: `celestial_tts/injectors.py:5` (all three dependency functions)

---

## Quick Reference Commands

```bash
# Install dependencies
uv sync

# Run application (development)
python main.py --host localhost --port 8080

# Run application (expose to network)
python main.py --host 0.0.0.0 --port 8000

# Access API docs (after starting)
# Swagger UI: http://localhost:8080/docs
# ReDoc: http://localhost:8080/redoc

# Set configuration via environment
export CELESTIAL_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/celestial"
export CELESTIAL_INTEGRATED_MODELS_DEVICE_MAP="cuda:0"
export CELESTIAL_INTEGRATED_MODELS_MAX_LOADED_MODELS=3

# Check Python version
python --version  # Should be 3.12+

# View dependency tree
uv tree

# Add new dependency
# Edit pyproject.toml and run:
uv sync
```

---

## Troubleshooting

### Model Won't Load
- Check `device_map` setting matches available hardware
- Verify sufficient RAM/GPU memory (~3.5GB+ per model)
- Check CUDA installation if using GPU
- Review model cache size vs. available memory

### Database Connection Errors
- Verify `DATABASE_URL` format
- For PostgreSQL, ensure server is running
- Check SSL settings for remote PostgreSQL
- Permissions on SQLite file location

### Routes Return 404
- Verify router is wired in `celestial_tts/__init__.py`
- Check route path matches request
- Review FastAPI startup logs

### Import Errors
- Run `uv sync` to install dependencies
- Check Python version (3.12+ required)
- Verify virtual environment activation

### CUDA Out of Memory
- Reduce `max_loaded_models`
- Use `device_map="cpu"` instead of CUDA
- Unload unused models manually
- Check for memory leaks (models not unloading)

---

## Additional Resources

### Project Links
- **Repository:** `/home/user/celestial-tts`
- **Main Branch:** (check git config)
- **Current Branch:** `claude/claude-md-mkrjkbu54v2wffdz-BKyRP`

### External Documentation
- FastAPI: https://fastapi.tiangolo.com/
- SQLModel: https://sqlmodel.tiangolo.com/
- Qwen TTS: https://github.com/QwenLM/Qwen-Audio
- PyTorch: https://pytorch.org/docs/
- Pydantic: https://docs.pydantic.dev/

---

## For AI Assistants: Key Reminders

1. **Always read files before modifying** - Never propose changes to code you haven't seen
2. **Routes aren't wired** - Remember that `generate` and `speakers` routes need to be added to the main app
3. **No tests exist** - Be cautious when refactoring, manual testing required
4. **LRU cache is critical** - Model loading is expensive, cache behavior matters
5. **Async everywhere** - All I/O should be async, use thread pools for sync ML operations
6. **Type safety** - Maintain comprehensive type hints throughout
7. **Configuration flexibility** - Support env vars, TOML, and defaults
8. **Memory management** - Explicitly unload models, call garbage collection
9. **Database agnostic** - Code should work with both SQLite and PostgreSQL
10. **Remote models are stubs** - Don't assume remote provider functionality exists

---

**Last Updated:** 2026-01-23
**Codebase Version:** 0.1.0
**Document Version:** 1.0.0
