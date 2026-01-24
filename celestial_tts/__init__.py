import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.injectors import get_authenticated_token
from celestial_tts.middleware import RequestLoggingMiddleware
from celestial_tts.model import ModelState
from celestial_tts.routes import auth, generate, health, speakers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application lifespan")
    try:
        logger.info("Initializing database")
        await app.state.database.init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.error(f"Database URL: {app.state.config.database.url}")
        logger.error("Please ensure your database is running and accessible.")
        raise
    yield
    # Shutdown
    logger.info("Shutting down application")
    await app.state.database.close()
    logger.info("Database closed")


app = FastAPI(lifespan=lifespan)
app.state.config = Config()
app.state.database = Database(app.state.config.database.url)
app.state.models = ModelState(config=app.state.config)

# Add request logging middleware if enabled
if app.state.config.logging.log_requests:
    app.add_middleware(RequestLoggingMiddleware)
    logger.info("Request logging middleware enabled")

v1 = APIRouter(
    prefix="/v1", tags=["v1"], dependencies=[Depends(get_authenticated_token)]
)
v1.include_router(generate.router)
v1.include_router(speakers.router)
v1.include_router(auth.router)

api = APIRouter(prefix="/api")
api.include_router(health.router)
api.include_router(v1)

app.include_router(api)
