from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.injectors import get_authenticated_token
from celestial_tts.model import ModelState
from celestial_tts.routes import auth, generate, health, speakers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await app.state.database.init_db()
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        print(f"Database URL: {app.state.config.database.url}")
        print("Please ensure your database is running and accessible.")
        raise
    yield
    # Shutdown
    await app.state.database.close()


app = FastAPI(lifespan=lifespan)
app.state.config = Config()
app.state.database = Database(app.state.config.database.url)
app.state.models = ModelState(config=app.state.config)

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
