from contextlib import asynccontextmanager

from fastapi import FastAPI

from celestial_tts.config import Config
from celestial_tts.database import Database
from celestial_tts.model import ModelState
from celestial_tts.routes import health


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

app.include_router(health.router)
