from fastapi import FastAPI

from celestial_tts.config import Config
from celestial_tts.model import ModelState
from celestial_tts.routes import health

app = FastAPI()
app.state.config = Config()
app.state.models = ModelState(config=app.state.config)

app.include_router(health.router)
