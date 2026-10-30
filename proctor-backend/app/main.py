from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import Base, engine
from .routers import alerts, sessions

from fastapi.responses import RedirectResponse

Path(settings.clips_dir).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)  # swap for Alembic migrations before production
    yield


app = FastAPI(title="Exam Hall Proctoring API", version="0.1.0", lifespan=lifespan)
app.include_router(sessions.router)
app.include_router(alerts.router)
app.mount("/clips", StaticFiles(directory=settings.clips_dir), name="clips")


@app.get("/health")
def health():
    return {"status": "ok"}



@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")