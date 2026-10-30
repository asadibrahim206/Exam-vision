import secrets

from fastapi import Header, HTTPException

from .config import settings


def require_detector_key(x_api_key: str = Header(...)) -> None:
    """Only the detection worker may create alerts / upload clips."""
    if not secrets.compare_digest(x_api_key, settings.detector_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
