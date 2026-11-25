"""FastAPI web application package for findex."""

from findex.web.app import app, create_app
from findex.web.config import Settings, get_settings

__all__ = ["Settings", "app", "create_app", "get_settings"]