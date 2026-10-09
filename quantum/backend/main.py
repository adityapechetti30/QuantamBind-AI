"""Entry point proxy forwarding to backend.app.main:app."""

from backend.app.main import app

__all__ = ["app"]
