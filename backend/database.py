"""Database configuration helpers for persistent local metadata storage.

The project is currently centered on filesystem-backed model and upload storage,
but the application still exposes database-related configuration in the architecture
and deployment docs. This module keeps that configuration in a single place and
makes it easy to wire an external database later without changing callers.
"""

from __future__ import annotations

import os
from typing import Final

DATABASE_URL: Final[str] = os.getenv(
    "DATABASE_URL",
    "sqlite:///./storage/text_to_automl.db",
)


def get_database_url() -> str:
    """Return the configured database connection string."""
    return DATABASE_URL
