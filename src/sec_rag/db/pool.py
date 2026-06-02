"""Postgres connection helper.

Uses psycopg 3 and registers the pgvector adapter so Python lists / numpy arrays
round-trip to the ``vector`` column type. ``register_vector`` is imported from
``pgvector.psycopg`` (the psycopg-3 binding); the psycopg-2 binding lives at
``pgvector.psycopg2`` instead — this project uses psycopg 3.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from sec_rag.config import Secrets


def new_connection(secrets: Secrets | None = None) -> psycopg.Connection:
    """Open a pgvector-aware connection. Caller is responsible for closing it.

    Used by the long-lived QueryEngine, which holds one connection across many
    queries instead of reconnecting per request.
    """
    secrets = secrets or Secrets()
    secrets.require("database_url")
    conn = psycopg.connect(secrets.database_url)
    register_vector(conn)
    return conn


@contextmanager
def connect(secrets: Secrets | None = None) -> Iterator[psycopg.Connection]:
    """Yield a pgvector-aware connection and close it on exit.

    Raises a clear error if DATABASE_URL is unset rather than letting psycopg
    fail with an opaque DSN error.
    """
    conn = new_connection(secrets)
    try:
        yield conn
    finally:
        conn.close()
