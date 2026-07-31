"""Postgres connection helper.

Uses psycopg 3 and registers the pgvector adapter so Python lists / numpy arrays
round-trip to the ``vector`` column type. ``register_vector`` is imported from
``pgvector.psycopg`` (the psycopg-3 binding); the psycopg-2 binding lives at
``pgvector.psycopg2`` instead — this project uses psycopg 3.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import psycopg
from pgvector.psycopg import register_vector

from sec_rag.config import Secrets

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool


def new_connection(
    secrets: Secrets | None = None, *, autocommit: bool = False
) -> psycopg.Connection:
    """Open a pgvector-aware connection. Caller is responsible for closing it.

    Used by the long-lived QueryEngine, which holds one connection across many
    queries instead of reconnecting per request.

    ``autocommit=True`` is the right mode for that long-lived read connection:
    psycopg3 otherwise opens an implicit transaction on the first query and
    leaves it open, so an idle engine sits "idle in transaction" — which Neon
    terminates (IdleInTransactionSessionTimeout), breaking the next request.
    Read-only SELECTs need no transaction, so autocommit avoids the lingering
    one entirely. Ingest keeps the default (False): it batches DELETE+INSERT per
    document and commits explicitly, which must stay atomic.
    """
    secrets = secrets or Secrets()
    secrets.require("database_url")
    conn = psycopg.connect(secrets.database_url, autocommit=autocommit)
    register_vector(conn)
    return conn


def new_pool(
    secrets: Secrets | None = None,
    *,
    min_size: int | None = None,
    max_size: int | None = None,
    autocommit: bool = True,
) -> ConnectionPool:
    """Open a pgvector-aware psycopg-3 connection pool for the query path.

    Replaces the single long-lived connection the QueryEngine used to hold. That
    one connection was thread-safe but **serialized** concurrent retrieval —
    psycopg locks a connection for the duration of a query, so N simultaneous
    ``/query`` requests queued their DB work one-behind-another (a throughput
    ceiling and a tail-latency contributor under load, the documented debt). A
    pool runs them in parallel and transparently reconnects a connection Neon
    dropped, removing the single-socket point of failure the old manual
    reconnect-on-OperationalError guarded against by hand.

    ``autocommit=True``: read-only SELECTs need no transaction, and it avoids the
    idle-in-transaction state Neon terminates (same reasoning as new_connection).
    ``register_vector`` runs via ``configure`` on **every** pooled connection, so
    the pgvector ``vector`` adapter is installed whichever connection serves a
    query. Sizes are env-tunable (``SEC_RAG_POOL_MIN`` / ``SEC_RAG_POOL_MAX``);
    defaults min=1 (one warm connection — no first-query connect penalty) / max=8
    (headroom for concurrent bursts, conservative for Neon's connection cap).
    """
    from psycopg_pool import ConnectionPool  # lazy: ingest/local-eval paths don't need it

    secrets = secrets or Secrets()
    secrets.require("database_url")
    if min_size is None:
        min_size = int(os.environ.get("SEC_RAG_POOL_MIN", "1"))
    if max_size is None:
        max_size = int(os.environ.get("SEC_RAG_POOL_MAX", "8"))

    pool = ConnectionPool(
        secrets.database_url,
        kwargs={"autocommit": autocommit},
        min_size=min_size,
        max_size=max_size,
        configure=register_vector,
        open=False,
        name="sec-rag-query",
    )
    # Open + wait so a bad DATABASE_URL fails loudly HERE (as connect-at-startup
    # used to), not at the first query. Bounded so a dead DB can't hang startup.
    pool.open(wait=True, timeout=10)
    return pool


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
