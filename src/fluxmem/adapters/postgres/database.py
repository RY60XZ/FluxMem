from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_database_engine(database_url: str, **kwargs: Any) -> Engine:
    """Create a synchronous SQLAlchemy engine for FluxMem."""

    return create_engine(database_url, **kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create sessions that keep ORM objects usable after commits."""

    return sessionmaker(bind=engine, expire_on_commit=False)
