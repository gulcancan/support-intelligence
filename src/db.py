from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from config import get_settings

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, pool_size=10, max_overflow=20, pool_pre_ping=True)
    return _engine

SessionLocal = sessionmaker()

@contextmanager
def get_db() -> Session:
    engine = get_engine()
    session = sessionmaker(bind=engine)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_db_dependency():
    engine = get_engine()
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()
