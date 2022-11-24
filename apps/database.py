# -*- encoding: utf-8 -*-

# database.py

import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir, "db.sqlite3")

engine = create_engine(
    SQLALCHEMY_DATABASE_URI, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_session(cleanup=False):
    session = Session(bind=engine)
    #Base.metadata.create_all(engine)
    try:
        return session
    except Exception:
        session.rollback()
    finally:
        session.close()

    if cleanup:
        pass
        #Base.metadata.drop_all(engine)