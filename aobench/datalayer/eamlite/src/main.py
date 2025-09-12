import inspect
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlmodel import SQLModel

import eam_models as models
from crud_generator import create_crud_router
from database import init_db

app = FastAPI(title="EAM API", version="1.0.0")


@app.on_event("startup")
def on_startup():
    init_db()


# Dynamically add routers for all SQLModel models
for name, cls in inspect.getmembers(models, inspect.isclass):
    if issubclass(cls, SQLModel) and getattr(cls, "__table__", None) is not None:
        router = create_crud_router(cls)
        app.include_router(
            router, prefix=f"/{cls.__tablename__.lower()}", tags=[cls.__name__]
        )
