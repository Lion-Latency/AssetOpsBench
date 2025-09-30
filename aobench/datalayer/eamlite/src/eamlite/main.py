import inspect

import eamlite.eam_models as models
from eamlite.crud_generator import create_crud_router
from eamlite.database import init_db
from fastapi import FastAPI
from sqlmodel import SQLModel

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
