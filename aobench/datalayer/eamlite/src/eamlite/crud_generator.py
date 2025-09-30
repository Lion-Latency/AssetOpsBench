import datetime
import decimal
from typing import Any, List, Optional, Type, get_args, get_origin

from eamlite.database import get_session
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, create_model
from sqlalchemy.orm import Mapped
from sqlalchemy.sql.sqltypes import (Boolean, Date, Float, Integer, Numeric,
                                     String)
from sqlmodel import Session, SQLModel, select


def sqlalchemy_type_to_python(sa_type):
    """Map SQLAlchemy column types to Python types for Pydantic."""
    if isinstance(sa_type, Integer):
        return int
    if isinstance(sa_type, String):
        return str
    if isinstance(sa_type, Float):
        return float
    if isinstance(sa_type, Boolean):
        return bool
    if isinstance(sa_type, Numeric):
        return decimal.Decimal
    if isinstance(sa_type, Date):
        return datetime.date
    return str  # fallback to string if unknown


def generate_filter_model(model: type[SQLModel]):
    """Generate a Pydantic model with *optional* query parameters for filtering."""
    fields = {}

    for column in model.__table__.columns:
        py_type = sqlalchemy_type_to_python(column.type)

        # Make every filter param optional, regardless of nullable status
        # (because a filter can always be omitted)
        optional_type = Optional[py_type]

        # Use Field(default=None) so Pydantic doesn't require it
        fields[column.name] = (
            optional_type,
            Field(default=None, description=f"Filter by {column.name}"),
        )

    FilterModel = create_model(
        f"{model.__name__}Filter",
        __base__=BaseModel,
        **fields,
    )
    FilterModel.model_config = ConfigDict(extra="ignore")
    return FilterModel


def convert_value(value: str, target_type: Any):
    try:
        if target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == decimal.Decimal:
            return decimal.Decimal(value)
        elif target_type == bool:
            return value.lower() in ("true", "1", "yes")
        elif target_type == datetime.date:
            return datetime.date.fromisoformat(value)
        return value
    except Exception:
        return value


def create_crud_router(model: Type[SQLModel]) -> APIRouter:
    router = APIRouter()
    model_name = model.__name__

    FilterModel = generate_filter_model(model)

    # Create
    @router.post("/", response_model=model)
    def create(item: model, session: Session = Depends(get_session)):
        session.add(item)
        session.commit()
        session.refresh(item)
        return item

    @router.get("/", response_model=List[model])
    def read_all(
        filters: FilterModel = Depends(),
        session: Session = Depends(get_session),
        limit: int = Query(10, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        query = select(model)

        for field_name, value in filters.dict(exclude_none=True).items():
            column = getattr(model, field_name)
            query = query.where(column == value)

        query = query.limit(limit).offset(offset)
        return session.exec(query).all()

    # Read one
    @router.get("/{item_id}", response_model=model)
    def read_one(item_id: int, session: Session = Depends(get_session)):
        db_item = session.get(model, item_id)
        if not db_item:
            raise HTTPException(status_code=404, detail=f"{model_name} not found")
        return db_item

    # Update
    @router.put("/{item_id}", response_model=model)
    def update(item_id: int, new_item: model, session: Session = Depends(get_session)):
        db_item = session.get(model, item_id)
        if not db_item:
            raise HTTPException(status_code=404, detail=f"{model_name} not found")

        for field, value in new_item.dict(exclude_unset=True).items():
            setattr(db_item, field, value)

        session.add(db_item)
        session.commit()
        session.refresh(db_item)
        return db_item

    # Delete
    @router.delete("/{item_id}")
    def delete(item_id: int, session: Session = Depends(get_session)):
        db_item = session.get(model, item_id)
        if not db_item:
            raise HTTPException(status_code=404, detail=f"{model_name} not found")
        session.delete(db_item)
        session.commit()
        return {"ok": True}

    return router
