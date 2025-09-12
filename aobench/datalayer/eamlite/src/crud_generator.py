import datetime
import decimal
from typing import Any, List, Optional, Type

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, or_
from sqlmodel import Session, SQLModel, select

from database import get_session


def generate_filter_model(model: type[SQLModel]) -> type[BaseModel]:
    """Create a Pydantic model for all columns to use as query params."""
    fields = {}
    for field_name, annotation in model.__annotations__.items():
        # Make all fields optional for filtering
        fields[field_name] = (Optional[annotation], None)
        # Also support operators
        if annotation in [int, float, decimal.Decimal, datetime.date, str]:
            for op in ["__gte", "__lte", "__gt", "__lt", "__ne", "__icontains"]:
                fields[f"{field_name}{op}"] = (Optional[annotation], None)

    FilterModel = type(f"{model.__name__}Filter", (BaseModel,), fields)
    return FilterModel


def convert_value(value: str, target_type: Any):
    """Convert query string values to the correct field type."""
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

    # Create
    @router.post("/", response_model=model)
    def create(item: model, session: Session = Depends(get_session)):
        session.add(item)
        session.commit()
        session.refresh(item)
        return item

    # Read all with filtering + pagination
    @router.get("/", response_model=List[model])
    def read_all(
        session: Session = Depends(get_session),
        limit: int = Query(10, ge=1, le=100, description="Max results to return"),
        offset: int = Query(0, ge=0, description="Results offset for pagination"),
        **filters,
    ):
        query = select(model)

        for field_expr, value in filters.items():
            if value is None:
                continue
            # Split into field + operator (e.g., "age__gte")
            if "__" in field_expr:
                field, op = field_expr.split("__", 1)
            else:
                field, op = field_expr, "eq"

            if not hasattr(model, field):
                continue

            column = getattr(model, field)
            target_type = model.__annotations__.get(field, str)
            converted_value = convert_value(value, target_type)

            # Map operators
            if op == "eq":
                query = query.where(column == converted_value)
            elif op == "ne":
                query = query.where(column != converted_value)
            elif op == "lt":
                query = query.where(column < converted_value)
            elif op == "lte":
                query = query.where(column <= converted_value)
            elif op == "gt":
                query = query.where(column > converted_value)
            elif op == "gte":
                query = query.where(column >= converted_value)
            elif op == "icontains" and isinstance(converted_value, str):
                query = query.where(column.ilike(f"%{converted_value}%"))
            elif op == "contains" and isinstance(converted_value, str):
                query = query.where(column.like(f"%{converted_value}%"))

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
