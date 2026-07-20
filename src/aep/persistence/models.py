from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from aep.persistence.db import Base


class TaskModel(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ActionModel(Base):
    __tablename__ = "actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    task_id: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(50), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class AuditEventModel(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
    digest: Mapped[str] = mapped_column(String(128), index=True)
    previous_digest: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ExperimentModel(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hypothesis: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
