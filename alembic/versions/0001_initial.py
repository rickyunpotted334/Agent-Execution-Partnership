"""initial

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-20
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tasks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("task_id", sa.String(length=100), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_tasks_task_id", "tasks", ["task_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_tasks_task_id", table_name="tasks")
    op.drop_table("tasks")
