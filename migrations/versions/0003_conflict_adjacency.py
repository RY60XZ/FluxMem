"""Add canonical undirected potential-conflict adjacency."""

import sqlalchemy as sa
from alembic import op


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_conflicts",
        sa.Column("memory_a_id", sa.Uuid(), nullable=False),
        sa.Column("memory_b_id", sa.Uuid(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "memory_a_id < memory_b_id",
            name="memory_conflicts_canonical_order",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="memory_conflicts_confidence",
        ),
        sa.ForeignKeyConstraint(
            ["memory_a_id"],
            ["memories.memory_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["memory_b_id"],
            ["memories.memory_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("memory_a_id", "memory_b_id"),
    )
    op.create_index(
        "memory_conflicts_by_b",
        "memory_conflicts",
        ["memory_b_id", "memory_a_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "memory_conflicts_by_b",
        table_name="memory_conflicts",
    )
    op.drop_table("memory_conflicts")
