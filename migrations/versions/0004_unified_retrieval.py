"""Remove the obsolete retrieval-purpose distinction."""

import sqlalchemy as sa
from alembic import op


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "retrieval_queries_type",
        "retrieval_queries",
        type_="check",
    )
    op.drop_column("retrieval_queries", "query_type")


def downgrade() -> None:
    op.add_column(
        "retrieval_queries",
        sa.Column(
            "query_type",
            sa.Text(),
            server_default=sa.text("'answering'"),
            nullable=False,
        ),
    )
    op.alter_column(
        "retrieval_queries",
        "query_type",
        server_default=None,
    )
    op.create_check_constraint(
        "retrieval_queries_type",
        "retrieval_queries",
        "query_type IN ('answering', 'adding')",
    )
