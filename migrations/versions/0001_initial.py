"""Create the FluxMem schema."""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index(
        "sessions_by_user",
        "sessions",
        ["user_id", "session_id"],
    )
    op.create_table(
        "messages",
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("agent_id", sa.Text()),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"]),
        sa.PrimaryKeyConstraint("message_id"),
    )
    op.create_index(
        "messages_history_idx",
        "messages",
        ["session_id", "created_at", "message_id"],
    )
    op.create_table(
        "memories",
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("valid_from", sa.DateTime(timezone=True)),
        sa.Column("valid_to", sa.DateTime(timezone=True)),
        sa.Column("session_applicability", sa.Uuid()),
        sa.CheckConstraint(
            "valid_from IS NULL OR valid_to IS NULL OR valid_to >= valid_from",
            name="memories_valid_interval",
        ),
        sa.ForeignKeyConstraint(["message_id"], ["messages.message_id"]),
        sa.ForeignKeyConstraint(
            ["session_applicability"],
            ["sessions.session_id"],
        ),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_index(
        "memories_by_message",
        "memories",
        ["message_id", "memory_id"],
    )
    op.create_index(
        "memories_by_session_applicability",
        "memories",
        ["session_applicability", "memory_id"],
    )
    op.create_table(
        "memory_lifecycle",
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("decay_class", sa.Text(), nullable=False),
        sa.Column("retention_snapshot", sa.Float(), nullable=False),
        sa.Column("retention_anchor", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "reinforcement_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "use_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("decision_source", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "tier IN ('working', 'short_term', 'long_term')",
            name="memory_lifecycle_tier",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'deleted')",
            name="memory_lifecycle_status",
        ),
        sa.CheckConstraint(
            "importance >= 0.1 AND importance <= 1",
            name="memory_lifecycle_importance",
        ),
        sa.CheckConstraint(
            "decay_class IN ('fast', 'standard', 'slow')",
            name="memory_lifecycle_decay_class",
        ),
        sa.CheckConstraint(
            "retention_snapshot >= 0.1 AND retention_snapshot <= 1",
            name="memory_lifecycle_retention",
        ),
        sa.CheckConstraint(
            "reinforcement_count >= 0 AND use_count >= 0",
            name="memory_lifecycle_counters",
        ),
        sa.CheckConstraint(
            "decision_source IN "
            "('llm_primary', 'llm_retry', 'rules_fallback', 'manual')",
            name="memory_lifecycle_decision_source",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.memory_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_index(
        "memory_lifecycle_by_status",
        "memory_lifecycle",
        ["status", "memory_id"],
    )
    op.create_table(
        "memory_indexes",
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", Vector(1536)),
        sa.Column("search_text", postgresql.TSVECTOR(), nullable=False),
        sa.Column("embedding_model", sa.Text()),
        sa.Column(
            "index_status",
            sa.Text(),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "index_status IN ('ready', 'pending')",
            name="memory_indexes_status",
        ),
        sa.CheckConstraint(
            "(index_status = 'ready' AND embedding IS NOT NULL "
            "AND embedding_model IS NOT NULL) OR "
            "(index_status = 'pending' AND embedding IS NULL "
            "AND embedding_model IS NULL)",
            name="memory_indexes_state",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.memory_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_index(
        "memory_indexes_search_text_gin",
        "memory_indexes",
        ["search_text"],
        postgresql_using="gin",
    )
    op.create_index(
        "memory_indexes_embedding_hnsw",
        "memory_indexes",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_table(
        "retrieval_queries",
        sa.Column("query_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"]),
        sa.PrimaryKeyConstraint("query_id"),
    )
    op.create_index(
        "retrieval_queries_by_session",
        "retrieval_queries",
        ["session_id", "created_at", "query_id"],
    )
    op.create_table(
        "retrieval_candidates",
        sa.Column("query_id", sa.Uuid(), nullable=False),
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.CheckConstraint("rank >= 1", name="retrieval_candidates_rank"),
        sa.CheckConstraint("score >= 0", name="retrieval_candidates_score"),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["retrieval_queries.query_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.memory_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("query_id", "memory_id"),
    )
    op.create_table(
        "memory_usage",
        sa.Column("usage_id", sa.Uuid(), nullable=False),
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("query_id", sa.Uuid(), nullable=False),
        sa.Column("usage_type", sa.Text(), nullable=False),
        sa.Column("rank", sa.Integer()),
        sa.Column("contribution", sa.Float()),
        sa.Column(
            "used_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "usage_type IN ('context_included', 'model_attributed')",
            name="memory_usage_type",
        ),
        sa.CheckConstraint(
            "rank IS NULL OR rank >= 1",
            name="memory_usage_rank",
        ),
        sa.CheckConstraint(
            "contribution IS NULL OR "
            "(contribution >= 0 AND contribution <= 1)",
            name="memory_usage_contribution",
        ),
        sa.ForeignKeyConstraint(
            ["query_id", "memory_id"],
            [
                "retrieval_candidates.query_id",
                "retrieval_candidates.memory_id",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("usage_id"),
        sa.UniqueConstraint(
            "memory_id",
            "query_id",
            name="memory_usage_memory_query_key",
        ),
    )
    op.create_index(
        "memory_usage_by_memory_time",
        "memory_usage",
        ["memory_id", "used_at"],
    )


def downgrade() -> None:
    op.drop_table("memory_usage")
    op.drop_table("retrieval_candidates")
    op.drop_table("retrieval_queries")
    op.drop_table("memory_indexes")
    op.drop_table("memory_lifecycle")
    op.drop_table("memories")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("users")
