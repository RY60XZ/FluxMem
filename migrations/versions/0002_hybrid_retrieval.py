"""Add hybrid retrieval projections and persisted retrieval provenance."""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "memory_indexes",
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("search_text", postgresql.TSVECTOR(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=True),
        sa.Column(
            "index_status",
            sa.Text(),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
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
    op.execute(
        """
        INSERT INTO memory_indexes (
            memory_id,
            embedding,
            search_text,
            embedding_model,
            index_status,
            indexed_at
        )
        SELECT
            memory_id,
            NULL,
            to_tsvector('english', coalesce(content, '')),
            NULL,
            1,
            'pending',
            now()
        FROM memories
        """
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
        sa.Column("query_type", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "query_type IN ('answering', 'adding')",
            name="retrieval_queries_type",
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
        sa.CheckConstraint(
            "rank >= 1",
            name="retrieval_candidates_rank",
        ),
        sa.CheckConstraint(
            "score >= 0",
            name="retrieval_candidates_score",
        ),
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

    op.add_column(
        "memory_usage",
        sa.Column("query_id", sa.Uuid(), nullable=True),
    )
    _backfill_legacy_usage_queries()
    op.alter_column("memory_usage", "query_id", nullable=False)
    op.create_unique_constraint(
        "memory_usage_memory_query_key",
        "memory_usage",
        ["memory_id", "query_id"],
    )
    op.create_foreign_key(
        "memory_usage_query_candidate_fkey",
        "memory_usage",
        "retrieval_candidates",
        ["query_id", "memory_id"],
        ["query_id", "memory_id"],
        ondelete="CASCADE",
    )
    op.drop_constraint(
        "memory_usage_memory_id_fkey",
        "memory_usage",
        type_="foreignkey",
    )


def _backfill_legacy_usage_queries() -> None:
    """Attach pre-provenance usage rows to synthetic, scoped retrievals."""

    op.execute(
        """
        CREATE TEMPORARY TABLE fluxmem_legacy_usage_queries
        ON COMMIT DROP
        AS
        SELECT
            usage.usage_id,
            gen_random_uuid() AS query_id,
            message.session_id,
            usage.memory_id,
            coalesce(usage.rank, 1) AS rank,
            greatest(coalesce(usage.contribution, 0), 0) AS score,
            usage.used_at
        FROM memory_usage AS usage
        JOIN memories AS memory
          ON memory.memory_id = usage.memory_id
        JOIN messages AS message
          ON message.message_id = memory.message_id
        """
    )
    op.execute(
        """
        INSERT INTO retrieval_queries (
            query_id,
            session_id,
            query_type,
            created_at
        )
        SELECT query_id, session_id, 'answering', used_at
        FROM fluxmem_legacy_usage_queries
        """
    )
    op.execute(
        """
        INSERT INTO retrieval_candidates (
            query_id,
            memory_id,
            rank,
            score
        )
        SELECT query_id, memory_id, rank, score
        FROM fluxmem_legacy_usage_queries
        """
    )
    op.execute(
        """
        UPDATE memory_usage AS usage
        SET query_id = legacy.query_id
        FROM fluxmem_legacy_usage_queries AS legacy
        WHERE legacy.usage_id = usage.usage_id
        """
    )
    op.execute("DROP TABLE fluxmem_legacy_usage_queries")


def downgrade() -> None:
    op.create_foreign_key(
        "memory_usage_memory_id_fkey",
        "memory_usage",
        "memories",
        ["memory_id"],
        ["memory_id"],
        ondelete="CASCADE",
    )
    op.drop_constraint(
        "memory_usage_query_candidate_fkey",
        "memory_usage",
        type_="foreignkey",
    )
    op.drop_constraint(
        "memory_usage_memory_query_key",
        "memory_usage",
        type_="unique",
    )
    op.drop_column("memory_usage", "query_id")
    op.drop_table("retrieval_candidates")
    op.drop_index(
        "retrieval_queries_by_session",
        table_name="retrieval_queries",
    )
    op.drop_table("retrieval_queries")
    op.drop_index(
        "memory_indexes_embedding_hnsw",
        table_name="memory_indexes",
        postgresql_using="hnsw",
    )
    op.drop_index(
        "memory_indexes_search_text_gin",
        table_name="memory_indexes",
        postgresql_using="gin",
    )
    op.drop_table("memory_indexes")
