"""Canceled jobs, and uploaded pictures.

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"


def upgrade() -> None:
    op.execute("ALTER TABLE jobs DROP CONSTRAINT jobs_status_check")
    op.execute(
        """
        ALTER TABLE jobs ADD CONSTRAINT jobs_status_check
            CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'canceled'))
        """
    )
    op.execute(
        """
        CREATE TABLE uploads (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id text NOT NULL,
            product_id text NOT NULL,
            owner_id text NOT NULL,
            key text NOT NULL,
            bytes integer NOT NULL,
            width integer NOT NULL,
            height integer NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            consumed_at timestamptz
        )
        """
    )
    op.execute("CREATE INDEX uploads_owner_idx ON uploads (tenant_id, product_id, owner_id)")
    op.execute("CREATE INDEX uploads_stale_idx ON uploads (created_at) WHERE consumed_at IS NULL")


def downgrade() -> None:
    op.execute("DROP TABLE uploads")
    op.execute("ALTER TABLE jobs DROP CONSTRAINT jobs_status_check")
    op.execute(
        """
        ALTER TABLE jobs ADD CONSTRAINT jobs_status_check
            CHECK (status IN ('queued', 'running', 'succeeded', 'failed'))
        """
    )
